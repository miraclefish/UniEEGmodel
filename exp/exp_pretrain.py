from data_provider.data_factory import data_provider
from utils.tools import cosine_scheduler
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.losses import UnifiedMaskRecLoss
from utils.dataloader import BalancedDataLoaderIterator
from utils.ddp import is_main_process, get_world_size

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

import os
import time
import warnings
import numpy as np
import yaml
import wandb
import importlib
import sys
import json

warnings.filterwarnings('ignore')

def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']
    return wrapper


# replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list


def init_and_merge_datasets(data_loader_list, logger=None):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    if logger:
        logger.info(f"data loader length: {dataloader.length_list}")
        logger.info(f"max dataloader length: {dataloader.max_length}")
        logger.info(f"epoch iteration: {dataloader.max_length * dataloader.num_dataloaders}")
    train_steps = dataloader.__len__()

    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args, logger=None):
        super(Exp_All_Task, self).__init__()

        self.args = args
        self.logger = logger
        self.task_data_config = read_task_data_config(
            self.args.task_data_config_path)
        self.task_data_config_list = get_task_data_config_list(
            self.task_data_config, default_batch_size=self.args.batch_size)
        if args.ddp:
            device_id = dist.get_rank() % torch.cuda.device_count()
        else:
            device_id = args.device

        if self.logger:
            self.logger.info(f"this device_id: {device_id}")
        self.device_id = device_id

    def _build_model(self, ddp=False):
        if ddp:
            ddp = self.args.ddp
        else:
            ddp = ddp
        module = importlib.import_module("models."+self.args.model)
        model = module.Model(
            self.args, self.task_data_config_list, pretrain=True).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device_id], find_unused_parameters=True)
        return model.to(self.device_id)

    def _get_data(self, flag):
        ddp = self.args.ddp
        data_set_list = []
        data_loader_list = []
        for task_data_name, task_config in self.task_data_config.items():
            # if task_config['data'] == 'UEA' and flag == 'val':
            #     # TODO strange that no val set is used for classification. Set to test set for val
            #     flag = 'test'
            data_set, data_loader = data_provider(
                self.args, task_config, flag, ddp=ddp)

            if self.logger:
                self.logger.info(f"loading dataset: {task_data_name} | {flag} | value_interval: {data_set.value_interval} | data length: {len(data_set)}")
            data_set_list.append(data_set)
            data_loader_list.append(data_loader)
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        if self.args.ddp:
            world_size = get_world_size()
        else:
            world_size = 1
        eff_batch_size = self.args.batch_size * self.args.acc_it * world_size
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        self.real_learning_rate = real_learning_rate

        if self.logger:
            self.logger.info(f"base lr: {self.args.learning_rate * 32 / eff_batch_size:.2e}")
            self.logger.info(f"actual lr: {real_learning_rate:.2e}")
            self.logger.info(f"accumulate grad iterations: {self.args.acc_it}")
            self.logger.info(f"effective batch size: {eff_batch_size}")

        model_optim = optim.Adam(self.model.parameters(
        ), lr=real_learning_rate, betas=(0.9, self.args.beta2), weight_decay=self.args.weight_decay, eps=self.args.eps)
        return model_optim

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        if self.args.ddp:
            torch.cuda.synchronize()
            dist.barrier()

        # Data loader
        _, train_loader_list = self._get_data(flag='train')
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list, self.logger)

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle)
            torch.cuda.empty_cache()

        if self.args.ddp:
            torch.cuda.synchronize()
            dist.barrier()

        # Model
        self.model = self._build_model()

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        if self.logger:
            self.logger.info(f"Parameters number {pytorch_total_params/1e6:.3f} M")
            self.logger.info(f"{train_steps} steps for each epoch")

        # Optimizer
        model_optim = self._select_optimizer()
        lr_schedule = cosine_scheduler(
            self.real_learning_rate,
            self.args.min_lr,
            self.args.train_epochs, train_steps,
            warmup_epochs=self.args.warmup_epochs,
        )

        # Loss
        criterion = UnifiedMaskRecLoss().to(self.device_id)
        scaler = NativeScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion, epoch, train_steps, scaler, lr_schedule)

            if is_main_process():
                self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Avg Train Loss: {train_loss:.7f}")
                if self.args.wandb:
                    wandb.log({'train_loss_avg': train_loss})

                save_dict = {
                    'student': self.model.state_dict(),
                    'optimizer': model_optim.state_dict(),
                    'epoch': epoch + 1,
                    'args': self.args,
                }

                torch.save(save_dict, path + '/' + 'pretrain_checkpoint.pth')

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion, epoch, train_steps, scaler, lr_schedule):
        #一次epoch训练完所有的数据集，每个数据集出现一次batch
        current_device = torch.cuda.current_device()
        train_loss_set = []

        acc_it = self.args.acc_it
        max_norm = self.args.clip_grad
        min_keep_ratio = self.args.min_keep_ratio

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum_display = 0

        loss = 0
        # 外循环训练不同数据集,每次采样不同数据集的一个batch
        for i, (sample_init, task_id) in enumerate(data_loader_cycle):
            it = train_steps * epoch + i
            for _, param_group in enumerate(model_optim.param_groups):
                param_group["lr"] = lr_schedule[it]

            # Get batch data based on the real batch size of each task: avoid OOM for large samples
            task_name = self.task_data_config_list[task_id][1]['task_name']
            dataset_name = self.task_data_config_list[task_id][1]['dataset']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']

            # 根据设备能够承受的最大batchsize，将一个batch分成多个小batch
            sample_list = self.get_multi_source_data(
                sample_init, task_name, small_batch_size, min_keep_ratio=min_keep_ratio)
            len_sample_list = len(sample_list)

            # Accumulate gradients of mulitple samples
            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                x_enc, x_mark_enc, pad_mask = sample
                with torch.cuda.amp.autocast():
                    model_output = self.model(
                        x_enc=x_enc, x_mark_enc=x_mark_enc, task_id=task_id, task_name=task_name, enable_mask=True)
                loss_dict = criterion(model_output, x_enc)
                loss = loss_dict['loss']
                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)

            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            norm_value = scaler(loss, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))

            if (i+1) % acc_it == 0:
                model_optim.zero_grad()

            if self.args.ddp:
                torch.cuda.synchronize()

            loss_sum_display += loss_display

            # release memory to avoid OOM
            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            if is_main_process():
                wandb_loss_dict = {
                    'norm': norm_value if norm_value is not None else 0.,
                    f'train_mask_loss_{self.task_data_config_list[task_id][0]}': loss_dict['mask_loss'].item(),
                    "loss_avg": loss_sum_display/(i+1)
                }
                if self.args.wandb:
                    wandb.log(wandb_loss_dict)

            if (i + 1) % 100 == 0 and is_main_process():
                self.logger.info(f"\titers: {i + 1}, epoch: {epoch + 1} | lr: {lr_schedule[it]:.8f} | loss_avg: {loss_sum_display/(i+1):.5f} | current_loss: {loss.item() * acc_it:.5f} | current_data: {dataset_name}")

        if is_main_process():
            self.logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
        train_loss = np.average(train_loss_set)

        return train_loss

    def get_multi_source_data(self, this_batch, task_name, small_batch_size, min_keep_ratio=None):
        """
        Splits the input batch into smaller batches based on the specified small_batch_size.

        Args:
            this_batch (tuple): The input batch containing all data of a task.
            task_name (str): The name of the task.
            small_batch_size (int): The size of the smaller batches to split the data into.
            min_keep_ratio (float, optional): The minimum ratio of data to keep in each smaller batch.

        Returns:
            list: A list of tuples, where each tuple contains a smaller batch of data, marks, and padding masks.
        """

        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]

        if "long_term_forecast" in task_name:
            batch_x, _, batch_x_mark, _ = this_batch
            batch_x = batch_x.float().to(self.device_id)
            batch_x_mark = batch_x_mark.float().to(self.device_id)
            batch_x_mark = batch_x_mark.max(dim=-1)[0]
            padding_mask = torch.ones(
                (batch_x.shape[0], batch_x.shape[1]), dtype=torch.bool).to(self.device_id)
        elif "classification" in task_name:
            batch_x, _, padding_mask = this_batch
            batch_x = batch_x.float().to(self.device_id)
            batch_x_mark = padding_mask.float().to(self.device_id)
            padding_mask = batch_x_mark.bool().to(self.device_id)
        elif 'EEG' in task_name:
            batch_x, _ = this_batch
            batch_x = batch_x.float().to(self.device_id)
            batch_x_mark = torch.ones(
                (batch_x.shape[0], batch_x.shape[1],batch_x.shape[2]), dtype=torch.bool).to(self.device_id)
            padding_mask = torch.ones(
                (batch_x.shape[0], batch_x.shape[1]), dtype=torch.bool).to(self.device_id)
        if min_keep_ratio is not None:
            keep_ratios = torch.rand(
                1, device=batch_x.device) * (1.0 - min_keep_ratio) + min_keep_ratio
            L = batch_x.shape[1]
            len_keeps = (L * keep_ratios).long()
            len_keeps = (torch.ceil(len_keeps/self.args.patch_len)
                         )*self.args.patch_len
            len_keeps = len_keeps.int()

            batch_x = batch_x[:, :len_keeps]
            batch_x_mark = batch_x_mark[:, :len_keeps]
            padding_mask = padding_mask[:, :len_keeps]

        split_batch_x = split_tensor(batch_x, small_batch_size)
        split_batch_x_mark = split_tensor(batch_x_mark, small_batch_size)
        split_padding_mask = split_tensor(padding_mask, small_batch_size)

        return list(zip(split_batch_x, split_batch_x_mark, split_padding_mask))

    def memory_check(self, data_loader_cycle, holdout_memory=6):
        """
        Checks the memory usage of the model by gradually increasing the batch size until it reaches the maximum batch size that can be supported without running out of memory.

        Args:
            data_loader_cycle (DataLoaderCycle): The data loader cycle object.
            holdout_memory (int): The amount of memory (in GB) to hold out for other operations.

        Returns:
            None
        """
        num_elements = holdout_memory * 1024 * 1024 * 1024 // 4
        extra_mem = torch.empty(
            num_elements, dtype=torch.float32, device=self.device_id)

        for data_loader_id in range(data_loader_cycle.num_dataloaders):
            batch_size = 1
            max_batch_size = 0

            if self.args.ddp:
                torch.cuda.synchronize()

            model_tmp = self._build_model(ddp=False)
            model_tmp.train()
            model_tmp.zero_grad(set_to_none=True)
            while True:
                try:
                    sample, task_id = data_loader_cycle.generate_fake_samples_for_batch(
                        data_loader_id, batch_size)
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    dataset_name = self.task_data_config_list[task_id][1]['dataset']
                    if "long_term_forecast" in task_name:
                        batch_x, _, batch_x_mark, _ = sample
                        batch_x = batch_x.float().to(self.device_id)
                        batch_x_mark = batch_x_mark.float().to(self.device_id)
                    elif "classification" in task_name:
                        batch_x, _, batch_x_mark = sample
                        batch_x = batch_x.float().to(self.device_id)
                        batch_x_mark = torch.ones(
                            (batch_x.shape[0], batch_x.shape[1]), dtype=torch.bool).to(self.device_id)
                    elif 'EEG' in task_name:
                        batch_x, _ = sample
                        batch_x = batch_x.float().to(self.device_id)
                        batch_x_mark = torch.ones(
                            (batch_x.shape[0], batch_x.shape[1],batch_x.shape[2]), dtype=torch.bool).to(self.device_id)
                    if self.logger:
                        self.logger.info(f"task_id: {task_id}, dataset_name: {dataset_name}, sample_shape: {sample[0].shape}, max_batch_size: {max_batch_size}")
                    with torch.cuda.amp.autocast():
                        model_output = model_tmp(
                            x_enc=batch_x, x_mark_enc=batch_x_mark, task_id=task_id, task_name=task_name, enable_mask=True)
                    loss = 0.0
                    #直接进行backward，测试最大训练内存容量
                    for each in model_output:
                        if each is not None:
                            #如果是list，就对每个元素求和
                            if isinstance(each, list):
                                for item in each:
                                    loss += item.sum()
                            else:
                                loss += each.sum()

                    loss.backward()
                    max_batch_size = batch_size
                    batch_size *= 2

                    if max_batch_size >= self.args.batch_size:
                        if self.logger:
                            self.logger.info(f"can support default batchsize: {self.args.batch_size}, {max_batch_size}")
                            self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                            self.task_data_config_list[task_id][1]['checkpointing'] = False
                        break

                except Exception as e:
                    if self.logger:
                        self.logger.info(f"An exception occurred: {e}")
                        self.logger.info(f"task_id: {task_id}, dataset_name: {dataset_name}, max_batch_size: {max_batch_size}")
                        self.logger.info(f"cannot support default batchsize: {self.args.batch_size}, {max_batch_size}")

                    self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                    del model_tmp
                    torch.cuda.empty_cache()
                    break
        del extra_mem
        try:
            del model_tmp
        except:
            pass
        torch.cuda.empty_cache()
        text = "Checking all task data config: \n"
        text += json.dumps(self.task_data_config_list, indent=4)
        if self.logger:
            self.logger.info(text)
        return
