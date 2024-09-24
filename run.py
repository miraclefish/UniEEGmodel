import argparse
import torch
from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
import random
import numpy as np
import wandb
import logging
import datetime
import os
from utils.ddp import is_main_process, init_distributed_mode


def get_logger(log_path):
    logger = logging.getLogger('Sup')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建一个handler用于写入日志文件
    fh = logging.FileHandler(log_path + 'pretrain.log', mode='w')
    fh.setLevel(logging.INFO)  # 设置handler的日志级别
    fh.setFormatter(formatter)  # 设置handler的格式

    # 再创建一个handler用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 设置handler的日志级别
    ch.setFormatter(formatter)  # 设置handler的格式

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTS supervised training')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='TUAR',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, default='UniEEG',
                        help='model name')
    parser.add_argument('--try_run', action='store_true', help='try run')

    # data loader
    parser.add_argument('--data', type=str, required=False,
                        default='All', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--task_data_config_path', type=str,
                        default='data_provider/multi_task.yaml', help='root path of the task and data yaml file')
    parser.add_argument('--subsample_pct', type=float,
                        default=None, help='subsample percent')
    # device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # ddp
    parser.add_argument('--ddp', action='store_true', help='whether to use ddp')
    parser.add_argument('--local-rank', type=int, help='local rank')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='data loader num workers')
    parser.add_argument("--memory_check", action="store_true", default=True)
    parser.add_argument("--large_model", action="store_true", default=True)

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument("--prompt_tune_epoch", type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--acc_it', type=int, default=1,
                        help='acc iteration to enlarge batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='optimizer min learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float,
                        default=None, help='optimizer layer decay')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--lradj', type=str,
                        default='supervised', help='adjust learning rate')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='save location of model checkpoints')
    parser.add_argument('--pretrained_weight', type=str, default='/dataYYF/YYF/UniEEGmodel/checkpoints/ALL_UniEEG_all_dm512_el8_Exp_07_26_11_53_58/pretrain_checkpoint.pth',
                        help='location of pretrained model checkpoints')
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--debug', type=str,
                        default='enabled', help='disabled')
    parser.add_argument('--project_name', type=str,
                        default='tsfm-multitask', help='wandb project name')

    # model settings
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=8,
                        help='num of encoder layers')
    parser.add_argument("--share_embedding",
                        action="store_true", default=False)
    parser.add_argument("--patch_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--prompt_num", type=int, default=8)
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')
    parser.add_argument('--mode_debug', type=bool, default=False, help='whether to debug')

    # task related settings
    # forecasting task
    parser.add_argument('--inverse', action='store_true',
                        help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=1.0, help='prior anomaly ratio (%)')

    # zero-shot-forecast-new-length
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument('--zero_shot_forecasting_new_length',
                        type=str, default=None, help='unify')

    args = parser.parse_args()
    if args.ddp:
        init_distributed_mode(args)
    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)

    date_str = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
    exp_name = f'{args.task_name}_{args.model}_{args.data}_' \
               f'dm{args.d_model}_el{args.e_layers}_{args.des}_{date_str}'
    log_path = args.checkpoints + 'supervise/' + exp_name + '/'

    if is_main_process():
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if args.wandb:
            wandb.init(
                name=exp_name,
                # set the wandb project where this run will be logged
                project=args.project_name,
                # track hyperparameters and run metadata
                config=args,
                mode=args.debug,
            )

        logger = get_logger(log_path)
        logger.info('Args in experiment:')
        logger.info(args)

        if int(args.prompt_tune_epoch) != 0:
            exp_name = 'Tune'+str(args.prompt_tune_epoch)+'_'+exp_name
            logger.info(exp_name)
    else:
        logger = None

    Exp = Exp_All_Task_SUP

    if args.is_training:
        # setting record of experiments
        setting = exp_name

        exp = Exp(args, logger)  # set experiments
        if logger:
            logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_dm{}_el{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.d_model,
            args.e_layers,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, load_pretrain=True)

