from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, GLUONTSDataset, EEGDataset, TUARDataset
from data_provider.uea import collate_fn
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'TUH': EEGDataset,
    'BCICIV': EEGDataset,
    'TUAR': TUARDataset,
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # # 'm4': Dataset_M4,  Removed due to the LICENSE file constraints of m4.py
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    # # datasets from gluonts package:
    # "gluonts": GLUONTSDataset,
}


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def data_provider(args, config, flag, ddp=False):  # args,
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        stride_size = config['window_size']
        batch_size = 1  # bsz=1 for evaluation
    else:
        stride_size = config['stride_size']
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    if 'EEG' in config['task_name']:

        data_set = Data(
            root_path=config['root_path'],
            dataset_name=config['dataset_name'],
            window_size=config['window_size'],
            stride_size=stride_size,
            flag=flag,
            try_run=args.try_run,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
