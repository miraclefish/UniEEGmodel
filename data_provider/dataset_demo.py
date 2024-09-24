import torch
import tqdm

from data_loader import EEGDataset
from data_loader import TUARDataset
from pathlib import Path
import h5py

if __name__ == '__main__':

    # file = Path('/dataYYF/YYF/EEGdata/TUAR.hdf5')
    # infile = h5py.File(str(file), 'r')
    # subjects = [i for i in infile]

    data_path = Path('/dataYYF/YYF/UniEEGmodel/data/TUAR.hdf5')
    # train_set_path = Path('/root/autodl-fs/TUAR_train.hdf5')
    # train_set = TUARDataset(
    #     root_path=data_path,
    #     dataset_name='TUAR',
    #     window_size=2048,
    #     stride_size=2048,
    #     task='multi_cls',
    #     # try_run=True,
    #     flag='train'
    # )
    #
    # count = torch.zeros(6)
    # L = 0
    # for i in tqdm.tqdm(range(len(train_set)), desc='train set'):
    #     data, label = train_set[i]
    #     label_count = label.sum(dim=0).sum(dim=0)
    #     count += label_count
    #     L += data.shape[0] * data.shape[1]
    #
    # for n, name in zip(count, ['Arti', 'eyem', 'chew', 'shiv', 'musc', 'elec']):
    #     print(f"{name}: {n.item()/L:.4f} | {n.item()}")

    test_set = TUARDataset(
        root_path=data_path,
        dataset_name='TUAR',
        window_size=2048,
        stride_size=2048,
        task='multi_cls',
        # try_run=True,
        flag='test'
    )

    count = torch.zeros(6)
    L = 0
    for i in tqdm.tqdm(range(len(test_set)), desc='test set'):
        data, label, _, _ = test_set[i]
        label_count = label.sum(dim=0).sum(dim=0)
        count += label_count
        L += data.shape[0] * data.shape[1]

    for n, name in zip(count, ['Arti', 'eyem', 'chew', 'shiv', 'musc', 'elec']):
        print(f"{name}: {n.item() / L:.4f} | {n.item()}")


