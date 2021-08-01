import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['blur'][idx], f['sharp'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['blur'])


class ValidDataset(Dataset):
    def __init__(self, h5_file):
        super(ValidDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['blur'][idx], f['sharp'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['blur'])


class TrainDatasetDeblur(Dataset):
    def __init__(self, h5_file):
        super(TrainDatasetDeblur, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['sharp'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['sharp'])


class ValidDatasetDeblur(Dataset):
    def __init__(self, h5_file):
        super(ValidDatasetDeblur, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['sharp'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['sharp'])