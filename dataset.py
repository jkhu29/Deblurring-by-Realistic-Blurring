import h5py
import numpy as np
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, h5_file):
        super(BasicDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['blur'][idx], f['sharp'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['blur'])


class BasicDatasetDeblur(Dataset):
    def __init__(self, h5_file):
        super(BasicDatasetDeblur, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['sharp'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['sharp'])

