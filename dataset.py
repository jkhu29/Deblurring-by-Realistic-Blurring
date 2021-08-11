import h5py
import numpy as np
from torch.utils.data import Dataset

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline


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


class BasicDALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, size, batch_size, output_map=["blur", "sharp"], auto_reset=True, onehot_label=False):
        super(BasicDALIDataloader, self).__init__()
        self.size = size
        self.batch_size = batch_size
        self.output_map = output_map

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        data = super().__next__()[0]
        return [data[self.output_map[0]], data[self.output_map[1]]]
    
    def __len__(self):
        if self.size % self.batch_size==0:
            return self.size // self.batch_size
        else:
            return self.size // self.batch_size + 1


class HybridTrainCycleGANPipe(Pipeline):
    """HybridTrainCycleGANPipe"""
    def __init__(self, batch_size, data_dir):
        super(HybridTrainCycleGANPipe, self).__init__()
        self.iterator = iter(CIFAR_INPUT_ITER(batch_size, "train", root=data_dir))
        dali_device = "gpu"
        self.input_blur = ops.ExternalSource()
        self.input_sharp = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
        )
        self.coin = ops.CoinFlip(probability=0.5)

    def _iter_setup(self):
        (blur, sharp) = self.iterator.next()
        self.feed_input(self.blurs, blur, layout="HWC")
        self.feed_input(self.sharps, sharp, layout="HWC")

    def _define_graph(self):
        rng = self.coin()
        self.blurs = self.input_blur()
        self.sharps = self.input_sharp()
        # blur
        blur = self.blurs
        blur = self.pad(blur.gpu())
        blur = self.crop(blur, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        blur = self.cmnp(blur, mirror=rng)
        # sharp
        sharp = self.sharps
        sharp = self.pad(sharp.gpu())
        sharp = self.crop(sharp, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        sharp = self.cmnp(sharp, mirror=rng)
        return [blur, sharp]
        