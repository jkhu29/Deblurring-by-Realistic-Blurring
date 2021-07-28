import argparse
import copy
import os

import cv2
import h5py
import numpy as np
import torch

import sys
sys.path.append("..")

import utils


def make_h5(data_path, h5_path, size_image=128, stride=10):
    # TODO(jkhu29): data amplification, like: totate random angle
    imgs_blur = []
    imgs_sharp = []
    for img_name in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, img_name)) / 127.5
        noise = utils.concat_mix(img, (4, size_image, size_image), img.shape[0])

        for x in np.arange(0, img.shape[0] - size_image + 1, stride):
            for y in np.arange(0, img.shape[1] - size_image + 1, stride):
                img_sharp = img[int(x): int(x + size_image),
                                int(y): int(y + size_image)]
                img_blur = noise[int(x): int(x + size_image),
                                 int(y): int(y + size_image)]
                imgs_blur.append(img_blur.transpose(2, 0, 1))
                imgs_sharp.append(img_sharp.transpose(2, 0, 1))

    print('begin to save h5 file to %s' % h5_path)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('blur', data=np.array(imgs_blur, dtype=np.float32))
        f.create_dataset('sharp', data=np.array(imgs_sharp, dtype=np.float32))
    print('saved')


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--size_image', type=int, default=128, help='the size of output image, default=128*128')
    parser.add_argument('--stride', type=int, default=10, help='stride when making dataset, default=10')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_options()

    train_path = opt.train_path
    valid_path = opt.valid_path
    train_data = opt.train_path
    valid_data = opt.valid_data

    make_h5(
        train_data, train_path,
        size_image=opt.size_image, stride=opt.stride
        )
    make_h5(
        valid_data, valid_path,
        size_image=opt.size_image, stride=opt.stride
        )
