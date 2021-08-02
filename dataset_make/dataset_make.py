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


def make_blur(data_path, h5_path, size_image=128, stride=100):
    # TODO(jkhu29): data amplification, like: totate random angle
    imgs_blur = []
    imgs_sharp = []
    for img_name in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, img_name))
        noise = utils.concat_noise(img, (4, size_image, size_image), img.shape[0])

        for x in np.arange(0, img.shape[0] - size_image + 1, stride):
            for y in np.arange(0, img.shape[1] - size_image + 1, stride):
                img_sharp = img[int(x): int(x + size_image),
                                int(y): int(y + size_image)]
                img_blur = noise[int(x): int(x + size_image),
                                 int(y): int(y + size_image)]
                imgs_sharp.append(img_sharp.transpose(2, 0, 1))
                imgs_blur.append(img_blur.transpose(2, 0, 1))

    print('begin to save blur data file to %s' % h5_path)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('sharp', data=np.array(imgs_sharp, dtype=np.float32))
        f.create_dataset('blur', data=np.array(imgs_blur, dtype=np.float32))
    print('saved')


def make_deblur(data_path, h5_path, size_image=128, stride=100):
    imgs = []
    for img_name in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, img_name))

        for x in np.arange(0, img.shape[0] - size_image + 1, stride):
            for y in np.arange(0, img.shape[1] - size_image + 1, stride):
                img_part = img[int(x): int(x + size_image),
                               int(y): int(y + size_image)]
                imgs.append(img_part.transpose(2, 0, 1))

    print('begin to save deblur data file to %s' % h5_path)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('sharp', data=np.array(imgs, dtype=np.float32))
    print('saved')


def make_cycle_blur(data1_path, data2_path, h5_path, size_image=128, stride=100):
    blur_imgs = []
    sharp_imgs = []

    length = 0
    for img_name in os.listdir(data1_path):
        img = cv2.imread(os.path.join(data1_path, img_name))
        length += 1

        for x in np.arange(0, img.shape[0] - size_image + 1, stride):
            for y in np.arange(0, img.shape[1] - size_image + 1, stride):
                img_part = img[int(x): int(x + size_image),
                               int(y): int(y + size_image)]
                blur_imgs.append(img_part.transpose(2, 0, 1))

    img2_names = os.listdir(data2_path)

    for i in range(length):
        img_name = img2_names[i]
        img = cv2.imread(os.path.join(data2_path, img_name))

        for x in np.arange(0, img.shape[0] - size_image + 1, stride):
            for y in np.arange(0, img.shape[1] - size_image + 1, stride):
                img_part = img[int(x): int(x + size_image),
                               int(y): int(y + size_image)]
                sharp_imgs.append(img_part.transpose(2, 0, 1))

    print(len(blur_imgs), len(sharp_imgs))

    print('begin to save blur data file to %s' % h5_path)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('blur', data=np.array(blur_imgs, dtype=np.float32))
        f.create_dataset('sharp', data=np.array(sharp_imgs, dtype=np.float32))
    print('saved')



def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--blur_train_path', type=str, default="./blur_train.h5")
    parser.add_argument('--blur_valid_path', type=str, default="./blur_valid.h5")
    parser.add_argument('--blur_train_data', type=str, default="./gopro/train")
    parser.add_argument('--blur_valid_data', type=str, default="./gopro/test")
    parser.add_argument('--deblur_train_path', type=str, default="./deblur_train.h5")
    parser.add_argument('--deblur_valid_path', type=str, default="./deblur_valid.h5")
    parser.add_argument('--deblur_train_data', type=str, default="./gopro/train")
    parser.add_argument('--deblur_valid_data', type=str, default="./gopro/test")
    parser.add_argument('--size_image', type=int, default=128, help='the size of output image, default=128*128')
    parser.add_argument('--stride', type=int, default=100, help='stride when making dataset, default=100')
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = get_options()

    blur_train_path = opt.blur_train_path
    deblur_train_path = opt.deblur_train_path
    blur_valid_path = opt.blur_valid_path
    deblur_valid_path = opt.deblur_valid_path
    blur_train_data = opt.blur_train_data
    deblur_train_data = opt.deblur_train_data
    blur_valid_data = opt.blur_valid_data
    deblur_valid_data = opt.deblur_valid_data

    mode = opt.mode

    if mode == "train":
    # for train.py
        make_blur(
            blur_train_data, blur_train_path,
            size_image=opt.size_image, stride=opt.stride
            )
        make_blur(
            blur_valid_data, blur_valid_path,
            size_image=opt.size_image, stride=opt.stride
            )
        make_deblur(
            deblur_train_data, deblur_train_path,
            size_image=opt.size_image, stride=opt.stride
            )
        make_deblur(
            deblur_valid_data, deblur_valid_path,
            size_image=opt.size_image, stride=opt.stride
            )
    elif mode == "train_blur":
    # for train_blur.py
        make_cycle_blur(
                blur_train_data, deblur_train_data,
                blur_train_path, 
                size_image=opt.size_image, stride=opt.stride
            )
        make_cycle_blur(
                blur_valid_data, deblur_valid_data,
                blur_valid_path, 
                size_image=opt.size_image, stride=opt.stride
            )
    elif mode == "train_deblur":
    # for train_deblur.py
        make_deblur(
            deblur_train_data, deblur_train_path
        )
        make_deblur(
            deblur_valid_data, deblur_valid_path
        )
    else:
        raise("please set mode in 'train'; 'train_blur'; 'train_deblur'")