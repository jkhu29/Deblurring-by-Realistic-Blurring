import os
import argparse

import cv2
import tfrecord
import numpy as np


class DataMaker(object):
    def __init__(self, opt) -> None:
        super().__init__()
        self.tf_path = opt.output_path
        self.size_image = opt.size_image
        self.stride = opt.stride
        self.data_path = opt.data_path
        self.mode = opt.mode

    def make_data(self):
        print('begin to save deblur data file to %s' % self.tf_path)
        self.writer = tfrecord.TFRecordWriter(self.tf_path)
        if self.mode == "defocus":
            self._make_defocus_image()
        elif "motion" in self.mode:
            if self.mode == "motion-cycle":
                self._make_single_image_cycle_blur()
            elif self.mode == "motion-deblur":
                self._make_single_image_deblur()
        else:
            raise IndexError("mode should in defocus | motion")
        self.writer.close()
        print('saved')

    def _make_defocus_image(self):
        for img_name in os.listdir(self.data_path):
            img = cv2.imread(os.path.join(self.data_path, img_name))
            gt = cv2.imread(self.data_path.replace("Image", "GT"), img_name, 0)
            height, width = gt.shape
            self.writer.write(
                {
                    "image": (img.transpose(2, 0, 1).tobytes(), "byte"),
                    "gt": (gt.tobytes(), "btye"),
                    "width": (width, "int"),
                    "height": (height, "int"),
                }
            )

    def _make_single_image_deblur(self):
        for img_name in os.listdir(self.data_path):
            img = cv2.imread(os.path.join(self.data_path, img_name))
            for x in np.arange(0, img.shape[0] - self.size_image + 1, self.stride):
                for y in np.arange(0, img.shape[1] - self.size_image + 1, self.stride):
                    img_part = img[int(x): int(x + self.size_image),
                                   int(y): int(y + self.size_image)]
                    if (img_part.shape[0] != self.size_image) or (np.all(img_part == 0)) or (np.all(img_part == 255)):
                        continue
                    self.writer.write(
                        {
                            "image": (img_part.transpose(2, 0, 1).tobytes(), "byte"),
                            "size": (self.size_image, "int"),
                        }
                    )

    def _make_single_image_cycle_blur(self, blur_path, sharp_path):
        blur_names = os.listdir(blur_path)
        sharp_names = os.listdir(sharp_path)
        length = min(len(blur_names), len(sharp_names))
        for i in range(length):
            blur_name = blur_names[i]
            sharp_name = sharp_names[i]
            blur = cv2.imread(os.path.join(blur_path, blur_name))
            sharp = cv2.imread(os.path.join(sharp_path, sharp_name))

            for x in np.arange(0, blur.shape[0] - self.size_image + 1, self.stride):
                for y in np.arange(0, blur.shape[1] - self.size_image + 1, self.stride):
                    blur_part = blur[int(x): int(x + self.size_image),
                                     int(y): int(y + self.size_image)]
                    sharp_part = sharp[int(x): int(x + self.size_image),
                                       int(y): int(y + self.size_image)]
                    if (sharp_part.shape[0] != self.size_image) or (np.all(sharp_part == 0)) or (np.all(blur_part == 0)) or (np.all(sharp_part == 255)) or (np.all(blur_part == 255)):
                        continue
                    self.writer.write(
                        {
                            "blur": (blur_part.transpose(2, 0, 1).tobytes(), "byte"),
                            "sharp": (sharp_part.transpose(2, 0, 1).tobytes(), "byte"),
                            "size": (self.size_image, "int"),
                        }
                    )


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    data_maker = DataMaker()
    opt = get_options()
    data_maker.make_data(opt)
