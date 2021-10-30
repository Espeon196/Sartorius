import pandas as pd
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

import glob
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import pydicom

def get_train_transforms(img_size):
    return Compose([
        # RandomResizedCrop(img_size, img_size),
        # RandomCrop(img_size, img_size),
        #Resize(img_size, img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        #ShiftScaleRotate(p=1.0),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        #Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms(img_size):
    return Compose([
        # CenterCrop(img_size, img_size, p=1.),
        #Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_inference_transforms(img_size):
    return Compose([
        # RandomResizedCrop(img_size, img_size),
        # RandomCrop(img_size, img_size),
        #Resize(img_size, img_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=1.0),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)

class BrainTumor2dSimpleDataset(Dataset):
    def __init__(self, df, data_path, img_size, transforms=None, output_label=True):
        self.paths = df["BraTS21ID"].values
        self.output_label = output_label
        if self.output_label:
            self.targets = df["MGMT_value"].values
        self.data_path = data_path
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = os.path.join(self.data_path, str(_id).zfill(5))
        channels = []
        for t in ("FLAIR", "T1w", "T1wCE"):
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, '*')),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            x = len(t_paths)
            if x < 10:
                r = range(x)
            else:
                d = x // 10
                r = range(d, x-d, d)
            channel = []
            for i in r:
                channel.append(cv2.resize(load_dicom(t_paths[i]), (self.img_size, self.img_size))/ 255)
            channel = np.mean(channel, axis=0)
            channels.append(channel)
        channels = np.array(channels).transpose(1, 2, 0).astype(np.float32)
        #print(channels.shape)

        if self.transforms:
            channels = self.transforms(image=channels)['image']

        if self.output_label:
            target = torch.tensor(self.targets[index], dtype=torch.float)
            return channels, target
        else:
            return channels

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

if __name__ == "__main__":
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
    INPUT_DIR = os.path.join(SRC_DIR, 'input')
    df = pd.read_csv(os.path.join(INPUT_DIR, 'train_labels.csv'))
    ds = BrainTumor2dSimpleDataset(df, data_path=os.path.join(INPUT_DIR, 'train'), img_size=256, transforms=get_train_transforms(256))
    print("Dataset size: {}".format(len(ds)))
    ds_iter = iter(ds)
    for i, (channels, target) in enumerate(ds_iter):
        print("[{}]channels shape: {}, target: {}".format(i, channels.shape, target))

    ds = BrainTumor2dSimpleDataset(df, data_path=os.path.join(INPUT_DIR, 'train'), img_size=256, transforms=None)
    ds_iter = iter(ds)
    for i, (channels, target) in enumerate(ds_iter):
        for j in range(3):
            plt.subplot(1, 3, j+1)
            plt.imshow(channels[:, :, j], cmap='gray')
        plt.show()