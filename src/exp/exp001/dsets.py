import collections

import pandas as pd
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, BboxParams
)

from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

import glob
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt

WIDTH = 704
HEIGHT = 520

def get_train_transforms():
    return Compose([
        # RandomResizedCrop(img_size, img_size),
        # RandomCrop(img_size, img_size),
        #Resize(img_size, img_size),
        #Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        #ShiftScaleRotate(p=1.0),
        #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        #RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        #CoarseDropout(p=0.5),
        #Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1., bbox_params=BboxParams(format='pascal_voc', label_fields=["bbox_classes"]))

def get_check_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1., bbox_params=BboxParams(format='pascal_voc', label_fields=["bbox_classes"]))

def get_valid_transforms():
    return Compose([
        #CenterCrop(img_size, img_size, p=1.),
        #Resize(img_size, img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1., bbox_params=BboxParams(format='pascal_voc', label_fields=["bbox_classes"]))


def get_inference_transforms():
    return Compose([
        # RandomResizedCrop(img_size, img_size),
        # RandomCrop(img_size, img_size),
        #Resize(img_size, img_size),
        #Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        #ShiftScaleRotate(p=1.0),
        #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        #RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1., bbox_params=BboxParams(format='pascal_voc', label_fields=["bbox_classes"]))

def rle_decode(mask_rle, shape, color=1):
    '''
    :param mask_rle: run-length as string formated (start length)
    :param shape: (height, width) of array to return
    :return: numpy array, 1-mask, 0-background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = color
    return img.reshape(shape)

class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir

        self.height = HEIGHT
        self.width = WIDTH

        self.image_info = collections.defaultdict(dict)
        for index, row in df.iterrows():
            self.image_info[index] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id']+'.png'),
                'annotations': row['annotation']
            }

    def get_box(self, a_mask):
        '''Get the bounding box of a given mask'''
        pos = np.where(a_mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        '''Get the image and the target'''
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        info = self.image_info[idx]

        n_objects = len(info['annotations'])
        masks = []
        boxes = []

        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            a_mask = Image.fromarray(a_mask)
            a_mask = np.where(np.array(a_mask) > 0, 1, 0)
            masks.append(a_mask)
            boxes.append(self.get_box(a_mask))

        labels = [1 for _ in range(n_objects)]

        if self.transforms is not None:
            transformed = self.transforms(image=img, masks=masks, bboxes=boxes, bbox_classes=labels)
            img = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']
            labels = transformed['bbox_classes']

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': is_crowd
        }
        return img, target

    def __len__(self):
        return len(self.image_info)


if __name__ == "__main__":
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
    INPUT_DIR = os.path.join(SRC_DIR, 'input')
    df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    df = df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
    ds = CellDataset(os.path.join(INPUT_DIR, 'train'), df, transforms=get_check_transforms())
    print("Dataset size: {}".format(len(ds)))
    ds_iter = iter(ds)
    for i, (img, target) in enumerate(ds_iter):
        print("[{}]image shape: {}, masks shape: {}, boxes shape: {}, labels shape: {}".format(i, img.shape, target['masks'].shape, target['boxes'].shape, target['labels'].shape))
        img = np.array(img).transpose((1, 2, 0))
        mask = np.array(target['masks']).sum(axis=0, dtype=np.float64).clip(min=0, max=1)
        result = img.astype(float)/255 * 0.8
        result[:, :, 0] = result[:, :, 0] + mask * 0.2
        result = result * 255
        result = Image.fromarray(result.astype(np.uint8))
        draw = ImageDraw.Draw(result)
        boxes = np.array(target['boxes'])
        for j in range(boxes.shape[0]):
            draw.rectangle(boxes[j, :], outline=(0, 0, 255))
        plt.imshow(result)
        plt.show()
        if i > 1:
            break


    def test(ds):
        dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        for imgs, targets in dl:
            print(imgs, targets)
            break

    test(ds)

