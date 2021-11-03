import argparse
import os

from PIL import Image, ImageDraw
import cv2
import yaml
import sys
import time
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
from tqdm.auto import tqdm
import shutil
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

import mlflow

from model import get_model
from dsets import CellDataset, get_train_transforms, get_valid_transforms
from utils import seed_everything, MlflowWriter

import warnings
warnings.filterwarnings("ignore")

WIDTH = 704
HEIGHT = 520

COMPETITION_NAME = "CellInstanceSegmentation"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument('--config', help="Config number", type=int, required=True)
parser.add_argument('--input', help="input data folder", type=str)
args = parser.parse_args()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_CODE = FILE_DIR[-6:]
SRC_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
INPUT_DIR = args.input if args.input else os.path.join(SRC_DIR, 'input')
SAVE_PATH = os.path.join(SRC_DIR, 'tmp_artifacts')

def prepare_dataloader(df, trn_idx, val_idx, data_path, config):
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = CellDataset(image_dir=data_path, df=train_, transforms=get_train_transforms())
    valid_ds = CellDataset(image_dir=data_path, df=valid_, transforms=get_valid_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=config['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        valid_ds,
        batch_size=config['valid_bs'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=False,
        collate_fn=lambda x: tuple(zip(*x))
    )
    return train_loader, val_loader

"""
def make_model(name, **kwargs):
    # to make model
    return my_model.__dict__[name](**kwargs)
"""

def make_optimizer(params, name, **kwargs):
    # to make Optimizer
    return torch.optim.__dict__[name](params, **kwargs)

def make_scheduler(optimizer, name, **kwargs):
    # to make scheduler
    return torch.optim.lr_scheduler.__dict__[name](optimizer, **kwargs)

def make_criterion(name, **kwargs):
    # to make criterion
    return nn.__dict__[name](**kwargs)


def train_one_epoch(fold, epoch, model, train_loader, optimizer, scaler, config, scheduler=None, writer=None):
    model.train()

    #t = time.time()
    running_loss = None
    epoch_loss = 0.
    epoch_mask_loss = 0.
    sample_num = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, targets) in pbar:
        imgs = list(img.to(DEVICE) for img in imgs)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            """
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """

            # Logging
            loss_mask = loss_dict['loss_mask'].item()
            epoch_loss += loss.item()
            epoch_mask_loss += loss_mask
            sample_num += len(imgs)

        scaler.scale(loss).backward()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % config['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if ((step + 1) % config['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'

            pbar.set_description(description)

    if scheduler is not None and not config['trn_schd_batch_update']:
        scheduler.step()

    epoch_loss = epoch_loss / sample_num
    epoch_mask_loss = epoch_mask_loss / sample_num
    if writer:
        writer.log_metric("fold-{}/train/loss".format(fold), epoch_loss, epoch)
        writer.log_metric("fold-{}/train/mask-loss".format(fold), epoch_mask_loss, epoch)

def valid_one_epoch(fold, epoch, model, val_loader, config, scheduler=None, writer=None):
    model.eval()

    # dummy variance
    epoch_loss = 0.

    #t = time.time()
    IoU_sum = 0
    sample_num = 0
    mask_imgs_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, targets) in pbar:
        imgs = list(img.to(DEVICE) for img in imgs)
        preds = model(imgs)

        sample_num += len(imgs)

        for pred, target, img in zip(preds, targets, imgs):
            all_pred_masks = np.zeros((HEIGHT, WIDTH))
            for mask in pred['masks'].cpu().detach().numpy():
                all_pred_masks = np.logical_or(all_pred_masks, mask[0]>config['mask_threshold'])
            all_target_masks = np.zeros((HEIGHT, WIDTH))
            for mask in target['masks'].cpu().detach().numpy():
                all_target_masks = np.logical_or(all_target_masks, mask)

            preds_or_target = np.sum(cv2.bitwise_or(all_pred_masks.astype(np.float32), all_target_masks.astype(np.float32)))
            preds_and_target = np.sum(cv2.bitwise_and(all_pred_masks.astype(np.float32), all_target_masks.astype(np.float32)))

            crude_IoU = preds_and_target / preds_or_target
            IoU_sum += crude_IoU

            mask_imgs_all.append(all_pred_masks.astype(np.float32)*255)

        """
        if ((step + 1) % config['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)
        """

    epoch_IoU = IoU_sum / sample_num
    print('validation IoU score = {:.4f}'.format(epoch_IoU))

    if scheduler is not None:
        if config['val_schd_loss_update']:
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    if writer:
        writer.log_metric("fold-{}/val/IoU".format(fold), epoch_IoU, epoch)

    return epoch_IoU, mask_imgs_all

def main():
    with open(os.path.join(FILE_DIR, "config/config_{:0=3}.yaml".format(args.config))) as file:
        CONFIG = yaml.safe_load(file)

    meta_config = CONFIG['meta']
    base_config = CONFIG['base']
    seed_everything(base_config['seed'])

    # MLflow client 準備
    writer = MlflowWriter(experiment_name=COMPETITION_NAME)
    # artifactsの保存先をGCSに変更
    if not meta_config['DEBUG']:
        writer.set_artifact_location_to_gs(meta_config['bucket_name'], SRC_DIR)
    run_name = "[{}-{:0=3}] {}".format(EXP_CODE, args.config, meta_config['run_name'])
    tags = {"mlflow.runName": run_name,
            "exp_code": EXP_CODE,
            "config": args.config}
    writer.create_run_id(tags=tags)
    writer.log_params_from_config(config=base_config, target='base')

    df = pd.read_csv(os.path.join(INPUT_DIR, meta_config['csv_file']))
    df = df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
    if meta_config['DEBUG']:
        df = df[:5]

    folds = KFold(n_splits=base_config['fold_num'], shuffle=True, random_state=base_config['seed']).split(df)

    model_config = CONFIG['model']
    dataset_config = CONFIG['dataset']
    optimizer_config = CONFIG['optimizer']
    scheduler_config = CONFIG['scheduler']
    criterion_config = CONFIG['criterion']
    writer.log_params_from_config(config=model_config, target='model')
    writer.log_params_from_config(config=dataset_config, target='dataset')
    writer.log_params_from_config(config=optimizer_config, target='optimizer')
    writer.log_params_from_config(config=scheduler_config, target='scheduler')
    writer.log_params_from_config(config=criterion_config, target='criterion')

    fold_best_IoUs = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print('Training with {} started'.format(fold))

        #SAVE_PATH_TMP = os.path.join(SAVE_PATH, "tmp_{}".format(fold))
        #os.makedirs(SAVE_PATH_TMP, exist_ok=True)

        train_loader, val_loader = prepare_dataloader(df, trn_idx, val_idx, data_path=os.path.join(INPUT_DIR, 'train'), config=base_config)

        # maskrcnn_resnet50_fpn
        model = get_model(pretrained=model_config['pretrained'])
        model.to(DEVICE)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = make_optimizer(model.parameters(), **optimizer_config)
        scheduler = make_scheduler(optimizer, **scheduler_config)
        # use criterion in model
        #criterion = make_criterion(**criterion_config).to(DEVICE)

        scaler = GradScaler()

        best_epoch = None
        best_IoU = -1.0
        best_imgs_all = None

        for epoch in range(base_config['epochs']):
            print("epoch: ", epoch)

            train_one_epoch(fold, epoch, model, train_loader, optimizer, scaler, config=base_config, scheduler=scheduler, writer=writer)

            with torch.no_grad():
                epoch_IoU, mask_imgs_all = valid_one_epoch(fold, epoch, model, val_loader, config=base_config, scheduler=None, writer=writer)
                writer.log_metric("lr", optimizer.param_groups[0]['lr'], epoch)

            if epoch_IoU >= best_IoU:
                best_epoch = epoch
                best_IoU = epoch_IoU
                best_imgs_all = mask_imgs_all

                torch.save(model.state_dict(), os.path.join(SAVE_PATH, '{}_fold_{}_best.pth'.format(run_name.replace(' ', '_'), fold)))

        print("best model is {} epoch ({})".format(best_epoch, best_IoU))
        writer.log_metric('fold-{}/IoU'.format(fold), best_IoU)
        fold_best_IoUs.append(best_IoU)
        # need to fix
        for idx, img in zip(val_idx, best_imgs_all):
            img_name = df.loc[idx, 'id']
            cv2.imwrite(os.path.join(SAVE_PATH, '{}.png'.format(img_name)), img)
        writer.log_artifact(local_path=os.path.join(SAVE_PATH, '{}_fold_{}_best.pth'.format(run_name.replace(' ', '_'), fold)))

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()

    writer.log_metric('Crude IoU', statistics.mean(fold_best_IoUs))
    writer.set_terminated()

if __name__ == "__main__":
    start_time = time.time()
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print(f'\nsuccess! [{(time.time()-start_time)/60:.1f} min]')