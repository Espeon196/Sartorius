import argparse
import os
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

import model as my_model
from dsets import BrainTumor2dSimpleDataset, get_train_transforms, get_valid_transforms
import dsets as my_dsets
from utils import seed_everything, MlflowWriter

import warnings
warnings.filterwarnings("ignore")

COMPETITION_NAME = "BrainTumorRadiogenomicClassification"

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

    train_ds = BrainTumor2dSimpleDataset(train_, data_path=data_path, img_size=config['img_size'], transforms=get_train_transforms(config['img_size']), output_label=True)
    valid_ds = BrainTumor2dSimpleDataset(valid_, data_path=data_path, img_size=config['img_size'], transforms=get_valid_transforms(config['img_size']), output_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=config['num_workers'],
    )

    val_loader = DataLoader(
        valid_ds,
        batch_size=config['valid_bs'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader

def make_model(name, **kwargs):
    # to make model
    return my_model.__dict__[name](**kwargs)

def make_optimizer(params, name, **kwargs):
    # to make Optimizer
    return torch.optim.__dict__[name](params, **kwargs)

def make_scheduler(optimizer, name, **kwargs):
    # to make scheduler
    return torch.optim.lr_scheduler.__dict__[name](optimizer, **kwargs)

def make_criterion(name, **kwargs):
    # to make criterion
    return nn.__dict__[name](**kwargs)


def train_one_epoch(fold, epoch, model, loss_fn, train_loader, optimizer, scaler, config, scheduler=None, writer=None):
    model.train()

    t = time.time()
    running_loss = None
    epoch_loss = 0.
    sample_num = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(DEVICE).float()
        image_labels = image_labels.to(DEVICE).float()

        with autocast():
            image_preds = model(imgs)  # output = model(input)

            # 次元をそろえる
            image_labels = image_labels.unsqueeze(1)
            loss = loss_fn(image_preds, image_labels)

        scaler.scale(loss).backward()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        epoch_loss += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % config['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and config['trn_schd_batch_update']:
                scheduler.step()

        if ((step + 1) % config['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'

            pbar.set_description(description)

    if scheduler is not None and not config['trn_schd_batch_update']:
        scheduler.step()

    epoch_loss = epoch_loss / sample_num
    if writer:
        writer.log_metric("fold-{}/train/loss".format(fold), epoch_loss, epoch)

def valid_one_epoch(fold, epoch, model, loss_fn, val_loader, config, scheduler=None, writer=None):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(DEVICE).float()
        image_labels = image_labels.to(DEVICE).float()


        image_preds = model(imgs)  # output = model(input)

        #image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_preds_all += [image_preds.detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        # 次元をそろえる
        image_labels = image_labels.unsqueeze(1)
        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % config['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)

    epoch_loss = loss_sum / sample_num

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    epoch_acc = (np.where(image_preds_all>0, 1, 0) == image_targets_all).mean()
    print('validation accuracy = {:.4f}'.format(epoch_acc))

    epoch_auc = roc_auc_score(image_targets_all, image_preds_all)
    print('validation roc auc score = {:.4f}'.format(epoch_auc))

    if scheduler is not None:
        if config['val_schd_loss_update']:
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    if writer:
        writer.log_metric("fold-{}/val/loss".format(fold), epoch_loss, epoch)
        writer.log_metric("fold-{}/val/acc".format(fold), epoch_acc, epoch)
        writer.log_metric("fold-{}/val/auc".format(fold), epoch_auc, epoch)

    return epoch_auc, image_preds_all, image_targets_all

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

    folds = StratifiedKFold(n_splits=base_config['fold_num'], shuffle=True, random_state=base_config['seed']).split(df, y=df.MGMT_value.tolist())

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

    fold_best_aucs = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print('Training with {} started'.format(fold))

        #SAVE_PATH_TMP = os.path.join(SAVE_PATH, "tmp_{}".format(fold))
        #os.makedirs(SAVE_PATH_TMP, exist_ok=True)

        train_loader, val_loader = prepare_dataloader(df, trn_idx, val_idx, data_path=os.path.join(INPUT_DIR, 'train'), config=base_config)
        model = make_model(**model_config).to(DEVICE)
        optimizer = make_optimizer(model.parameters(), **optimizer_config)
        scheduler = make_scheduler(optimizer, **scheduler_config)
        criterion = make_criterion(**criterion_config).to(DEVICE)

        scaler = GradScaler()

        best_epoch = None
        best_auc = -1.0
        fold_preds = None
        fold_targets = None

        for epoch in range(base_config['epochs']):
            print("epoch: ", epoch)

            train_one_epoch(fold, epoch, model, criterion, train_loader, optimizer, scaler, config=base_config, scheduler=None, writer=writer)

            with torch.no_grad():
                epoch_auc, image_preds, image_targets = valid_one_epoch(fold, epoch, model, criterion, val_loader, config=base_config, scheduler=scheduler, writer=writer)
                writer.log_metric("lr", optimizer.param_groups[0]['lr'], epoch)

            if epoch_auc >= best_auc:
                best_epoch = epoch
                best_auc = epoch_auc
                fold_preds = image_preds
                fold_targets = image_targets

                torch.save(model.state_dict(), os.path.join(SAVE_PATH, '{}_fold_{}_best.pth'.format(run_name.replace(' ', '_'), fold)))

        print("best model is {} epoch ({})".format(best_epoch, best_auc))
        writer.log_metric('fold-{}/AUC'.format(fold), best_auc)
        fold_best_aucs.append(best_auc)
        writer.log_artifact(local_path=os.path.join(SAVE_PATH, '{}_fold_{}_best.pth'.format(run_name.replace(' ', '_'), fold)))

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()

    writer.log_metric('AUC', statistics.mean(fold_best_aucs))
    writer.set_terminated()

if __name__ == "__main__":
    start_time = time.time()
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print(f'\nsuccess! [{(time.time()-start_time)/60:.1f} min]')