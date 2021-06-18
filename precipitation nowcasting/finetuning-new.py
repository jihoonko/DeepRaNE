from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
import time

from dataset import RadarAndRainMultiDataset
from model import UNetClassificationV2

import os

def train(args):
    train_dataset = RadarAndRainMultiDataset(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2014, year_to=2018)
    val_dataset = RadarAndRainMultiDataset(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2019, year_to=2019)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, collate_fn = train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, collate_fn = val_dataset.collate_fn)
    model = UNetClassificationV2(num_classes=3, img_dim=7, time_dim=36, initial_channels=32).to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
        model.module.unet.load_state_dict(torch.load(args.pretrained_weights_path))
    else:
        model.unet.load_state_dict(torch.load(args.pretrained_weights_path))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    step_cnt = 0
    train_loss_sum, train_acc = 0., 0
    TP1, TN1, FP1, FN1 = 0, 0, 0, 0
    TP2, TN2, FP2, FN2 = 0, 0, 0, 0
    model.train()
    f_log = open("./logs/finetuning_new.log", "a")
    while step_cnt < args.n_steps:
        start = time.time()
        for imgs, indices, rows, cols, vals, wheres in tqdm(iter(train_loader)):
            optimizer.zero_grad()
            
            imgs = imgs.to(args.default_device)
            indices, rows, cols, vals = indices.to(args.default_device), rows.to(args.default_device) - 381, cols.to(args.default_device) - 381, vals.to(args.default_device)
            
            _preds = model(imgs, (wheres * 6 + 5))
            preds = torch.softmax(_preds[indices, rows, cols], dim=-1)
            labels = torch.argmax(preds, dim=-1).to(args.default_device)
            
            smooth_TP_1mm = (preds[:, 1:].sum(-1) * (vals > 0).float()).sum()
            smooth_TN_1mm = (preds[:, :1].sum(-1) * (vals == 0).float()).sum()
            smooth_FP_1mm = (preds[:, 1:].sum(-1) * (vals == 0).float()).sum()
            smooth_FN_1mm = (preds[:, :1].sum(-1) * (vals > 0).float()).sum()
            smooth_TP_10mm = (preds[:, 2:].sum(-1) * (vals == 2).float()).sum()
            smooth_TN_10mm = (preds[:, :2].sum(-1) * (vals < 2).float()).sum()
            smooth_FP_10mm = (preds[:, 2:].sum(-1) * (vals < 2).float()).sum()
            smooth_FN_10mm = (preds[:, :2].sum(-1) * (vals == 2).float()).sum()
            loss = -((smooth_TP_1mm / (smooth_TP_1mm + smooth_FN_1mm + smooth_FP_1mm + 1e-6))
                     + (smooth_TP_10mm / (smooth_TP_10mm + smooth_FN_10mm + smooth_FP_10mm + 1e-6)))
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_acc += (labels == vals).float().sum().item()
            TP1 += ((labels > 0) & (vals > 0)).float().sum().item()
            TN1 += ((labels == 0) & (vals == 0)).float().sum().item()
            FP1 += ((labels > 0) & (vals == 0)).float().sum().item()
            FN1 += ((labels == 0) & (vals > 0)).float().sum().item()
            TP2 += ((labels == 2) & (vals == 2)).float().sum().item()
            TN2 += ((labels < 2) & (vals < 2)).float().sum().item()
            FP2 += ((labels == 2) & (vals < 2)).float().sum().item()
            FN2 += ((labels < 2) & (vals == 2)).float().sum().item()
            
            del _preds, preds, loss, labels
            
            step_cnt += 1
            if step_cnt % 100 == 0:
                f_log.write(f'Step #{step_cnt}: train_loss is {train_loss_sum / (TP1 + TN1 + FP1 + FN1)} train_acc: {train_acc / (TP1 + TN1 + FP1 + FN1)} ')
                f_log.write(f'F1_1mm: {(2 * TP1) / (2 * TP1 + FN1 + FP1)} F1_10mm: {(2 * TP2) / (2 * TP2 + FN2 + FP2)} ')
                f_log.write(f'CSI_1mm: {(TP1) / (TP1 + FN1 + FP1)} CSI_10mm: {(TP2) / (TP2 + FN2 + FP2)}\n')
                f_log.flush()
                train_loss_sum, train_acc = 0., 0
                TP1, TN1, FP1, FN1 = 0, 0, 0, 0
                TP2, TN2, FP2, FN2 = 0, 0, 0, 0
    
            if step_cnt % 1000 == 0:
                end = time.time()
                f_log.write(f'Duration time is {end - start}\n')
                f_log.flush()
                model.eval()
                torch.save(model.module.state_dict(), f'./checkpoints/finetuning_new_{step_cnt}.pkt')
                val_loss_sum, val_acc = 0., 0
                _TP1, _TN1, _FP1, _FN1 = 0, 0, 0, 0
                _TP2, _TN2, _FP2, _FN2 = 0, 0, 0, 0
                with torch.no_grad():
                    for imgs, indices, rows, cols, vals, wheres in tqdm(iter(val_loader)):
                        imgs = imgs.to(args.default_device)
                        indices, rows, cols, vals = indices.to(args.default_device), rows.to(args.default_device) - 381, cols.to(args.default_device) - 381, vals.to(args.default_device)
                        
                        _preds = model(imgs, (wheres * 6 + 5))
                        preds = torch.softmax(_preds[indices, rows, cols], dim = -1)
                        labels = torch.argmax(preds, dim=-1).to(args.default_device)
                        
                        mTP1 = (preds[:, 1:].sum(-1) * (vals > 0).float()).sum()
                        mTN1 = (preds[:, :1].sum(-1) * (vals == 0).float()).sum()
                        mFP1 = (preds[:, 1:].sum(-1) * (vals == 0).float()).sum()
                        mFN1 = (preds[:, :1].sum(-1) * (vals > 0).float()).sum()
                        mTP2 = (preds[:, 2:].sum(-1) * (vals == 2).float()).sum()
                        mTN2 = (preds[:, :2].sum(-1) * (vals < 2).float()).sum()
                        mFP2 = (preds[:, 2:].sum(-1) * (vals < 2).float()).sum()
                        mFN2 = (preds[:, :2].sum(-1) * (vals == 2).float()).sum()
                        loss = -((mTP1 / (mTP1 + mFN1 + mFP1 + 1e-6)) + (mTP2 / (mTP2 + mFN2 + mFP2 + 1e-6)))
            
                        val_loss_sum += loss.item()
                        
                        val_acc += (labels == vals).float().sum().item()
                        _TP1 += ((labels > 0) & (vals > 0)).float().sum().item()
                        _TN1 += ((labels == 0) & (vals == 0)).float().sum().item()
                        _FP1 += ((labels > 0) & (vals == 0)).float().sum().item()
                        _FN1 += ((labels == 0) & (vals > 0)).float().sum().item()
                        _TP2 += ((labels == 2) & (vals == 2)).float().sum().item()
                        _TN2 += ((labels < 2) & (vals < 2)).float().sum().item()
                        _FP2 += ((labels == 2) & (vals < 2)).float().sum().item()
                        _FN2 += ((labels < 2) & (vals == 2)).float().sum().item()
                        del _preds, preds, loss, labels
                        
                f_log.write(f'val_loss is {val_loss_sum / (_TP1 + _TN1 + _FP1 + _FN1)} val_acc: {val_acc / (_TP1 + _TN1 + _FP1 + _FN1)} ')
                f_log.write(f'F1_1mm: {(2 * _TP1) / (2 * _TP1 + _FN1 + _FP1)} F1_10mm: {(2 * _TP2) / (2 * _TP2 + _FN2 + _FP2)} ')
                f_log.write(f'CSI_1mm: {(_TP1) / (_TP1 + _FN1 + _FP1)} CSI_10mm: {(_TP2) / (_TP2 + _FN2 + _FP2)}\n')
                f_log.flush()
                model.train()
                start = time.time()
            if step_cnt == args.n_steps:
                break    
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='fine-tuning (U-Net)')
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--n-steps", type=int, default=35000,
                        help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default=None,
                        help="path of radar data")
    parser.add_argument("--sampled-path", type=str, default='sampled/sampled',
                        help="path of sampled data for training")
    parser.add_argument("--pretrained-weights-path", type=str, default='./checkpoints/pretrained.pkt',
                        help="path of pretrained weights")
    parser.add_argument("--gpus", type=str, default=None,
                        help="gpu id for execution")
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        if args.gpus == 'cpu':
            args.all_devices = None
            args.default_device = 'cpu:0'
        elif args.gpus is not None:
            args.all_devices = list(map(int, args.gpus.split(',')))
            args.default_device = args.all_devices[0]
        else:
            args.all_devices = [i for i in range(torch.cuda.device_count())]
            args.default_device = torch.cuda.current_device()
    else:
        args.all_devices = None
        args.default_device = 'cpu:0'
    
    if args.data_path is None:
        print('Path of the RADAR data should be provided!')
        exit(0)
    
    print(f'enabled gpu_devices: {args.all_devices}, default device: {args.default_device}')
    print(args)
    train(args)