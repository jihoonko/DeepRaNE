from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
from dataset import RadarAndRainCumulatedDataset
from model import UNetV2
import time

def train(args):
    train_dataset = RadarAndRainCumulatedDataset(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2014, year_to=2018)
    val_dataset = RadarAndRainCumulatedDataset(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2019, year_to=2019)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=15, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=15, collate_fn=val_dataset.collate_fn)
    
    print(len(train_dataset))
    model = UNetV2().to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
        model.module.unet.load_state_dict(torch.load(args.pretrained_weights_path))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
    f_log = open("./logs/finetuning.log", "a")
    step_cnt = 0
    train_loss_sum, val_loss_sum = 0., 0.
    train_cnt = 0
    model.train()
    
    while step_cnt < args.n_steps:
        start = time.time()
        for imgs, indices, rows, cols, vals in tqdm(iter(train_loader)):
            optimizer.zero_grad()
            imgs, indices, rows, cols, vals = imgs.to(args.default_device), indices.to(args.default_device), rows.to(args.default_device) - 381, cols.to(args.default_device) - 381, vals.to(args.default_device)
            _preds = model(imgs)
            preds = torch.clamp(_preds[indices, rows, cols], 0)
            loss = ((preds - vals) * (preds - vals)).sum()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_cnt += vals.shape[0]
            
            del _preds, preds, loss
            
            step_cnt += 1
            if step_cnt % 100 == 0:
                f_log.write(f'Step #{step_cnt}: train_loss is {train_loss_sum / train_cnt}\n')
                f_log.flush()
                train_loss_sum, train_cnt = 0., 0
        
            if step_cnt % 1000 == 0:
                end = time.time()
                f_log.write(f'Duration time is {end - start}\n')
                f_log.flush()
                torch.save(model.module.state_dict(), f'./checkpoints/finetuning_{step_cnt}.pkt')
                model.eval()
                val_loss_sum, val_cnt = 0., 0
                for imgs, indices, rows, cols, vals in tqdm(iter(val_loader)):
                    imgs, indices, rows, cols, vals = imgs.to(args.default_device), indices.to(args.default_device), rows.to(args.default_device) - 381, cols.to(args.default_device) - 381, vals.to(args.default_device)
                    _preds = model(imgs)
                    preds = torch.clamp(_preds[indices, rows, cols], 0)
                    loss = ((preds - vals) * (preds - vals)).sum()
                    val_loss_sum += loss.item()
                    val_cnt += vals.shape[0]
                    del _preds, preds, loss
                    
                f_log.write(f'val_loss is {val_loss_sum / val_cnt}\n')
                f_log.flush()
                model.train()
                start = time.time()
            if step_cnt == args.n_steps:
                break
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cumulative precipitation')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--n-steps", type=int, default=50000,
                        help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=15,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default=None,
                        help="path of radar data")
    parser.add_argument("--pretrained-weights-path", type=str, default='./checkpoints/pretrained.pkt',
                        help="path of pretrained weights")
    parser.add_argument("--sampled-path", type=str, default='./sampled/sampled',
                        help="path of sampled data for training")
    parser.add_argument("--gpus", type=str, default=None,
                        help="list of gpu_devices")
    
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