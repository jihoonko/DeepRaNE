from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
import pickle
from dataset import RadarOnlyDataset
from model import UNetPretrain
import time

def train(args):
    train_dataset = RadarOnlyDataset(data_path = args.data_path, year_from=2014, year_to=2018) # changed to 2018
    val_dataset = RadarOnlyDataset(data_path = args.data_path, year_from=2019, year_to=2019) # changed to 2019
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    model = UNetPretrain(num_classes=100, initial_channels=32).to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    f_log = open("./logs/pretraining.log", "a")
    step_cnt = 0
    train_loss = 0.
    model.train()
    
    num_chunks = 2
    while step_cnt < args.n_steps:
        start = time.time()
        for imgs, _gt in tqdm(iter(train_loader)):
            optimizer.zero_grad()
            
            preds = model(imgs)
            _preds_chunk = [torch.softmax(model.module.last_conv(preds[(args.batch_size * i) // num_chunks:(args.batch_size * (i+1)) // num_chunks]).to(args.all_devices[i + 1]).permute(0, 2, 3, 1), dim=-1) for i in range(num_chunks)]
            _gt_chunk = [torch.clamp((_gt[(args.batch_size * i) // num_chunks:(args.batch_size * (i+1)) // num_chunks, 381:-381, 381:-381] * 100).to(args.all_devices[i + 1]), min=0., max=99.) for i in range(num_chunks)]
            _err_chunk = [(torch.arange(100).to(args.all_devices[i + 1]).float() - _gt_chunk[i].unsqueeze(-1)).abs() for i in range(num_chunks)]
            _loss_chunk = [torch.sum((_err_chunk[i] * _preds_chunk[i]), dim=-1).mean() for i in range(num_chunks)]
            loss = 0.
            for i in range(num_chunks):
                loss += _loss_chunk[i].to(args.default_device)
            (loss / num_chunks).backward()
            
            optimizer.step()
            train_loss += loss.item() / num_chunks
            del imgs, _preds_chunk, _gt_chunk, _err_chunk, _loss_chunk, loss
            
            step_cnt += 1
            if step_cnt % 100 == 0:
                f_log.write(f'Step #{step_cnt}: train_loss is {train_loss / 100}\n')
                f_log.flush()
                train_loss = 0.
            
            if step_cnt % 1000 == 0:
                end = time.time()
                f_log.write(f'Duration time is {end - start}\n')
                f_log.flush()
                torch.save(model.module.unet.state_dict(), f'./checkpoints/pretraining_{step_cnt}.pkt')
                model.eval()
                val_loss = 0.
                with torch.no_grad():
                    for i, (imgs, _gt) in enumerate(iter(val_loader)):
                        preds = model(imgs)
                        _preds_chunk = [torch.softmax(model.module.last_conv(preds[(args.batch_size * i) // num_chunks:(args.batch_size * (i+1)) // num_chunks]).to(args.all_devices[i + 1]).permute(0, 2, 3, 1), dim=-1) for i in range(num_chunks)]
                        _gt_chunk = [torch.clamp((_gt[(args.batch_size * i) // num_chunks:(args.batch_size * (i+1)) // num_chunks, 381:-381, 381:-381] * 100).to(args.all_devices[i + 1]), min=0., max=99.) for i in range(num_chunks)]
                        _err_chunk = [(torch.arange(100).to(args.all_devices[i + 1]).float() - _gt_chunk[i].unsqueeze(-1)).abs() for i in range(num_chunks)]
                        _loss_chunk = [torch.sum((_err_chunk[i] * _preds_chunk[i]), dim=-1).mean() for i in range(num_chunks)]
                        loss = 0.
                        for i in range(num_chunks):
                            loss += _loss_chunk[i].to(args.default_device)
                        val_loss += loss.item() / num_chunks
                        del imgs, _preds_chunk, _gt_chunk, _err_chunk, _loss_chunk, loss
            
                    f_log.write(f'val_loss is {val_loss / len(val_loader)}\n')
                    f_log.flush()
                model.train()
                start = time.time()
            if step_cnt == args.n_steps:
                break
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='pre-training (U-Net)')
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--n-steps", type=int, default=50000,
                        help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default='/data/raw',
                        help="path of radar data")
    parser.add_argument("--gpus", type=str, default='4,5,6,7',
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