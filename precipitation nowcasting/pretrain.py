from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np

from dataset import RadarOnlyMultiDataset
from model import UNetClassificationV2

def train(args):
    train_dataset = RadarOnlyMultiDataset(data_path = args.data_path, year_from=2014, year_to=2018) # changed to 2018
    val_dataset = RadarOnlyMultiDataset(data_path = args.data_path, year_from=2019, year_to=2019) # changed to 2019
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    
    model = UNetClassificationV2(num_classes=100, img_dim=7, time_dim=36, initial_channels=32).to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    f_log = open(f"./logs/pretraining.log", "a")
    step_cnt = 0
    train_loss = 0.
    model.train()
    
    while step_cnt < args.n_steps:
        
        for imgs, _gt, _time in tqdm(iter(train_loader)):
            optimizer.zero_grad()
            gt = torch.clamp((_gt[:, 381:-381, 381:-381] * 100).to(args.default_device), min=0., max=99.)
            preds = torch.softmax(model(imgs, _time.squeeze()), dim=-1)
            err = (torch.arange(100).to(args.default_device).float() - gt.unsqueeze(-1)).abs()
            loss = torch.sum((err * preds), dim=-1).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del imgs, preds, gt, err, loss
            
            step_cnt += 1
            if step_cnt % 100 == 0:
                f_log.write(f'Step #{step_cnt}: train_loss is {train_loss / 100}\n')
                f_log.flush()
                train_loss = 0.
            
            if step_cnt % 1000 == 0:
                torch.save(model.module.unet.state_dict(), f'./checkpoints/pretraining_{step_cnt}.pkt')
                model.eval()
                val_loss = 0.
                with torch.no_grad():
                    for i, (imgs, _gt, _time) in enumerate(iter(val_loader)):
                        gt = torch.clamp((_gt[:, 381:-381, 381:-381] * 100).to(args.default_device), min=0., max=99.)
                        preds = torch.softmax(model(imgs, _time.squeeze()), dim=-1)
                        err = (torch.arange(100).to(args.default_device).float() - gt.unsqueeze(-1)).abs()
                        loss = torch.sum((err * preds), dim=-1).mean()
                        val_loss += loss.item()
                        del imgs, preds, gt, err, loss
            
                    f_log.write(f'val_loss is {val_loss / 100}\n')
                    f_log.flush()
                model.train()
                
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