from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
import pickle
from dataset import RadarAndRainCumulatedDataset
from model import UNetV2

from scipy.stats import linregress

def evaluation(args):
    test_dataset = RadarAndRainCumulatedDataset(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2019, year_to=2019)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, collate_fn = test_dataset.collate_fn)
    
    model = UNetV2().to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
        check = torch.load(args.finetuned_weights_path)
        model.module.load_state_dict(check)
    else:
        check = torch.load(args.finetuned_weights_path)
        model.load_state_dict(check)
    
    gt = []
    predict = []
    mseLoss = []
    cnt = 0
    center=(1024, 1024)
    radius=734
    ans = []
    model.eval()
    with torch.no_grad():
        for imgs, indices, rows, cols, vals in tqdm(iter(test_loader)):
            cnt+=1
            imgs, indices, rows, cols, vals = imgs.to(args.default_device), indices.to(args.default_device), rows.to(args.default_device) - 381, cols.to(args.default_device) - 381, vals.to(args.default_device)
            _preds = model(imgs)
            preds = torch.clamp(_preds[indices, rows, cols], min = 0)
            __vals = vals.detach().cpu().numpy()
            __preds = preds.detach().cpu().numpy()
            gt.append(__vals.tolist())
            predict.append(__preds.tolist())
            
            del _preds, preds
        
    r_log = open("./logs/result.log", 'w')
    r_log.write('gt\tprediction\n')
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            ground = gt[i][j]
            pre = predict[i][j]
            r_log.write(f'{ground}\t{pre}\n')
    r_log.flush()
    print("See logs/result.log for prediction results.")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cumulative precipitation evaluation')
    parser.add_argument("--batch-size", type=int, default=24,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default=None,
                        help="path of radar data")
    parser.add_argument("--sampled-path", type=str, default='./sampled/sampled',
                        help="path of sampled data for training")
    parser.add_argument("--finetuned-weights-path", type=str, default='./checkpoints/finetuned.pkt',
                        help="path of pretrained weights")
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
    evaluation(args)