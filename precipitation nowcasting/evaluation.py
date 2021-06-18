from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from dataset import RadarAndRainMultiDataset
from model import UNetClassificationV2

import os

def evaluation(args):
    test_dataset = RadarAndRainMultiDataset(sampled_path = args.sampled_path, data_path = args.data_path, year_from=2020, year_to=2020)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, collate_fn = test_dataset.collate_fn)
    
    model = UNetClassificationV2(num_classes=3, img_dim=7, time_dim=36, initial_channels=32).to(args.default_device)
    if args.all_devices is not None:
        model = torch.nn.DataParallel(model, device_ids=args.all_devices, output_device=args.default_device)
        model.module.load_state_dict(torch.load(args.finetuned_weights_path))
    else:
        model.load_state_dict(torch.load(args.finetuned_weights_path))
    
    _preds_history, _vals_history, _interval_history = [], [], []
    
    model.eval()
    with torch.no_grad():
        for imgs, indices, rows, cols, vals, wheres in tqdm(iter(test_loader)):
            imgs = imgs.to(args.default_device)
            indices, rows, cols, vals, wheres = indices.to(args.default_device), rows.to(args.default_device) - 381, cols.to(args.default_device) - 381, vals.to(args.default_device), wheres.to(args.default_device)
            
            _preds = model(imgs, (wheres * 6 + 5))
            preds = torch.softmax(_preds[indices, rows, cols], dim=-1)
            preds = torch.argmax(preds, dim=-1).to(3)
            
            _preds_history += preds.detach().cpu().numpy().tolist()
            _vals_history += vals.detach().cpu().numpy().tolist()
            _interval_history += wheres[indices].detach().cpu().numpy().tolist()
        
        confusion_matrix = [[[0 for _1 in range(3)] for _2 in range(3)] for _3 in range(6)]
        for r, c, i in zip(_preds_history, _vals_history, _interval_history):
            confusion_matrix[i][r][c] += 1

        print('confusion matrix (1h):\n', np.array(confusion_matrix[0]))
        print('confusion matrix (2h):\n', np.array(confusion_matrix[1]))
        print('confusion matrix (3h):\n', np.array(confusion_matrix[2]))
        print('confusion matrix (4h):\n', np.array(confusion_matrix[3]))
        print('confusion matrix (5h):\n', np.array(confusion_matrix[4]))
        print('confusion matrix (6h):\n', np.array(confusion_matrix[5]))

        print("CSI Score")
        ## CSI Measure
        for i in range(6):
            print(f'CSI score ({i+1}h, >= 10mm)', confusion_matrix[i][2][2] / (confusion_matrix[i][2][2] + confusion_matrix[i][2][1] + confusion_matrix[i][2][0] + confusion_matrix[i][1][2] + confusion_matrix[i][0][2]))
            print(f'CSI score ({i+1}h, >= 1mm)', (confusion_matrix[i][1][1]+confusion_matrix[i][1][2]+confusion_matrix[i][2][1]+confusion_matrix[i][2][2]) / (np.sum(confusion_matrix[i]) - confusion_matrix[i][0][0]))
            print(f'CSI score ({i+1}h, < 1mm)', (confusion_matrix[i][0][0]/(np.sum(confusion_matrix[i]) - (confusion_matrix[i][1][1]+confusion_matrix[i][1][2]+confusion_matrix[i][2][1]+confusion_matrix[i][2][2]))))

        print("F1 Score")
        ## F1 Score
        for i in range(6):
            print(f'F1 score ({i+1}h, >= 10mm)', 2*confusion_matrix[i][2][2] / (2*confusion_matrix[i][2][2] + confusion_matrix[i][2][1] + confusion_matrix[i][2][0] + confusion_matrix[i][1][2] + confusion_matrix[i][0][2]))
            x = (confusion_matrix[i][1][1]+confusion_matrix[i][1][2]+confusion_matrix[i][2][1]+confusion_matrix[i][2][2])
            print(f'F1 score ({i+1}h, >= 1mm)', 2*x / (2*x + confusion_matrix[i][0][1] + confusion_matrix[i][0][2] + confusion_matrix[i][1][0] + confusion_matrix[i][2][0]))
            print(f'F1 score ({i+1}h, < 1mm)', 2* confusion_matrix[i][0][0]/(2* confusion_matrix[i][0][0] + confusion_matrix[i][0][1] + confusion_matrix[i][0][2] + confusion_matrix[i][1][0] + confusion_matrix[i][2][0]))
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='evaluation (U-Net)')
    parser.add_argument("--batch-size", type=int, default=24,
                        help="number of batch size")
    parser.add_argument("--data-path", type=str, default="/data/raw/",
                        help="path of radar data")
    parser.add_argument("--sampled-path", type=str, default='./sampled/sampled',
                        help="path of sampled data for evaluation")
    parser.add_argument("--finetuned-weights-path", type=str, default='./checkpoints/finetuned.pkt',
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
    evaluation(args)