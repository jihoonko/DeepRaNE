import numpy as np
import torch
from model import UNetClassificationV2
from pyproj import Proj
from itertools import chain
import gzip
import os

def load_data(point, timestamp, data_path, gap):
    target_year = int(timestamp[:4])
    center, radius = (1024, 1024), 734
    
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1,
                                                                              year * 10000 + (month + 1) * 100 + 1 +
                                                                              days[month] + int((month == 1) and (year % 4 == 0))))
                                                           for month in range(12)]) for year in range(target_year, target_year+1)]))
    inv_dates = {v: i for i, v in enumerate(dates)}
    
    def load_image_inner(_idx):
        datestr = "%s%02d%1d0" % (dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
        try:
            with gzip.open(f'{data_path}/radar_{datestr}.bin.gz', 'rb') as f:
                target = f.read()
                result = np.frombuffer(target, 'i2').reshape(2048, 2048)
                img = np.maximum(result[(center[0]+radius):(center[0]-radius):-1, (center[1]-radius):(center[1]+radius)], 0) / 10000.
        except:
            return None
        return img
        
    def location(lat, long):
        p = Proj("+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38.0 +lon_0=126.0 +x_0=2001000 +y_0=2401000 +no_defs +ellps=WGS84 +units=km", preserve_units=True)
        x0, y0 = p(126.0, 38.0)
        x, y = p(long, lat)
        return (int(round(1263 + y - y0)), int(round(828 + x - x0)))    

    date, hour = timestamp[:-2], int(timestamp[-2:])
    idx = inv_dates[date] * 144 + hour * 6
    
    lat, long = point
    y, x = location(lat, long)
    row, col = radius-(y-center[0]), radius+(x-center[1])
    try:
        datestr = "%s%02d%1d0" % (dates[idx // 144], (idx % 144) // 6, idx % 6)
        imgs = torch.FloatTensor(np.stack([load_image_inner(idx - 6 + i) for i in range(7)], axis=0))
    except:
        return (None, row, col)
    
    return imgs, row, col
        
def evaluation(args):
    imgs, row, col = load_data((args.latitude, args.longitude), args.time, args.data_path, args.gap)
    if imgs is None:
        print('Invalid RADAR data')
        exit(0)
    
    model = UNetClassificationV2(num_classes=3, img_dim=7, time_dim=36, initial_channels=32).to(args.default_device)
    model.load_state_dict(torch.load(args.finetuned_weights_path))
    
    model.eval()
    msg = ["<1mm", "1~10mm", ">10mm"]
    with torch.no_grad():
        imgs = imgs.to(args.default_device).unsqueeze(0)
        print(imgs.shape)
        _preds = model(imgs, torch.LongTensor([args.gap * 6 - 1]).to(args.default_device))
        preds = _preds[0, row - 381, col - 381].squeeze()
        preds = torch.argmax(preds).item()
        print(f'Prediction result for point {(args.latitude, args.longitude)} at {int(args.time) + args.gap} from radar data (before {args.gap} hour(s)): {msg[preds]}')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("latitude", type=float, default=None,
                        help="latitude information")
    parser.add_argument("longitude", type=float, default=None,
                        help="longitude information")
    parser.add_argument("time", type=str, default='2019082912',
                        help="time information (Format: YYYYMMDDHH)")
    parser.add_argument("gap", type=int, default=1,
                        help="gap between given images and target time (hour)")
    parser.add_argument("--data-path", type=str, default='../example_data',
                        help="path of radar data")
    parser.add_argument("--finetuned-weights-path", type=str, default='./example_checkpoints/finetuned.pkt',
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