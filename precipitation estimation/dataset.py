import gzip
import numpy as np
import torch
import torch.nn as nn
from itertools import chain
import pickle
import os
import random
import tqdm
import math

class RadarAndRainCumulatedDataset(torch.utils.data.Dataset):
    def __init__(self, sampled_path, data_path, year_from=2014, year_to=2018, interval=0, input_dim=7, center=(1024, 1024), radius=734, original_radar_size=(2048, 2048)):
        assert type(year_from) == int and type(year_to) == int and type(input_dim) == int
        assert type(interval) == int and interval % 10 == 0
        
        self.date = []
        self.days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1,
                                                                                       year * 10000 + (month + 1) * 100 + 1 +
                                                                                       self.days[month] + int((month == 1) and (year % 4 == 0))))
                                                                    for month in range(12)]) for year in range(year_from, year_to + 1)]))        
        self.inv_dates = {v: i for i, v in enumerate(self.dates)}
        self.interval, self.input_dim = interval, input_dim
        
        self.center = center
        self.radius = radius
        self.original_radar_size = original_radar_size
        
        self.data_path = data_path
        
        with open(sampled_path, "rb") as f:
            raw_raindata = pickle.load(f)
            self.raindata = list(filter(lambda x: (year_from <= int(x[0][:4]) <= year_to) and self.load_images(x[0], just_check=True), raw_raindata))
        
    def load_images(self, timestamp, just_check=False):
        def load_image_inner_check(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            return os.path.exists(f'{self.data_path}/radar_{datestr}.bin.gz')
        
        def load_image_inner(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            with gzip.open(f'{self.data_path}/radar_{datestr}.bin.gz', 'rb') as f:
                target = f.read()
                result = np.frombuffer(target, 'i2').reshape(*self.original_radar_size)
                img = np.maximum(result[(self.center[0]+self.radius):(self.center[0]-self.radius):-1, (self.center[1]-self.radius):(self.center[1]+self.radius)], 0) / 10000.
            return img
        
        date, hour = timestamp[:-4], int(timestamp[-4:-2])
        idx = self.inv_dates[date] * 144 + hour * 6
        
        if just_check:
            valid = True
            for i in range(self.input_dim):
                if not load_image_inner_check(idx - ((self.input_dim - 1) + (self.interval // 10)) + i):
                    valid = False
                    break
            if not load_image_inner_check(idx):
                valid = False
            return valid
        else:
            history = torch.FloatTensor(np.stack([load_image_inner(idx - ((self.input_dim - 1) + (self.interval // 10)) + i) for i in range(self.input_dim)], axis=0))
            return history
    
    def __len__(self) -> int:
        return len(self.raindata)
    
    def __getitem__(self, idx): # history & gt
        history = self.load_images(self.raindata[idx][0])
        row, col, val = zip(*map(lambda x: (self.radius - (x[0] - self.center[0]), self.radius + (x[1] - self.center[1]), x[2]), self.raindata[idx][1]))
        return history, torch.LongTensor(row), torch.LongTensor(col), torch.FloatTensor(val)
    
    def collate_fn(self, samples):
        historys, rows, cols, vals = zip(*samples)
        historys = torch.stack(historys)
        
        indices = torch.LongTensor(list(chain.from_iterable([([i] * len(_row)) for i, _row in enumerate(rows)])))
        rows, cols, vals = map(lambda xs: torch.cat(xs, dim=0), [rows, cols, vals])
        
        return historys, indices, rows, cols, vals
    
class RadarAndRainZRDataset(torch.utils.data.Dataset):
    def __init__(self, sampled_path, data_path, year_from=2020, year_to=2020, interval=0, input_dim=6, center=(1024, 1024), radius=734, original_radar_size=(2048, 2048)):
        assert type(year_from) == int and type(year_to) == int and type(input_dim) == int
        assert type(interval) == int and interval % 10 == 0
        
        self.date = []
        self.days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1,
                                                                                       year * 10000 + (month + 1) * 100 + 1 +
                                                                                       self.days[month] + int((month == 1) and (year % 4 == 0))))
                                                                    for month in range(12)]) for year in range(year_from, year_to + 1)]))        
        self.inv_dates = {v: i for i, v in enumerate(self.dates)}
        self.interval, self.input_dim = interval, input_dim
        
        self.center = center
        self.radius = radius
        self.original_radar_size = original_radar_size
        
        self.data_path = data_path
        
        with open(sampled_path, "rb") as f:
            raw_raindata = pickle.load(f)
            self.raindata = list(filter(lambda x: (year_from <= int(x[0][:4]) <= year_to) and self.load_images(x[0], just_check=True), raw_raindata))
        
    def load_images(self, timestamp, just_check=False):
        def load_image_inner_check(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            return os.path.exists(f'{self.data_path}/radar_{datestr}.bin.gz')
        
        def load_image_inner(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            with gzip.open(f'{self.data_path}/radar_{datestr}.bin.gz', 'rb') as f:
                target = f.read()
                result = np.frombuffer(target, 'i2').reshape(*self.original_radar_size)
                img = result[(self.center[0]+self.radius):(self.center[0]-self.radius):-1, (self.center[1]-self.radius):(self.center[1]+self.radius)] / 100.
                x = torch.as_tensor(10**(0.1*img), dtype=torch.double)
                x = ((x/148)**(100/159)) / 6
            return x
        
        date, hour = timestamp[:-4], int(timestamp[-4:-2])
        idx = self.inv_dates[date] * 144 + hour * 6
        
        if just_check:
            valid = True
            for i in range(self.input_dim):
                if not load_image_inner_check(idx - ((self.input_dim - 1) + (self.interval // 10)) + i):
                    valid = False
                    break
            if not load_image_inner_check(idx):
                valid = False
            return valid
        else:
            z = torch.zeros(self.radius + self.radius, self.radius + self.radius) # np.empty((1468, 1468))
            for i in range(self.input_dim):
                y = load_image_inner(idx - ((self.input_dim - 1) + (self.interval // 10)) + i)
                z += y
            return z.detach()
    
    def __len__(self) -> int:
        return len(self.raindata)
    
    def __getitem__(self, idx): # history & gt
        history = self.load_images(self.raindata[idx][0])
        row, col, val = zip(*map(lambda x: (self.radius - (x[0] - self.center[0]), self.radius + (x[1] - self.center[1]), x[2]), self.raindata[idx][1]))
        return history, torch.LongTensor(row), torch.LongTensor(col), torch.FloatTensor(val)
    
    def collate_fn(self, samples):
        historys, rows, cols, vals = zip(*samples)
        historys = torch.stack(historys)
        
        indices = torch.LongTensor(list(chain.from_iterable([([i] * len(_row)) for i, _row in enumerate(rows)])))
        rows, cols, vals = map(lambda xs: torch.cat(xs, dim=0), [rows, cols, vals])
        
        return historys, indices, rows, cols, vals
    
    
class RadarOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, year_from=2014, year_to=2018, interval=0, input_dim=7, center=(1024, 1024), radius=734, original_radar_size=(2048, 2048)):
        assert type(year_from) == int and type(year_to) == int and type(input_dim) == int
        assert type(interval) == int and interval % 10 == 0
        
        self.date = []
        self.days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.dates = list(chain.from_iterable([chain.from_iterable([map(str, np.arange(year * 10000 + (month + 1) * 100 + 1,
                                                                                       year * 10000 + (month + 1) * 100 + 1 +
                                                                                       self.days[month] + int((month == 1) and (year % 4 == 0))))
                                                                    for month in range(12)]) for year in range(year_from, year_to + 1)]))        
        self.inv_dates = {v: i for i, v in enumerate(self.dates)}
        self.interval, self.input_dim = interval, input_dim
        
        self.center = center
        self.radius = radius
        self.original_radar_size = original_radar_size
        
        self.data_path = data_path
        
        self.indices = []
        last_err_idx = -1
        for _idx in tqdm.trange(len(self.dates) * 144):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            if os.path.exists(f'{self.data_path}/radar_{datestr}.bin.gz'):
                if _idx - last_err_idx >= (input_dim):
                    self.indices.append(_idx)
            else:
                last_err_idx = _idx
        
        print(len(self.indices))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx):
        def parse(_idx):
            datestr = "%s%02d%1d0" % (self.dates[_idx // 144], (_idx % 144) // 6, _idx % 6)
            with gzip.open(f"{self.data_path}/radar_{datestr}.bin.gz", "rb") as f:
                target = f.read()
                result = np.frombuffer(target, 'i2').reshape(*self.original_radar_size)
                return np.maximum(result[(self.center[0]+self.radius):(self.center[0]-self.radius):-1, (self.center[1]-self.radius):(self.center[1]+self.radius)], 0) / 10000.
        _idx = self.indices[idx]
        history = torch.FloatTensor(np.stack([parse(_idx - ((self.input_dim - 1)) + i) for i in range(self.input_dim)], axis=0))
        gt = torch.FloatTensor(parse(_idx))
        return history, gt
    
