import torch
from torch import nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn = nn.ReLU):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_fn()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, pool_fn = nn.MaxPool2d):
        super(DownSample, self).__init__()
        self.pool = pool_fn(2)
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)
        
    def forward(self, x):
        return self.conv2(self.conv1(self.pool(x)))

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = ConvBNAct(out_channels + out_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)
        
    def forward(self, x, snapshot):
        x = self.upsample(x)
        crop_idx = (snapshot.shape[-1] - x.shape[-1]) // 2
        x = self.conv2(self.conv1(torch.cat((snapshot[:, :, crop_idx:-crop_idx, crop_idx:-crop_idx], x), dim=-3)))
        return x
        
        
class UNet(nn.Module):
    def __init__(self, initial_channels = 32, apply_last_conv=True):
        super(UNet, self).__init__()
        self.initial_convs = nn.Sequential(ConvBNAct(7, initial_channels),
                                           ConvBNAct(initial_channels, initial_channels))
        self.down_layers = nn.ModuleList([DownSample((initial_channels << i), (initial_channels << (i+1))) for i in range(6)])
        self.up_layers = nn.ModuleList([UpSample((initial_channels << (6-i)), (initial_channels << (5-i))) for i in range(6)])
        self.last_conv = nn.Conv2d(initial_channels, 1, 3)
        self.apply_last_conv = apply_last_conv
        
    def forward(self, img):
        x = self.initial_convs(img)
        self.snapshots = []
        for layer in self.down_layers:
            self.snapshots.append(x)
            x = layer(x)
        for i, layer in enumerate(self.up_layers):
            x = layer(x, self.snapshots[-(i+1)])
        del self.snapshots
        if self.apply_last_conv:
            x = self.last_conv(x).squeeze(-3)
        return x
    
class UNetV2(nn.Module):
    def __init__(self, initial_channels = 32):
        super(UNetV2, self).__init__()
        self.unet = UNet(initial_channels, apply_last_conv=True)
        
    def forward(self, img):
        x = self.unet(img)
        return x
    
class UNetPretrain(nn.Module):
    def __init__(self, num_classes, initial_channels = 32):
        super(UNetPretrain, self).__init__()
        self.unet = UNet(initial_channels, apply_last_conv=False)
        self.last_conv = nn.Conv2d(initial_channels, num_classes, 3)
        
    def forward(self, img):
        x = self.unet(img)
        return x