import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import netron


class KeypointHeadNet(nn.Module):
    
    def __init__(self, outdim=128):
        super().__init__()
        self.convPa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.convDa = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # detect
        cPa = self.convPa(x)
        semi = self.convPb(cPa)
        
        # descriptor
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)
        
        return semi, desc

def netron_vis_net():
    output_path = '/home/wenhuanyao/317VO/pretrained/keypointhead_vis.pth'
    net = KeypointHeadNet()
    torch.onnx.export(net, torch.randn(1, 256, 64, 64), output_path, verbose=True)
    netron.start(output_path)
    
if __name__ == '__main__':
    netron_vis_net()
    pass