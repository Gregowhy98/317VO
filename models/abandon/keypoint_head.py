import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import netron


class KeypointHeadNet(nn.Module):
    
    def __init__(self):
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
        self.convDb = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.convDb = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)  # 原版sfd2的outputdim是可替换的
    
        self.ConvSta = nn.Conv2d(256, 3, kernel_size=1)
    
    def forward(self, x):
        # detect
        cPa = self.convPa(x)
        semi = self.convPb(cPa)   # 1 x 65 x Hc/8 x Wc/8
        
        semi_e = torch.exp(semi)
        semi_norm = semi_e / (torch.sum(semi_e, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)
        
        # descriptor
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)   # 1 x 256 x Hc/4 x Wc/4
        
        desc = F.interpolate(desc, size=(Hc, Wc), mode='bilinear', align_corners=False)
        # dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        # desc = desc.div(torch.unsqueeze(dn, 1)) 
        
        # stability
        stability = self.ConvSta(x)
        # stability = F.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
        stability = F.interpolate(stability, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear')
        
        return semi, desc, score, stability #, cDa #, score, semi_norm, 

def netron_vis_net():
    output_path = '/home/wenhuanyao/317VO/pretrained/keypointhead_vis.pth'
    net = KeypointHeadNet()
    torch.onnx.export(net, torch.randn(1, 256, 80, 160), output_path, verbose=True)
    netron.start(output_path)
    
if __name__ == '__main__':
    netron_vis_net()
    pass