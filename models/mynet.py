import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_encoder import SharedEncoder
from line_head import LineHeadNet
from keypoint_head import KeypointHeadNet
from seg_head import SegmentHeadNet

import netron


class Stability(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.ConvSta = nn.Conv2d(256, 3, kernel_size=1)
    
    def forward(self, x):
        stability = self.ConvSta(x)
        stability = F.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
        # score = score * self.cls_to_value(x=stability)
        # stability = torch.softmax(stability, dim=1)    
        return stability


class GreVONet(nn.Module):
    
    def __init__(self, feature='point', segment=False, is_training=True):
        super().__init__()
        self.feature = feature      # 'point', 'line', 'combined'
        self.segment = segment      # True, False
        self.is_training = is_training
        
        # encoder
        self.shared_encoder = SharedEncoder()
        
        # decoder
        # self.line_head = LineHeadNet(n_channels=outdim, n_classes=1, bilinear=False)
        self.keypoint_head = KeypointHeadNet()
        # self.stability = Stability()
        # self.seg_head = SegmentHeadNet()
    
    def forward(self, x):
        if self.is_training:
            out4, out2c, out3c = self.shared_encoder(x)
            semi, desc, score, stability = self.keypoint_head(out4)
            # seg = self.seg_head(out4)
            return out4, semi, desc, score, stability
            
def netron_vis_net():
    output_path = '/home/wenhuanyao/317VO/pretrained/system_vis.pth'
    net = GreVONet(feature='point', segment=False, is_training=True)
    torch.onnx.export(net, torch.randn(1, 3, 320, 640), output_path, verbose=True)
    netron.start(output_path)
    
if __name__ == '__main__':
    netron_vis_net()
    pass