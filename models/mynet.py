import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shared_encoder import SharedEncoder
from models.line_head import LineHeadNet
from models.keypoint_head import KeypointHeadNet
from models.seg_head import SegmentHeadNet

import netron

class GreVONet(nn.Module):
    
    def __init__(self, outdim=128, feature='point', segment=False):
        super().__init__()
        self.outdim = outdim
        self.feature = feature      # 'point', 'line', 'combined'
        self.segment = segment      # True, False
        
        # encoder
        self.shared_encoder = SharedEncoder()
        
        # decoder
        self.line_head = LineHeadNet(n_channels=outdim, n_classes=1, bilinear=False)
        self.keypoint_head = KeypointHeadNet()
        self.seg_head = SegmentHeadNet()
        
    def detect_points(self, x):
        out_shared = self.shared_encoder(x)
        out_kpt = self.keypoint_head(out_shared)
        pass
    
    def forward(self, x):
        if self.feature == 'point':
            out_point = self.detect_points(x)
            return out_point
            
def netron_vis_net():
    output_path = '/home/wenhuanyao/317VO/pretrained/system_vis.pth'
    net = GreVONet(feature='point', segment=False)
    torch.onnx.export(net, torch.randn(1, 3, 1024, 2048), output_path, verbose=True)
    netron.start(output_path)
    
def train():
    pass
    
if __name__ == '__main__':
    netron_vis_net()
    pass