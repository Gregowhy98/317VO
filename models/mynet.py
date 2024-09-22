import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.abandon.sfd2 import ResSegNetV2
# from superpoint import SuperPointNet
from shared_encoder import SharedEncoder
from line_head import LineHeadNet
from keypoint_head import KeypointHeadNet
from seg_head import SegmentHeadNet

import netron

class GreVONet(nn.Module):
    # defult_config = {
    #     "num_classes": 1,
    #     "num_stacks": 2,
    #     "num_blocks": 1,
    #     "num_features": 256,
    #     "num_hourglass_features": 256
    # }
    
    def __init__(self, outdim=128, is_feature=False, is_segment=False):
        super().__init__()
        # self.superpoint = SuperPointNet()
        # self.ressegnet = ResSegNetV2()    # 暂时使用
        self.outdim = outdim
        self.is_feature = is_feature
        self.is_segment = is_segment
        
        # encoder
        self.encoder = SharedEncoder()
        
        # decoder
        self.line_head = LineHeadNet(n_channels=outdim, n_classes=1, bilinear=False)
        self.keypoint_head = KeypointHeadNet()
        self.seg_head = SegmentHeadNet()
    
    def shared_encoder(self, x):
        x = self.encoder(x)
        return x
        
    def detect(self, x):
        pass
    
    def detect_train(self, x):
        pass
    
    def forward(self, x):
        if self.is_feature:
            out1 = self.shared_encoder(x)
            out_kpt = self.keypoint_head(out1)
            out_line = self.line_head(out1)
            return {"kpt": out_kpt, "line": out_line}
        # if self.is_segment:
        #     x = self.shared_encoder(x)
        #     return x
        if self.is_feature and self.is_segment:
            out1 = self.shared_encoder(x)
            out_seg = self.seg_head(out1)
            out_kpt = self.keypoint_head(out1)
            out_line = self.line_head(out1)
            return {"seg": out_seg, "kpt": out_kpt, "line": out_line}
            
def netron_vis_net():
    output_path = '/home/wenhuanyao/317VO/pretrained/system_vis.pth'
    net = GreVONet(128, True, True)
    torch.onnx.export(net, torch.randn(1, 3, 256, 256), output_path, verbose=True)
    netron.start(output_path)
    
if __name__ == '__main__':
    netron_vis_net()
    # net = GreVONet(128, True, True)
    # print(net)
    pass