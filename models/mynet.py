import torch
import torch.nn as nn
import torch.nn.functional as F
from sfd2 import ResSegNetV2
from superpoint import SuperPointNet

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# def batch_normalization(channels, relu=False):
#     if relu:
#         return nn.Sequential(
#             nn.BatchNorm2d(channels, affine=False, track_running_stats=True),
#             nn.ReLU(), )
#     else:
#         return nn.Sequential(
#             nn.BatchNorm2d(channels, affine=False, track_running_stats=True), )

# def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, use_bn=True, dilation=1):
#     if not use_bn:
#         if relu:
#             return nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                           padding=padding, dilation=dilation),
#                 nn.ReLU(),
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                           padding=padding, dilation=dilation)
#             )
#     else:
#         if relu:
#             return nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                           padding=padding, dilation=dilation),
#                 nn.BatchNorm2d(out_channels, affine=False),
#                 nn.ReLU(),
#             )
#         else:
#             return nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                           padding=padding, dilation=dilation),
#                 nn.BatchNorm2d(out_channels, affine=False),
#                 # nn.ReLU(),
#             )

# class ResBlock(nn.Module):
#     def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None):
#         super(ResBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self.conv1 = conv1x1(inplanes, outplanes)
#         self.bn1 = norm_layer(outplanes)
#         self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
#         self.bn2 = norm_layer(outplanes)
#         self.conv3 = conv1x1(outplanes, outplanes)
#         self.bn3 = norm_layer(outplanes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         out += identity
#         out = self.relu(out)

#         return out

class GreVONet(nn.Module):
    # defult_config = {
    #     "num_classes": 1,
    #     "num_stacks": 2,
    #     "num_blocks": 1,
    #     "num_features": 256,
    #     "num_hourglass_features": 256
    # }
    
    def __init__(self, outdim=128, require_feature=False, require_stability=False):
    # def __init__(self, outdim=128, require_feature=False, require_stability=False, ms_detector=True):
        super().__init__()
        self.superpoint = SuperPointNet()
        self.ressegnet = ResSegNetV2()    # 暂时使用
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        
        # self.ms_detector = ms_detector

        # d1, d2, d3, d4, d5, d6 = 64, 128, 256, 256, 256, 256
        # self.conv1a = conv(in_channels=3, out_channels=d1, kernel_size=3, relu=True, use_bn=True)
        # self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, stride=2, relu=False, use_bn=False)
        # self.bn1b = batch_normalization(channels=d1, relu=True)

        # self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, relu=True, use_bn=True)
        # self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, stride=2, relu=False, use_bn=False)
        # self.bn2b = batch_normalization(channels=d2, relu=True)

        # self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, relu=True, use_bn=True)
        # self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=False, use_bn=False)
        # self.bn3b = batch_normalization(channels=d3, relu=True)

        # self.conv4 = nn.Sequential(
        #     ResBlock(inplanes=256, outplanes=256, groups=32),
        #     ResBlock(inplanes=256, outplanes=256, groups=32),
        #     ResBlock(inplanes=256, outplanes=256, groups=32),
        # )

        # self.convPa = nn.Sequential(
        #     torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        # )
        # self.convDa = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # )

        # self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        # self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

        # if self.require_stability:
        #     self.ConvSta = nn.Conv2d(256, 3, kernel_size=1)
        
    def detect(self, x):
        pass
    
    def detect_train(self, x):
        x = self.superpoint(x)
        out = self.ressegnet(x)
        return out
    
    def forward(self, x):
        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.detect_train(x)
            return {
                "reliability": score,
                "score": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.detect_train(x)
            return {
                "reliability": score,
                "score": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }
            
if __name__ == '__main__':
    model = GreVONet()
    print(model)
    pass