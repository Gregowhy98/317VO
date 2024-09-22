import torch
import torch.nn as nn
import torch.nn.functional as F
import netron

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def batch_normalization(channels, relu=False):
    if relu:
        return nn.Sequential(
            nn.BatchNorm2d(channels, affine=False, track_running_stats=True),
            nn.ReLU(), )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(channels, affine=False, track_running_stats=True), )

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, use_bn=True, dilation=1):
    if not use_bn:
        if relu:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation)
            )
    else:
        if relu:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_channels, affine=False),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_channels, affine=False),
                # nn.ReLU(),
            )

class ResBlock(nn.Module):
    
    def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
    
    
class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        
        d1, d2, d3, d4, d5, d6 = 64, 128, 256, 256, 256, 256
        self.conv1a = conv(in_channels=3, out_channels=d1, kernel_size=3, relu=True, use_bn=True)
        self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, stride=2, relu=False, use_bn=False)
        self.bn1b = batch_normalization(channels=d1, relu=True)

        self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, relu=True, use_bn=True)
        self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, stride=2, relu=False, use_bn=False)
        self.bn2b = batch_normalization(channels=d2, relu=True)

        self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, relu=True, use_bn=True)
        self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=False, use_bn=False)
        self.bn3b = batch_normalization(channels=d3, relu=True)

        self.conv4 = nn.Sequential(
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
        )
    
    def forward(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(out1a)
        out1c = self.bn1b(out1b)

        out2a = self.conv2a(out1c)
        out2b = self.conv2b(out2a)
        out2c = self.bn2b(out2b)

        out3a = self.conv3a(out2c)
        out3b = self.conv3b(out3a)
        out3c = self.bn3b(out3b)

        out = self.conv4(out3c)
        return out
    

def netron_vis_net():
    output_path = '/home/wenhuanyao/317VO/pretrained/encoder_vis.pth'
    net = SharedEncoder()
    torch.onnx.export(net, torch.randn(1, 3, 1024, 2048), output_path, verbose=True)
    netron.start(output_path)


if __name__ == '__main__':
    # model = SharedEncoder()
    # print(model)
    
    netron_vis_net()
    pass