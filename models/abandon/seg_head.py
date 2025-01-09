import torch
import torch.nn as nn
import cv2
import numpy as np





class SegmentHeadNet(nn.Module):

    def __init__(self, n_classes=1):  
        super(SegmentHeadNet, self).__init__()  

        # 编码器
        self.enc1 = self.contracting_block(1, 64)  
        self.enc2 = self.contracting_block(64, 128)  
        self.enc3 = self.contracting_block(128, 256)  
        self.enc4 = self.contracting_block(256, 512)  

        # 底部  
        self.bottleneck = self.contracting_block(512, 1024)  

        # 解码器  
        self.dec4 = self.upconv_block(1024, 512)  
        self.dec3 = self.upconv_block(512, 256)  
        self.dec2 = self.upconv_block(256, 128)  
        self.dec1 = self.upconv_block(128, 64)  

        # 输出层  
        self.final = nn.Conv3d(64, n_classes, kernel_size=1)  

    def contracting_block(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.MaxPool3d(kernel_size=2, stride=2)  
        )  

    def upconv_block(self, in_channels, out_channels):  
        return nn.Sequential(  
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
        )  

    def forward(self, x):  
        # 编码器  
        enc1 = self.enc1(x)  
        enc2 = self.enc2(enc1)  
        enc3 = self.enc3(enc2)  
        enc4 = self.enc4(enc3)  

        # Bottleneck  
        bottleneck = self.bottleneck(enc4)  

        # 解码器  
        dec4 = self.dec4(bottleneck)  
        dec4 = torch.cat((dec4, enc4), dim=1)  
        dec3 = self.dec3(dec4)  
        dec3 = torch.cat((dec3, enc3), dim=1)  
        dec2 = self.dec2(dec3)  
        dec2 = torch.cat((dec2, enc2), dim=1)  
        dec1 = self.dec1(dec2)  
        dec1 = torch.cat((dec1, enc1), dim=1)  

        # 输出  
        output = self.final(dec1)  
        return output  