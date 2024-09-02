import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
        
    
    
    
    
    

# ----------------------------------------------------------------
# chen
class DualConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DualConv, self).__init__()
        self._conv1 = nn.Conv2d(input_dim, output_dim, 3, 1, 1, 1)
        self._bn1 = nn.BatchNorm2d(output_dim)
        self._relu1 = nn.ReLU(inplace=True)
        self._conv2 = nn.Conv2d(output_dim, output_dim, 3, 1, 1)
        self._bn2 = nn.BatchNorm2d(output_dim)
        self._relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv1(x)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._conv2(x)
        x = self._bn2(x)
        x = self._relu2(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dualconv1 = DualConv(3, 64)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.dualconv2 = DualConv(64, 128)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dualconv3 = DualConv(128, 256)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.dualconv4 = DualConv(256, 512)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.dualconv5 = DualConv(512, 1024)
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.up1 = DualConv(1024, 512)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.up2 = DualConv(512, 256)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.up3 = DualConv(256, 128)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.up4 = DualConv(128, 64)
        self.conv1x1_1 = nn.Conv2d(64, 3, 1, 1, 0)
        self.batchnorm6 = nn.BatchNorm2d(3)


    def crop_and_concat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        # print(x.shape)
        x1 = self.dualconv1(x)
        # print("x1", x1.shape)
        x2 = self.maxpool1(x1)
        # print("x2", x2.shape)
        x2 = self.dualconv2(x2)
        x3 = self.maxpool2(x2)
        # print("x3", x3.shape)
        x3 = self.dualconv3(x3)
        x4 = self.maxpool3(x3)
        # print("x4", x4.shape)
        x4 = self.dualconv4(x4)
        x5 = self.maxpool4(x4)
        # print("x5", x5.shape)
        x5 = self.dualconv5(x5)
        # print("x5", x5.shape)
        x = self.up_conv1(x5)
        # print("up_x1", x.shape)
        x = self.batchnorm1(x)
        x = self.crop_and_concat(x, x4)
        # print("up_x1", x.shape)
        x = self.up1(x)
        # print("up_x1", x.shape)
        x = self.up_conv2(x)
        # print("up_x2", x.shape)
        x = self.batchnorm2(x)
        x = self.crop_and_concat(x, x3)
        x = self.up2(x)
        x = self.up_conv3(x)
        # print("up_x3", x.shape)
        x = self.batchnorm3(x)
        x = self.crop_and_concat(x, x2)
        x = self.up3(x)
        x = self.batchnorm4(x)
        x = self.up_conv4(x)
        x = self.batchnorm5(x)
        x = self.crop_and_concat(x, x1)
        x = self.up4(x)
        x = self.conv1x1_1(x)
        x = self.batchnorm6(x)
        # print("output", x.shape)
        return x

