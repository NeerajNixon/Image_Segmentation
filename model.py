import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Downscaling
        self.down_conv1 = self.conv_stage(3, 16)
        self.down_conv2 = self.conv_stage(16, 32)
        self.down_conv3 = self.conv_stage(32, 64)
        self.down_conv4 = self.conv_stage(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_stage(128, 256)

        # Upscaling
        self.up_conv4 = self.conv_stage(256 + 128, 128)
        self.up_conv3 = self.conv_stage(128 + 64, 64)
        self.up_conv2 = self.conv_stage(64 + 32, 32)
        self.up_conv1 = self.conv_stage(32 + 16, 16)

        # Output layer
        self.final_layer = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_stage(self, in_channels, out_channels):
        conv_stage = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv_stage

    def forward(self, x):
        # Downscaling
        conv1 = self.down_conv1(x)
        x = F.max_pool2d(conv1, 2)
        conv2 = self.down_conv2(x)
        x = F.max_pool2d(conv2, 2)
        conv3 = self.down_conv3(x)
        x = F.max_pool2d(conv3, 2)
        conv4 = self.down_conv4(x)
        x = F.max_pool2d(conv4, 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Upscaling + concatenate
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)

        # Output
        x = self.final_layer(x)
        x = self.sigmoid(x)
        return x
