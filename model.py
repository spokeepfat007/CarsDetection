"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn
from matplotlib import pyplot

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


darknet_config = [(3, 32, 2, 1),
                  (3, 64, 2, 1),
                  [64, 128],
                  "M",
                  [128, 256],
                  "M",
                  [256, 512],
                  "M",
                  (3, 512, 1, 1)]
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.CBLBlock1 = CNNBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.CBLBlock2 = CNNBlock(in_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.CBLBlock3 = CNNBlock(in_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.CBLBlock4 = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.CBLBlock1(x)
        y = torch.split(x, self.out_channels // 2, dim=1)[1]
        y = self.CBLBlock2(y)
        z = self.CBLBlock3(y)
        y = torch.cat([z, y], dim=1)
        y = self.CBLBlock4(y)
        x = torch.cat([x, y], dim=1)
        return x


class SCPDarknet53Tiny(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.layers = self.create_conv_layers()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for layer in darknet_config:
            if isinstance(layer, tuple):
                kernel_size, out_channels, stride, padding = layer
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
                in_channels = out_channels
            elif isinstance(layer, list):
                out_channels = layer[1]
                layers.append(CSPBlock(in_channels, out_channels // 2))
                in_channels = out_channels
            elif isinstance(layer, str):
                if layer == "M":
                    layers.append(nn.MaxPool2d(2))
        return layers


def darknet53_tiny_pretrained(weights, in_channels=3):
    dt_model = SCPDarknet53Tiny()
    dt_model.load_state_dict(torch.load(weights))
    return dt_model


class YoloBody(nn.Module):
    def __init__(self, in_channels=3, B=2, C=1, weights="weights/weightsDarknet.pth"):
        super(YoloBody, self).__init__()
        self.C = C
        self.B = B
        self.darknet = darknet53_tiny_pretrained(weights)
        self.fConv1 = CNNBlock(512, 512,  kernel_size=3, stride=1, padding=1)
        self.fConv2 = CNNBlock(512, 512,  kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(512,  (C + B * 5), kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.darknet(x)
        x = self.fConv1(x)
        x = self.fConv2(x)
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = torch.flatten(x,start_dim=1)
        return x
if __name__ =="__main__":
    x = torch.randn(2,3,416,416)
    #model = Yolov1(in_channels=3,  split_size=7,num_classes=1,num_boxes=2)
    #print(model(x).shape)
    model = YoloBody()
    print(model(x).shape)
    #print(model(x).shape)