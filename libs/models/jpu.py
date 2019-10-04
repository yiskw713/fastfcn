import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(
            self, inplanes, planes, kernel_size=3,
            stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, inplanes, kernel_size,
            stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    """
    Joint Pyramid Upsampling Module proposed in:
    H. Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation
    https://arxiv.org/abs/1903.11816
    """

    def __init__(self, in_channels, width=512):
        super(JPU, self).__init__()
        """
        Args:
            in channels: tuple. in ascending order
        """

        self.convs = []
        self.dilations = []

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels[0], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation0 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.convs.append(self.conv0)
        self.dilations.append(self.dilation0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels[1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation1 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.convs.append(self.conv1)
        self.dilations.append(self.dilation1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels[2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(
                3 * width, width, kernel_size=3,
                padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.convs.append(self.conv2)
        self.dilations.append(self.dilation2)

    def forward(self, *inputs):
        """
        Args:
            inputs: tuple. in order from high resolution feature to low resolution feature
        """
        feats = []

        for input, conv in zip(inputs, self.convs):
            feats.append(conv(input))

        _, _, h, w = feats[0].shape

        for i in range(1, len(feats)):
            feats[i] = F.interpolate(
                feats[i], size=(h, w), mode='bilinear', align_corners=True
            )

        feat = torch.cat(feats, dim=1)

        outputs = []

        for dilation in self.dilations:
            outputs.append(
                dilation(feat)
            )

        outputs = torch.cat(outputs, dim=1)

        return feat
