import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from jpu import JPU


class FastFCN(nn.Module):
    """
        Fast FCN with VGG backbone.
        Originally, Fast FCN has ResNet101 as a backbone.
        Please refer to:
            H. Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation
            https://arxiv.org/abs/1903.11816
    """

    def __init__(self, n_classes, jpu_in_channels=(256, 512, 512), width=512):
        super().__init__()
        vgg = torchvision.models.vgg16_bn(pretrained=True).features

        # confirm the architecture of vgg16 by "print(vgg)"

        self.pool3 = vgg[:24]
        self.pool4 = vgg[24:34]
        self.pool5 = vgg[34:]

        self.jpu = JPU(jpu_in_channels, width)

        self.Conv_end1 = nn.Conv2d(
            len(jpu_in_channels) * width, 256, 3, 1, 1, bias=False)
        self.bn_end1 = nn.BatchNorm2d(256)
        self.conv_end2 = nn.Conv2d(
            in_channels=256, out_channels=n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        _, _, h, w = x.shape

        # vgg16
        x3 = self.pool3(x)     # output size => (N, 256, H/8, W/8)
        x4 = self.pool4(x3)    # output size => (N, 512, H/16, W/16)
        x5 = self.pool5(x4)    # output size => (N, 512, H/32, W/32)

        x = self.jpu(x3, x4, x5)
        x = self.Conv_end1(x)
        x = self.bn_end1(x)
        x = self.conv_end2(x)
        result = F.interpolate(
            x, size=(h, w), mode='bilinear', align_corners=True)

        return result


# for debug
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = torch.randn(10, 3, 224, 224).to(device)

    model = FastFCN(n_classes=10).to(device)
    y = model(img)

    print(y.shape)
    print("Done")
