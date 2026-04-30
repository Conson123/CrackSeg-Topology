"""Atrous Spatial Pyramid Pooling (ASPP) module (Chen et al., 2018).

Five parallel branches capture multi-scale context without reducing spatial
resolution:
  - 1x1 convolution (receptive field = 1)
  - 3x3 atrous convolution, dilation = 6
  - 3x3 atrous convolution, dilation = 12
  - 3x3 atrous convolution, dilation = 18
  - Global average pooling (image-level context)
Branch outputs are concatenated and fused through a 1x1 projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_channels, out_channels=256, dilations=(6, 12, 18)):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = self._atrous_block(in_channels, out_channels, dilations[0])
        self.branch3 = self._atrous_block(in_channels, out_channels, dilations[1])
        self.branch4 = self._atrous_block(in_channels, out_channels, dilations[2])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    @staticmethod
    def _atrous_block(in_ch, out_ch, dilation):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = F.interpolate(
            self.global_branch(x), size=(h, w), mode="bilinear", align_corners=False
        )
        return self.fusion(torch.cat([b1, b2, b3, b4, b5], dim=1))


class ASPPUNet(nn.Module):
    """Insert ASPP between encoder bottleneck and decoder.

    Workflow: encoder -> ASPP (multi-scale enhancement) -> decoder -> head.
    """

    def __init__(self, base_model, aspp_in_channels, aspp_out_channels=256,
                 dilations=(6, 12, 18)):
        super().__init__()
        self.base_model = base_model
        self.aspp = ASPPModule(aspp_in_channels, aspp_out_channels, dilations)
        self.channel_adapt = nn.Sequential(
            nn.Conv2d(aspp_out_channels, aspp_in_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = list(self.base_model.encoder(x))
        features[-1] = self.channel_adapt(self.aspp(features[-1]))
        decoder_output = self.base_model.decoder(*features)
        return self.base_model.segmentation_head(decoder_output)
