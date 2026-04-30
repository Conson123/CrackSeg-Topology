"""Attention modules: CBAM, SE, and ECA blocks."""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention with average and max pooling (CBAM component)."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention with concatenated avg/max pooling (CBAM component)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (Hu et al., 2018)."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    """Efficient Channel Attention block."""

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y).expand_as(x)


class AttentionUNet(nn.Module):
    """Wrapper that inserts attention blocks into a segmentation decoder."""

    def __init__(self, base_model, attention_type="cbam", attention_positions=None):
        super().__init__()
        if attention_positions is None:
            attention_positions = ["decoder"]
        self.base_model = base_model
        self.attention_type = attention_type
        self.attention_positions = attention_positions
        if "decoder" in attention_positions:
            self._add_attention_to_decoder()

    def _add_attention_to_decoder(self):
        if not hasattr(self.base_model, "decoder"):
            return
        decoder = self.base_model.decoder
        for block in decoder.blocks:
            if not hasattr(block, "conv1"):
                continue
            in_channels = (
                block.conv1[0].out_channels
                if isinstance(block.conv1, nn.Sequential)
                else block.conv1.out_channels
            )
            att_cls = {"cbam": CBAM, "se": SEBlock, "eca": ECABlock}.get(
                self.attention_type
            )
            if att_cls is None:
                continue
            att_module = att_cls(in_channels)
            original_forward = block.forward

            def _make_forward(att, orig):
                def _forward(x):
                    return att(orig(x))
                return _forward

            block.forward = _make_forward(att_module, original_forward)

    def forward(self, x):
        return self.base_model(x)
