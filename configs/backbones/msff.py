# msff.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(len(in_channels_list)*out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        resized = []
        target_size = features[-1].shape[2:]  # 最后一个尺度为目标尺度

        for i, f in enumerate(features):
            x = self.convs[i](f)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            resized.append(x)

        fused = torch.cat(resized, dim=1)  # concat across channel
        return self.fusion(fused)
