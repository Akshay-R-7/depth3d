"""Small multi-task U-Net: one shared encoder-decoder, two heads.

Depth and semantics share features, so each task helps the other
(object boundaries are also depth discontinuities).
"""
import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DepthSegNet(nn.Module):
    def __init__(self, n_classes: int = 4, width: int = 32):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.bottleneck = block(4 * w, 8 * w)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, stride=2)
        self.dec3 = block(8 * w, 4 * w)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = block(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = block(2 * w, w)
        self.depth_head = nn.Conv2d(w, 1, 1)  # predicts log-depth
        self.seg_head = nn.Conv2d(w, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.depth_head(d1)[:, 0], self.seg_head(d1)
