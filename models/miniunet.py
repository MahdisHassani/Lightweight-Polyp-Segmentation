import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LightBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_c, out_c)
        self.conv2 = DepthwiseSeparableConv(out_c, out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)


class MiniUNet(nn.Module):
    def __init__(self, in_c=3, base=24):
        super().__init__()

        self.enc1 = LightBlock(in_c, base)
        self.enc2 = LightBlock(base, base*2)
        self.enc3 = LightBlock(base*2, base*4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = LightBlock(base*4, base*8)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec3 = LightBlock(base*8 + base*4, base*4)
        self.dec2 = LightBlock(base*4 + base*2, base*2)
        self.dec1 = LightBlock(base*2 + base, base)

        self.seg_head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.seg_head(d1)
