"""Networks: a U-Net lesion localizer (ResNet-18 encoder) and a ResNet-18 patch classifier with a dropout head."""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models


class _Dec(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNetR18(nn.Module):
    """Input: B x 1 x H x W in [0,1] (H, W divisible by 32). Output: B x 1 x H x W logits of 'lesion here'."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # /2,  64 ch
        self.pool = r.maxpool
        self.e1, self.e2, self.e3, self.e4 = r.layer1, r.layer2, r.layer3, r.layer4   # /4 64, /8 128, /16 256, /32 512
        self.d4, self.d3, self.d2, self.d1 = _Dec(512 + 256, 256), _Dec(256 + 128, 128), _Dec(128 + 64, 64), _Dec(64 + 64, 64)
        self.head = nn.Conv2d(64, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        s0 = self.stem(x)
        s1 = self.e1(self.pool(s0)); s2 = self.e2(s1); s3 = self.e3(s2); s4 = self.e4(s3)
        d = self.d4(torch.cat([self._up(s4, s3), s3], 1))
        d = self.d3(torch.cat([self._up(d, s2), s2], 1))
        d = self.d2(torch.cat([self._up(d, s1), s1], 1))
        d = self.d1(torch.cat([self._up(d, s0), s0], 1))
        return self.head(F.interpolate(d, size=size, mode='bilinear', align_corners=False))


def build_classifier(pretrained=True, dropout=0.3, freeze_early=True):
    """ResNet-18 with a dropout + linear head. Dropout stays available for Monte Carlo uncertainty."""
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if freeze_early:
        for n, p in m.named_parameters():
            if not n.startswith(('layer3', 'layer4', 'fc')):
                p.requires_grad = False
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 1))
    return m
