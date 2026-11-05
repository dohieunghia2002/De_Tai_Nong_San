"""DeepLabv3+ with a ResNet-50 backbone, output stride 8, as used by the WeedsGalore authors
(their code: VainF/DeepLabV3Plus-Pytorch). The ImageNet backbone comes from torchvision; the first convolution is
replaced by a randomly initialised 5-channel convolution for multispectral input, as the authors did.
Needs PyTorch (Colab)."""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ASPPConv(nn.Sequential):
    def __init__(self, cin, cout, dilation):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ASPPPooling(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(super().forward(x), size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, cin, rates, cout=256):
        super().__init__()
        branches = [nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))]
        branches += [ASPPConv(cin, cout, r) for r in rates]
        branches.append(ASPPPooling(cin, cout))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(nn.Conv2d(cout * len(branches), cout, 1, bias=False), nn.BatchNorm2d(cout),
                                     nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=5, num_classes=3, pretrained_backbone=True, rates=(12, 24, 36)):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        r = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])  # output stride 8
        if in_channels != 3:
            r.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.low_proj = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.aspp = ASPP(2048, rates)
        self.classifier = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        for m in [*self.low_proj.modules(), *self.aspp.modules(), *self.classifier.modules()]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.layer1(self.stem(x))
        high = self.layer4(self.layer3(self.layer2(low)))
        out = F.interpolate(self.aspp(high), size=low.shape[-2:], mode='bilinear', align_corners=False)
        out = self.classifier(torch.cat([self.low_proj(low), out], dim=1))
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)
