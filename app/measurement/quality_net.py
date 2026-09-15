"""条纹质量评估 CNN — 复用 FringeCenterNet encoder + 质量回归头

输入: (B,1,256,256) 灰度条纹  输出: (B,) 质量分 [0,1] (visibility/清晰度)
用途: grab 线程打分, 低质帧剔除/降权, 提升过零计数鲁棒性
架构: MobileNetV3 InvertedResidual (与 FringeCenterNet 同, ~78K 参数, 实时)
"""
import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=3):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio > 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                   nn.Conv2d(hidden, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class QualityNet(nn.Module):
    def __init__(self, in_channels=1, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.block1 = InvertedResidual(16, 24, stride=2, expand_ratio=3)
        self.block2 = InvertedResidual(24, 40, stride=2, expand_ratio=3)
        self.block3 = InvertedResidual(40, 64, stride=2, expand_ratio=4)
        self.block4 = InvertedResidual(64, 96, stride=2, expand_ratio=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(96, 48), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(48, 32), nn.ReLU(inplace=True))
        self.head = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.stem(x)
        f = self.block1(f); f = self.block2(f); f = self.block3(f); f = self.block4(f)
        f = self.pool(f).flatten(1)
        f = self.fc(f)
        return self.head(f).squeeze(-1)


if __name__ == "__main__":
    m = QualityNet()
    x = torch.randn(2, 1, 256, 256)
    print("quality:", m(x))
    print("params:", sum(p.numel() for p in m.parameters()))
