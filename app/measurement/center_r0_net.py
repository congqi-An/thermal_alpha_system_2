"""CenterR0Net — 全图下采样 256×256 → 双头输出 (cx, cy) + (r0_1, r0_2, r0_3)

动机: 原 CenterNet 用 256×256 裁剪窗只看中心附近 (~128px 半径), 无法感知外层条纹.
     全图下采样后, r0=150~600px(原图) 映射到 ~20~80px(256 空间), 全在视野内.
     CNN 能同时看到条纹间距/对比度的空间分布 → 理解 "哪段半径条纹最稳定".

架构: 共享 encoder (与 CenterNet 同结构, ~400K params) + 双头
  - head_center: GAP → FC → sigmoid → (cx, cy) 归一化 [0,1]
  - head_r0: GAP → FC → sigmoid → (r0_1, r0_2, r0_3) 归一化 [0,1]

输入: (B, 1, 256, 256) 全图下采样, per-image min-max 归一化
输出:
  center: (B, 2) 归一化坐标 → cx = out[0] * W, cy = out[1] * H
  r0s: (B, 3) 归一化半径 → r0 = out * (R_MAX - R_MIN) + R_MIN (原图 px)

训练标签来源 (CV 蒸馏):
  center ← find_radial_center (已有, 可靠)
  r0 ← auto_r0_detect std 排名 top-3 (免费自动标签)

参考文献:
  - Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" CVPR 2018
  - Vandenhende et al. "Multi-Task Learning for Dense Prediction Tasks: A Survey" TPAMI 2022
"""
import numpy as np
import torch
import torch.nn as nn

DS = 256          # 下采样尺寸
R_MIN = 100.0     # r0 下限 (原图 px, 避开中心曲率大区)
R_MAX = 560.0     # r0 上限 (原图 px, 须 < COUNT_PROFILE_LEN-5=595, 防切片越界;
                  #  实况光路条纹强区可达 310-430px, 旧值 280 覆盖不到)


class CenterR0Net(nn.Module):
    """双头网络: 全图下采样 → (cx, cy) + (r0_1, r0_2, r0_3)."""

    def __init__(self):
        super().__init__()
        # 共享 encoder (与原 CenterNet 同结构, 保证 encoder 部分权重可迁移)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),   # 128
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),   # 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),  # 32
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)) # 16

        # 头1: 光心 (cx, cy) 归一化 [0,1]
        self.head_center = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(True),
            nn.Linear(64, 2), nn.Sigmoid())

        # 头2: 三个 r0 归一化 [0,1] → 映射到 [R_MIN, R_MAX]
        self.head_r0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(True),
            nn.Linear(64, 3), nn.Sigmoid())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.enc(x)
        center = self.head_center(f)   # (B, 2)
        r0 = self.head_r0(f)           # (B, 3)
        return center, r0


# ── 推理辅助函数 ──

def normalize_r0(raw_r0):
    """sigmoid 输出 → 原图像素半径, 排序 + 最小间距约束."""
    r0s = np.sort(raw_r0) * (R_MAX - R_MIN) + R_MIN
    # 最小间距 40px (后处理)
    chosen = [r0s[0]]
    for r in r0s[1:]:
        if r - chosen[-1] >= 40:
            chosen.append(r)
        else:
            chosen.append(chosen[-1] + 40)
    return tuple(int(c) for c in chosen)


@torch.no_grad()
def predict_center_r0(red, H, W, model, device):
    """单帧推理: 全图下采样 → (cx, cy, score) + (r0_1, r0_2, r0_3).

    red: [0,1] float32 (H, W) 红通道
    返回: (cx, cy, score), (r0_1, r0_2, r0_3) 全图 px
    """
    import cv2
    small = cv2.resize(red, (DS, DS))
    mn, mx = float(small.min()), float(small.max())
    small = (small - mn) / (mx - mn + 1e-6)

    x = torch.from_numpy(small.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    center_out, r0_out = model(x)
    c = center_out[0].cpu().numpy()
    r = r0_out[0].cpu().numpy()

    cx = float(c[0]) * W
    cy = float(c[1]) * H
    score = 1.0   # 双头网无独立 score, 返回 1.0
    r0s = normalize_r0(r)
    return (cx, cy, score), r0s


def load_pretrained_center_enc(model, center_cnn_path, device):
    """从原 CenterNet checkpoint 迁移 encoder 权重 (冻结/微调可选)."""
    try:
        from .center_cnn import CenterNet
        old = CenterNet().to(device)
        old.load_state_dict(torch.load(center_cnn_path, map_location=device, weights_only=True))
        # 迁移 encoder 权重
        model.enc.load_state_dict(old.enc.state_dict())
        print(f"[CenterR0Net] encoder 权重已从 {center_cnn_path} 迁移")
        return True
    except Exception as e:
        print(f"[CenterR0Net] encoder 迁移失败 (结构不兼容?): {e}")
        return False


if __name__ == "__main__":
    m = CenterR0Net()
    x = torch.randn(2, 1, 256, 256)
    center, r0 = m(x)
    print(f"center: {center.shape} {center[0].detach().numpy()}")
    print(f"r0: {r0.shape} {r0[0].detach().numpy()}")
    print(f"params: {sum(p.numel() for p in m.parameters())}")
