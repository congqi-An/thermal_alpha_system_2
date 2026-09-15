"""光心定位 CNN — 人工标签训练, 裁剪推理. 替代/补充 find_radial_center.

训练见项目根 center_cnn_human.py (48 帧人工标签 + 随机裁剪增强, 全分辨率裁 256).
留出帧评测: CNN L1 误差 16.5px vs find_radial_center 24.8px (CNN 准 33%).
推理: 全分辨率裁 256 around 预期中心(上一帧中心/图像中心) -> CNN 出中心-in-crop
      -> 全分辨率. 跟踪式(用上一帧中心定裁剪窗)精度最佳; 冷启动用图像中心.
与 find_radial_center 并存, 由 state.center_method ("cnn"/"cv") 运行时切换.
"""
import numpy as np
import torch
import torch.nn as nn

DS = 256   # CNN 输入尺寸(全分辨率裁剪窗)


class CenterNet(nn.Module):
    """与根目录 center_cnn_crop.CenterNet 同结构(保证权重兼容). ~40 万参数."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(256, 64), nn.ReLU(True), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.enc(x))   # (B,2) in [0,1] = 中心在裁剪图的归一化坐标


@torch.no_grad()
def find_center_cnn(red, H, W, model, device, prev_cx=None, prev_cy=None):
    """CNN 光心定位. red∈[0,1] float32 (H,W). prev 缺省用图像中心(冷启动).
    返回 (cx, cy) 全分辨率; 裁剪越界/模型空 返回 None(调用方回退 CV)."""
    if model is None:
        return None
    cx0 = prev_cx if prev_cx is not None else W / 2.0
    cy0 = prev_cy if prev_cy is not None else H / 2.0
    o_x = int(np.clip(cx0 - DS / 2, 0, max(W - DS, 0)))
    o_y = int(np.clip(cy0 - DS / 2, 0, max(H - DS, 0)))
    crop = red[o_y:o_y + DS, o_x:o_x + DS]
    if crop.shape[0] < DS or crop.shape[1] < DS:
        return None    # 图小于裁剪窗, 交回 CV
    mn = float(crop.min()); mx = float(crop.max())
    crop = (crop - mn) / (mx - mn + 1e-6)     # 与训练一致 per-crop min-max 归一化
    x = torch.from_numpy(crop.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    p = model(x)[0].cpu().numpy()             # (2,) in [0,1]
    cx = float(p[0]) * DS + o_x
    cy = float(p[1]) * DS + o_y
    return cx, cy
