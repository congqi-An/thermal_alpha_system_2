"""FrameAnalysisNet — 三头统一网络: 光心 + ROI半径 + 帧质量

将 CenterR0Net 和 QualityNet 合并为单次前向推理:
  输入: (B, 1, 256, 256) 全图下采样灰度帧
  主 encoder (空间结构): Conv(1->32->64->128->256), stride=2 x4
    ├─ head_center v2: 空间热图 + soft-argmax → (cx, cy)
    │    (v1 的 GAP+FC 回归把空间维平均掉, 定位天花板实测 ~136px;
    │     热图保留位置信息, softmax 期望做亚格点插值, 2026-07-29 换)
    └─ head_r0:      GAP -> FC(256,64) -> FC(64,3) -> Sigmoid  → (r0_1, r0_2, r0_3)
  QualityNet (纹理质量): 原 MobileNetV3 InvertedResidual 架构
    └─ quality: stem -> block1~4 -> GAP -> FC -> sigmoid → [0,1]

checkpoint: 现行权重由 minicpm/train_frame_analysis.py 用真实人工标签训练。
"""
import numpy as np
import torch
import torch.nn as nn

DS = 256          # 下采样尺寸
# r0 范围与 CenterR0Net 统一 (单一定义源, 防双份漂移)
from .center_r0_net import R_MIN, R_MAX


# ── QualityNet 组件 (原封不动搬入) ──

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


class FrameAnalysisNet(nn.Module):
    """三头网络: 一次 forward 出 (cx, cy) + (r0_1, r0_2, r0_3) + quality."""

    def __init__(self):
        super().__init__()
        # ── 主 encoder: 空间结构 (CenterR0Net 同结构) ──
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))

        # 光心头 v2: 空间热图 (16x16 logits) + soft-argmax
        self.head_center = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 1))
        # 格点中心坐标 [0,1] (惰性按特征图尺寸缓存)
        self._grid_hw = None
        self._grid_x = None
        self._grid_y = None

        self.head_r0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(True),
            nn.Linear(64, 3), nn.Sigmoid())

        # ── QualityNet: 纹理质量 (原架构, 已训好权重) ──
        self.q_stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.q_block1 = InvertedResidual(16, 24, stride=2, expand_ratio=3)
        self.q_block2 = InvertedResidual(24, 40, stride=2, expand_ratio=3)
        self.q_block3 = InvertedResidual(40, 64, stride=2, expand_ratio=4)
        self.q_block4 = InvertedResidual(64, 96, stride=2, expand_ratio=4)
        self.q_pool = nn.AdaptiveAvgPool2d(1)
        self.q_fc = nn.Sequential(
            nn.Linear(96, 48), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(48, 32), nn.ReLU(inplace=True))
        self.q_head = nn.Linear(32, 1)

    def _soft_argmax(self, hm):
        """(B,1,h,w) logits → (B,2) 坐标 [0,1]: softmax 期望, 亚格点精度."""
        B, _, h, w = hm.shape
        if self._grid_hw != (h, w) or self._grid_x.device != hm.device:
            xs = (torch.arange(w, device=hm.device, dtype=hm.dtype) + 0.5) / w
            ys = (torch.arange(h, device=hm.device, dtype=hm.dtype) + 0.5) / h
            self._grid_y, self._grid_x = torch.meshgrid(ys, xs, indexing="ij")
            self._grid_hw = (h, w)
        p = torch.softmax(hm.flatten(1), dim=1).view(B, h, w)
        cx = (p * self._grid_x).sum(dim=(1, 2))
        cy = (p * self._grid_y).sum(dim=(1, 2))
        return torch.stack([cx, cy], dim=1)

    def forward(self, x):
        # 主 encoder → center (热图 soft-argmax) + r0
        f = self.enc(x)
        center = self._soft_argmax(self.head_center(f))
        r0 = self.head_r0(f)
        # QualityNet → quality [0,1]
        qf = self.q_stem(x)
        qf = self.q_block1(qf)
        qf = self.q_block2(qf)
        qf = self.q_block3(qf)
        qf = self.q_block4(qf)
        qf = self.q_pool(qf).flatten(1)
        qf = self.q_fc(qf)
        quality = torch.sigmoid(self.q_head(qf)).squeeze(-1)
        return center, r0, quality


# ── 推理辅助函数 ──

def normalize_r0(raw_r0):
    """sigmoid 输出 → 原图像素半径, 排序 + 最小间距约束."""
    r0s = np.sort(raw_r0) * (R_MAX - R_MIN) + R_MIN
    chosen = [r0s[0]]
    for r in r0s[1:]:
        if r - chosen[-1] >= 40:
            chosen.append(r)
        else:
            chosen.append(chosen[-1] + 40)
    return tuple(int(c) for c in chosen)


@torch.no_grad()
def predict_frame(red, H, W, model, device, return_raw=False):
    """单帧推理: 全图下采样 → (cx, cy), r0s, quality.

    return_raw=True 时额外返回 r0 头 raw sigmoid 输出 (用于退化检测).
    """
    import cv2
    small = cv2.resize(red, (DS, DS))
    mn, mx = float(small.min()), float(small.max())
    small = (small - mn) / (mx - mn + 1e-6)

    x = torch.from_numpy(small.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    center_out, r0_out, q_out = model(x)

    c = center_out[0].cpu().numpy()
    r = r0_out[0].cpu().numpy()
    q = float(q_out[0].item())

    cx = float(c[0]) * W
    cy = float(c[1]) * H
    r0s = normalize_r0(r)
    if return_raw:
        return (cx, cy), r0s, q, r
    return (cx, cy), r0s, q


def r0_degenerate(r_raw) -> bool:
    """判断 r0 头 raw sigmoid 输出是否退化 (塌缩/饱和贴边).

    塌缩: 三输出几乎相同 (极差<0.05), normalize_r0 会强制 +40 撑开掩盖问题;
    饱和: 均值贴 0/1 边界, 说明输入分布偏移导致输出漂到边界.
    """
    r = np.asarray(r_raw, np.float32)
    return bool((r.max() - r.min()) < 0.05 or r.mean() < 0.08 or r.mean() > 0.92)


def load_frame_analysis_net(checkpoint_path, device="cpu"):
    """加载合并后的 FrameAnalysisNet."""
    from pathlib import Path
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    model = FrameAnalysisNet()
    model.load_state_dict(torch.load(str(p), map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def merge_checkpoints(center_r0_net_path, quality_net_path, output_path, device="cpu"):
    """[已过时] 将 CenterR0Net + QualityNet 合并为 v1 FrameAnalysisNet.

    仅适用于 v1 GAP+FC 光心头; v2 热图头与 CenterR0Net 的 head_center
    键不兼容 (head_center.load_state_dict 会失败), 现行权重由
    minicpm/train_frame_analysis.py 直接训练产出.
    """
    from pathlib import Path
    model = FrameAnalysisNet()

    # 1. 迁移 CenterR0Net (enc + head_center + head_r0)
    crn_ckpt = torch.load(str(center_r0_net_path), map_location=device, weights_only=True)
    enc_state = {k.replace("enc.", "", 1): v for k, v in crn_ckpt.items() if k.startswith("enc.")}
    model.enc.load_state_dict(enc_state)
    center_state = {k.replace("head_center.", "", 1): v for k, v in crn_ckpt.items() if k.startswith("head_center.")}
    model.head_center.load_state_dict(center_state)
    r0_state = {k.replace("head_r0.", "", 1): v for k, v in crn_ckpt.items() if k.startswith("head_r0.")}
    model.head_r0.load_state_dict(r0_state)
    print(f"  [OK] CenterR0Net weights loaded from {Path(center_r0_net_path).name}")

    # 2. 迁移 QualityNet (stem/block1~4/pool/fc/head -> q_stem/q_block1~4/q_pool/q_fc/q_head)
    qn_ckpt = torch.load(str(quality_net_path), map_location=device, weights_only=True)
    # QualityNet key mapping: stem.* -> q_stem.*, block1.* -> q_block1.*, etc.
    q_mapping = {
        "stem": "q_stem", "block1": "q_block1", "block2": "q_block2",
        "block3": "q_block3", "block4": "q_block4", "pool": "q_pool",
        "fc": "q_fc", "head": "q_head",
    }
    q_state = {}
    for k, v in qn_ckpt.items():
        for old_prefix, new_prefix in q_mapping.items():
            if k.startswith(old_prefix + "."):
                new_key = k.replace(old_prefix + ".", new_prefix + ".", 1)
                q_state[new_key] = v
                break
    model.load_state_dict(q_state, strict=False)
    print(f"  [OK] QualityNet weights loaded from {Path(quality_net_path).name}")

    # 3. 保存合并模型
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_path))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Merged model saved: {output_path} ({n_params:,} params)")
    return model


if __name__ == "__main__":
    m = FrameAnalysisNet()
    x = torch.randn(2, 1, 256, 256)
    center, r0, quality = m(x)
    print(f"center: {center.shape} {center[0].detach().numpy()}")
    print(f"r0: {r0.shape} {r0[0].detach().numpy()}")
    print(f"quality: {quality.shape} {quality.detach().numpy()}")
    print(f"params: {sum(p.numel() for p in m.parameters()):,}")
