"""径向对称投票找等倾中心 — 吸收热漂移

源: newton_rings_count_accurate.py / newton_rings_mark_one.py (已验证, corr 0.977)
v2: 峰谷一体投票 (局部对比度 mask, 亮环暗环过渡带都参与)
    + 峰谷镜像对称精化 refine_center_symmetry
"""
import numpy as np
import cv2


def find_radial_center(red: np.ndarray, H: int, W: int, ds: int = 256):
    """返回 (cx, cy, score)。red 为归一化 [0,1] 灰度 float32。

    v2 改进: 投票 mask 用局部对比度(条纹区特征)替代亮度阈值 —
    波峰/波谷的过渡带都是高局部对比度, 一起投票;
    孤立强梯度噪点(散斑/杂光边缘)局部对比度低, 被排除 → 抗干扰更强。"""
    small = cv2.resize(red, (ds, ds))
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.hypot(gx, gy)
    # 峰谷一体 mask: 条纹区 = 局部对比度高 (亮环+暗环过渡带都算),
    # 背景弱纹理/孤立噪点被排除
    mean_f = cv2.boxFilter(small, -1, (15, 15))
    sq_f = cv2.boxFilter(small * small, -1, (15, 15))
    local_std = np.sqrt(np.maximum(sq_f - mean_f * mean_f, 0))
    fringe = local_std > np.percentile(local_std, 70)
    thr = np.percentile(mag, 80)
    mask = (mag > thr) & fringe
    if int(mask.sum()) < 20:
        mask = mag > thr       # fallback
    nrm = np.where(mag > 0, mag, 1)
    ux = gx / nrm
    uy = gy / nrm
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return W / 2, H / 2, 0.0
    acc = np.zeros_like(small)
    for R in (15, 25, 40, 60):
        for s in (1, -1):
            cxv = np.clip((xs + s * ux[ys, xs] * R).round().astype(int), 0, ds - 1)
            cyv = np.clip((ys + s * uy[ys, xs] * R).round().astype(int), 0, ds - 1)
            np.add.at(acc, (cyv, cxv), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), 3)
    # 限制中心在中央区域(排除边缘角点误判)
    m = int(ds * 0.12)
    acc[:m, :] = 0; acc[-m:, :] = 0; acc[:, :m] = 0; acc[:, -m:] = 0
    cy, cx = np.unravel_index(np.argmax(acc), acc.shape)
    return cx * W / ds, cy * H / ds, float(acc.max())


def _sym_score_1d(profile: np.ndarray, c: int, half: int) -> float:
    """位置 c 的镜像对称得分: 左段翻转与右段的归一化互相关。
    峰和谷的全部结构都参与匹配 (峰对峰、谷对谷)。"""
    left = profile[max(0, c - half):c][::-1]
    right = profile[c + 1:c + 1 + half]
    n = min(len(left), len(right))
    if n < 20:
        return -2.0
    l = left[:n] - left[:n].mean()
    r = right[:n] - right[:n].mean()
    denom = float(np.sqrt((l * l).sum()) * np.sqrt((r * r).sum())) + 1e-9
    return float((l * r).sum() / denom)


def refine_center_symmetry(red: np.ndarray, cx: float, cy: float,
                           H: int, W: int, ds: int = 512, search: int = 40,
                           accept_score: float = 0.55):
    """峰谷镜像对称精化: 干涉环剖面关于真中心镜像对称 (峰对峰、谷对谷)。

    关键: 先去除光斑高斯包络(否则会锁到光斑亮度峰而非条纹环中心),
    再在纯条纹振荡结构上寻优。剖面取 ±6px 多行/列平均降噪;
    对称得分 < accept_score 时拒绝精化原值返回。
    33 帧实况人工真值评估: 中位 22px / 均值 62px (优于 CNN 65px 与粗投票 114px)。

    返回 (cx, cy, score)。"""
    small = cv2.resize(red, (ds, ds))
    small = cv2.GaussianBlur(small, (0, 0), 2)
    # 去光斑包络: 减去大核高斯背景, 只留条纹振荡结构
    # (否则对称寻优会锁到光斑亮度峰而非条纹环中心)
    env = cv2.GaussianBlur(small, (0, 0), 30)
    small = small - env
    sx = int(np.clip(cx * ds / W, 0, ds - 1))
    sy = int(np.clip(cy * ds / H, 0, ds - 1))
    half = ds // 3
    best_total = -2.0
    for _ in range(2):
        # 沿 x: y=sy 附近 ±6 行平均剖面寻优
        row = small[max(0, sy - 6):sy + 7].mean(axis=0)
        best, best_c = -2.0, sx
        for c in range(max(20, sx - search), min(ds - 20, sx + search) + 1):
            s = _sym_score_1d(row, c, half)
            if s > best:
                best, best_c = s, c
        sx = best_c
        # 沿 y: x=sx 附近 ±6 列平均剖面寻优
        col = small[:, max(0, sx - 6):sx + 7].mean(axis=1)
        best2, best_c2 = -2.0, sy
        for c in range(max(20, sy - search), min(ds - 20, sy + search) + 1):
            s = _sym_score_1d(col, c, half)
            if s > best2:
                best2, best_c2 = s, c
        sy = best_c2
        best_total = min(best, best2)
    if best_total < accept_score:   # 对称性不足 → 拒绝精化, 保留原值
        return cx, cy, best_total
    return sx * W / ds, sy * H / ds, best_total


def auto_r0_detect(red, cx, cy, n_r0=3, r_max=700, bin_w=20, ds=256):
    """算径向条纹能量(std), 自适应选条纹强区 n_r0 个 r0(下采样加速)。
    取流时实时调用, 显示自适应 ROI, 不依赖 measure。"""
    H, W = red.shape
    small = cv2.resize(red, (ds, ds))
    sx, sy = cx * ds / W, cy * ds / H
    ys, xs = np.indices(small.shape)
    r = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
    rmax_s = r_max * ds / min(W, H)
    bin_s = bin_w * ds / min(W, H)
    nb = int(rmax_s / bin_s)
    std_p = np.zeros(nb)
    for i in range(nb):
        m = (r >= i * bin_s) & (r < (i + 1) * bin_s)
        if int(m.sum()) > 20:
            std_p[i] = small[m].std()
    idx = np.argsort(std_p)[::-1]
    chosen = []
    for i in idx:
        rr = (i * bin_s + bin_s / 2) * min(W, H) / ds   # 映射回原图
        if rr < 60:
            continue
        if all(abs(rr - c) > 40 for c in chosen):
            chosen.append(rr)
        if len(chosen) >= n_r0:
            break
    return tuple(sorted(int(c) for c in chosen)) if len(chosen) >= n_r0 else (150, 200, 250)
