# -*- coding: utf-8 -*-
"""PhaseFringeCounter — 流式带符号空间相位条纹计数（替代时序过零法）。

原理（离线版已在 3 段真实视频上验证：+18.6 / +1.93 / +59.7 条，三重核验）：
  每帧: 红通道降采样 -> 椭圆极坐标展开(s=r^2 均匀采样) -> 掩膜角向 nanmedian
  -> 径向剖面 -> FFT 窄带解析信号 -> K 个高幅值半径通道相位解缠绕
  -> 两遍 Theil-Sen 拟合 phi(s) -> 截距外推 s=0 => 带符号 N(t)。
  吐/吞方向天然区分，加热初始/末端"又吞又吐"往复自动抵消——这是过零
  计数法（方向盲，往复必虚计）的根本性替代。

流式架构（三层，保证实时性与离线级准确度兼得）：
  1) 增量层（每帧 ~8ms）：增量解缠绕 + 逐帧拟合，实时显示；
  2) 权威层（每 REFIT 帧，~0.1s/万帧）：对本段全部历史剖面重跑离线级
     解算（矢量化 FFT、全程 unwrap、扰动区间聚类再定锚、两遍拟合），
     校正增量层的任何滑移漂移；
  3) 定标层：热身 64 帧临时定标；运行最大值亮度图随条纹移动逐渐完备，
     在里程碑帧重建掩膜/映射/f0/通道并链接计数（热身期条纹静止时
     p90 掩膜会把暗条纹环误判成网格——真实短视频实测过的失效模式）。

与原 FringeCounter 的接口兼容（drop-in）：
  update(red, cx, cy, T, t) / total_N -> (N, per_channel) / reset() /
  r0s / series / n_frames
差异说明：
  - total_N[0] 为 |净条纹数|（浮点含小数，往复已抵消）；带方向值见
    N_signed 属性（>0 = 条纹向内/吞）；
  - r0s/series 定标后由算法自选的 K 个高幅值环带替代固定值，series
  仍按 COUNT_STRIDE 抽稀存环带强度时序。
"""
import numpy as np
import cv2

try:
    from ..config import COUNT_R0S
except ImportError:          # 独立测试/离线验证时无包上下文
    COUNT_R0S = (150, 200, 250)

COUNT_STRIDE = 5        # series 抽稀节奏（与原验证脚本/MC 误差链路一致）

# ── 算法参数（与离线验证版一致）──
_DS = 4                 # 降采样倍率
_N_THETA = 360          # 角向采样
_N_R = 400              # 径向采样（s=r^2 均匀）
_R_MIN = 25.0           # 内边界（降采样 px，避饱和芯区）
_BAND = (0.4, 1.7)      # 带通（相对空间基频 f0）
_K = 9                  # 相位通道数（冗余度支撑滑移仲裁）
_SAT_LEVEL = 225.0 / 255.0
_GRID_DARK = 0.85
_VIGNETTE = 22.0 / 255.0
_WARMUP = 64            # 热身帧数（当前实现为 64, 与 README 对齐; 曾计划 64→16 减启动
                        # N=0 时长与 _make_config 卡顿, 但未落地, 若定标不稳可后续回调）
_MILESTONE = 500        # 里程碑周期（帧）：重估中心漂移 + 热替换掩膜
_REFIT = 200            # 权威层重算周期（帧）
_RESID_GOOD = 0.5       # 拟合残差门限（条）
_ELLIPSE_MIN_REL = 0.02
_Q_BAD = 0.08           # 逐帧增量离散度阈值（扰动区间定位）
_PAD = 6
_C_EMA = 0.02           # 中心慢跟随 EMA 系数（热变形漂移是分钟级慢过程）
_C_RATE = 0.15          # 中心单帧最大移动（降采样 px，防光心输入跳变）
_C_REBUILD = 0.5        # 中心累计偏移超此重建映射（降采样 px）

# ── 时序稳定度/振动 burst 探测 (probe_temporal 标定, 跨场次泛化: 由直播动力学算) ──
# 每帧 badness 由 相位抖动/光心跳变/幅值塌缩 三分量线性映射取 max; 单帧太稀疏
# (干净基线~11% 尖峰), 故 EMA 平滑 + 滞回判 burst。常数取干净中位~calm、振动 p95~bad。
_JIT_CALM, _JIT_BAD = 0.02, 0.18
_CJ_CALM, _CJ_BAD = 4.0, 14.0
_AMP_GOOD, _AMP_BAD = 0.06, 0.02     # amp 低于 _AMP_BAD 全坏, 高于 _AMP_GOOD 正常
_STAB_EMA = 0.15
_BURST_ON, _BURST_OFF = 0.70, 0.22   # 平滑 badness 滞回门 (0.40 在干净段误报, 提到 0.70)
_BURST_ON_N, _BURST_OFF_N = 24, 24   # 持续帧数 (开需 sustained, 干净瞬态不装机)

# ── 振动期相位滑行 (coast): 坏帧冻结测量, 以保持的干净速率 h 推进 N ──
# 双门(抖动 + 增量偏离 h)保证干净帧永不被判坏 → 干净段绝不 coast → α 路安全。
_J_GOOD = 0.12          # 跨通道抖动低于此才算干净帧
_GATE_M = 0.25          # 测量增量偏离 h 超此(条/帧)判坏 (抓整幅刚性跳变)
_COAST_BOOT = 12        # 引导帧数 (建立 h 之前不滑行)
_H_EMA = 0.1            # h 的 EMA 系数 (≈ 前面干净帧速率的稳定平均)
_INERTIA_W = 100         # 惯性窗口: 最近干净帧增量 (坏帧推进速率, 实验用)


def _fill_nan(v):
    bad = ~np.isfinite(v)
    if not bad.any():
        return v
    if bad.all():
        return np.zeros_like(v)
    idx = np.arange(v.size)
    v = v.copy()
    v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def _mavg_rows(A, win):
    """按行滑动均值（A 可为 1D 或 2D [T, n]）。"""
    win = int(win) | 1
    r = win // 2
    A2 = np.atleast_2d(A)
    Ap = np.pad(A2, ((0, 0), (r, r)), mode="reflect")
    c = np.cumsum(np.concatenate([np.zeros((A2.shape[0], 1)), Ap], axis=1),
                  axis=1)
    out = (c[:, win:] - c[:, :-win]) / win
    return out[0] if A.ndim == 1 else out


def _fft_peak(profile, fmin_cyc=3.0):
    v = profile - profile.mean()
    F = np.abs(np.fft.rfft(v * np.hanning(v.size)))
    kmin = int(max(2, np.ceil(fmin_cyc)))
    if kmin >= F.size - 2:
        return 0.0, 0.0
    # argmax 可落在末 bin (F.size-1, 含 Nyquist 能量), F[k+1] 越界; 钳到末有效 bin
    k = min(kmin + int(np.argmax(F[kmin:])), F.size - 2)
    a, b, c = F[k - 1], F[k], F[k + 1]
    den = a - 2 * b + c
    d = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
    return (k + d) / v.size, float(F[k] / (np.median(F[kmin:]) + 1e-9))


def _theil_sen_frame(Y, X, iu, dX):
    """单帧两遍 Theil-Sen -> (intercept, good_mask)。Y: [K] 条。"""
    def fit(yy):
        dY = (yy[None, :] - yy[:, None])[iu]
        with np.errstate(all="ignore"):
            sl = np.nanmedian(dY / dX)
            sl = 0.0 if not np.isfinite(sl) else float(sl)
            ic = np.nanmedian(yy - sl * X)
        return sl, (0.0 if not np.isfinite(ic) else float(ic))

    sl, ic = fit(Y)
    resid = Y - (ic + sl * X)
    good = np.abs(resid) < _RESID_GOOD
    if good.sum() >= 2:
        sl, ic = fit(np.where(good, Y, np.nan))
    return ic, good


class _Config:
    """定标配置（掩膜/映射/f0/通道）+ 剖面历史与解算状态。

    支持慢速中心跟随：映射表 = 以 (0,0) 为中心的基准偏移 + 当前中心标量，
    set_center 仅平移映射并重采掩膜（几何连续，相位无跳变）。
    """

    def __init__(self, bx, by, mask_small, cx, cy, f0, jk, s_norm, amp_k,
                 r0s_full):
        self.bx, self.by = bx, by            # 基准偏移（中心在原点）
        self.mask_small = mask_small          # 降采样像素掩膜（可热替换）
        self.f0 = f0
        self.jk = jk
        self.s_norm = s_norm
        self.amp_k = amp_k
        self.r0s_full = r0s_full
        self.trend_win = max(9, int(round(1.5 / f0)) | 1)
        freqs = np.fft.fftfreq(_N_R)
        self.Hb = ((freqs >= _BAND[0] * f0)
                   & (freqs <= _BAND[1] * f0)).astype(float)
        self.mx = self.my = self.mp = None
        self.cx = self.cy = None
        self.set_center(cx, cy)
        self.P = []              # 本段剖面历史（float32）
        # 增量层状态
        self.phi = None
        self.phi_w = None
        self._phi0 = None        # 段首帧每通道初始相位(cycles) —— N_k 的相对基准
        self.icpt0 = None
        self.N_inc = 0.0         # 增量层 N
        self.N_auth = 0.0        # 权威层 N（refit 时更新）
        self.auth_at = 0
        self.N_k = None
        # coast 状态 (振动期相位滑行)
        self.h = None            # 保持的干净速率 (条/帧)
        self._h_sum = 0.0
        self._ic_prev = None
        self.ngood = 0
        self.coast = 0           # 本段滑行帧数
        self.coast_enabled = False   # 由外层在 _make_config 注入 (默认关=旧行为)
        # 惯性提交 (实验, 默认关=旧行为): 坏帧按最近干净帧均值推进并提交, 恢复重定基防回跳
        self.commit_inertia = False
        self.force_coast = False     # 忽略抖动门, 所有 coast_enabled 帧都 coast (CNN 门控用)
        self.rec = np.zeros(_INERTIA_W)   # 最近干净帧增量环形缓冲
        self.rec_i = 0
        self.rec_n = 0
        self._last_coast = False

    def set_center(self, cx, cy):
        self.cx, self.cy = float(cx), float(cy)
        self.mx = (self.bx + cx).astype(np.float32)
        self.my = (self.by + cy).astype(np.float32)
        self.mp = cv2.remap(self.mask_small.astype(np.float32),
                            self.mx, self.my, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0) > 0.99

    def profile(self, small):
        v = cv2.remap(small, self.mx, self.my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        v[~self.mp] = np.nan
        with np.errstate(all="ignore"):
            return _fill_nan(np.nanmedian(v, axis=0)).astype(np.float32)

    def step(self, prof):
        """增量层：单帧相位步进（实时显示用）。"""
        self.P.append(prof)
        d = prof - _mavg_rows(prof, self.trend_win)
        z = np.fft.ifft(np.fft.fft(d) * self.Hb)[self.jk]
        phi_w = np.angle(z) / (2 * np.pi)
        # 时序诊断(只存标量, 不改解算): 解调幅值 + 帧间相位跨通道抖动
        self._last_amp = float(np.mean(np.abs(z)))
        # 快照仅 coast 开启时需要 (关时零开销、零行为变化)
        phi_snap = self.phi.copy() if (self.coast_enabled and self.phi is not None) else None
        phiw_snap = self.phi_w.copy() if (self.coast_enabled and self.phi_w is not None) else None
        if self.phi is None:
            self.phi = phi_w.copy()
            self.phi_w = phi_w.copy()
            self._phi0 = phi_w.copy()   # 段首帧初始相位: N_k 的相对基准
            self._last_jit = 0.0
        else:
            dphi = phi_w - self.phi_w
            dphi -= np.round(dphi)
            self.phi = self.phi + dphi
            self.phi_w = phi_w
            self._last_jit = float(np.std(dphi - np.median(dphi)))
        X = self.s_norm
        iu = np.triu_indices(len(X), 1)
        dX = (X[None, :] - X[:, None])[iu]
        ic_meas, _ = _theil_sen_frame(self.phi, X, iu, dX)
        if self.icpt0 is None:
            self.icpt0 = ic_meas
        m_valid = self._ic_prev is not None
        m = (ic_meas - self._ic_prev) if m_valid else 0.0   # 本帧测量增量(条/帧)
        # ── coast 双门 (coast_enabled 关 → 恒 accept == 旧行为) ──
        if self.force_coast:                         # Q 门控: Q<0.5 时 coast
            _q_bad = (getattr(self, '_q_hint', None) is not None
                      and self._q_hint < 0.5)
            coast_now = (self.coast_enabled and _q_bad
                         and (self.h is not None or self.rec_n > 0))
        else:
            coast_now = (self.coast_enabled and self.h is not None
                         and (self._last_jit >= _J_GOOD or abs(m - self.h) >= _GATE_M))
        if coast_now:
            self.phi = phi_snap                      # 撤回解绕
            self.phi_w = phiw_snap
            if self.commit_inertia and self.rec_n > 0:
                self.N_inc = self.N_inc + float(self.rec.sum() / _INERTIA_W)  # 惯性推进(sum/100)
            else:
                self.N_inc = self.N_inc + self.h     # 以保持速率滑行
            self.coast += 1
            self._last_coast = True
        else:
            if self.commit_inertia and self._last_coast:
                # 恢复帧: 惯性推进已提交, 重定基保留之, 防快照回跳
                self.icpt0 = ic_meas - self.N_inc
            self.N_inc = ic_meas - self.icpt0        # 累计量, 恢复帧自动纠正
            self._ic_prev = ic_meas
            if m_valid and not self._last_coast:     # 恢复帧(大 m)不污染惯性窗/h
                if self.commit_inertia:              # 干净帧增量入惯性窗
                    self.rec[self.rec_i % _INERTIA_W] = m
                    self.rec_i += 1
                    self.rec_n = min(self.rec_n + 1, _INERTIA_W)
                if self.h is None:                   # 引导期: 累积建 h
                    self._h_sum += m
                    self.ngood += 1
                    if self.ngood >= _COAST_BOOT:
                        self.h = self._h_sum / self.ngood
                elif abs(m - self.h) < _GATE_M:      # 干净速率样本才入 h
                    self.ngood += 1
                    self.h += _H_EMA * (m - self.h)
                # else 恢复帧/离群: 接受累计 N, 不污染 h
            self._last_coast = False
        # 每通道相对段首的累计相位(cycles, 供误差预算层1通道分散度):
        # 修复: 原 `self.phi - self.phi[0] if False else self.phi` 恒为绝对相位
        # (含段首初始偏移), 污染 N_per_channel 与层1分散度; 权威层 refit 后
        # 该式与 N_k = phi[-1]-phi[0] 保持一致。
        self.N_k = (self.phi - self._phi0 if self._phi0 is not None else self.phi)

    # ── 权威层：本段全历史离线级解算 ──
    def refit(self):
        P = np.asarray(self.P, dtype=np.float64)
        T = P.shape[0]
        if T < 16:
            return
        if self.coast > 0:
            # 本段滑行过: P 含坏帧剖面, 顺序 unwrap 必在坏帧滑移 -> 信任增量(已滑行)N, 不重解绕
            self.N_auth = self.N_inc
            self.auth_at = T
            return
        D = P - _mavg_rows(P, self.trend_win)
        Z = np.fft.ifft(np.fft.fft(D, axis=1) * self.Hb[None, :], axis=1)
        phi = np.unwrap(np.angle(Z[:, self.jk]), axis=0) / (2 * np.pi)
        self._reanchor(phi, D)
        X = self.s_norm
        iu = np.triu_indices(len(X), 1)
        dX = (X[None, :] - X[:, None])[iu]
        ic0, _ = _theil_sen_frame(phi[0], X, iu, dX)
        icT, _ = _theil_sen_frame(phi[-1], X, iu, dX)
        self.N_auth = icT - ic0
        self.auth_at = T
        self.N_k = phi[-1] - phi[0]
        # 增量层与权威层对齐（消除累计漂移/滑移）
        self.phi = phi[-1].copy()
        self.phi_w = np.angle(Z[-1, self.jk]) / (2 * np.pi)
        self.icpt0 = ic0 - 0.0
        self.N_inc = self.N_auth

    def _reanchor(self, phi, D):
        """扰动区间通道整数滑移校正（离线验证算法，作用于本段历史）。"""
        T, K = phi.shape
        dN_k = np.diff(phi, axis=0)
        med = np.median(dN_k, axis=1)
        q0 = np.std(dN_k - med[:, None], axis=1)
        bad = np.zeros(T, bool)
        bad[1:] = q0 > _Q_BAD
        idx = np.where(bad)[0]
        if idx.size == 0:
            return
        intervals = []
        s = e = idx[0]
        for i in idx[1:]:
            if i - e <= 2 * _PAD:
                e = i
            else:
                intervals.append((max(0, s - _PAD), min(T - 1, e + _PAD)))
                s = e = i
        intervals.append((max(0, s - _PAD), min(T - 1, e + _PAD)))

        X = self.s_norm
        iu = np.triu_indices(K, 1)
        dX = (X[None, :] - X[:, None])[iu]
        for a, b in intervals:
            delta = phi[b] - phi[a]
            dD = (delta[None, :] - delta[:, None])[iu]
            sl = float(np.median(dD / dX))
            ic = float(np.median(delta - sl * X))
            resid = delta - (ic + sl * X)
            order = np.argsort(resid)
            clusters = [[order[0]]]
            for k in order[1:]:
                if resid[k] - resid[clusters[-1][-1]] < 0.35:
                    clusters[-1].append(k)
                else:
                    clusters.append([k])
            if len(clusters) < 2:
                continue
            cand = self._bridge(D, a, b)
            best, best_score = None, -1.0
            for cl in clusters:
                val = ic + float(np.median(resid[cl]))
                wgt = float(sum(self.amp_k[k] for k in cl))
                support = 0.5
                for dn_c, corr in cand:
                    if abs(dn_c - val) < 0.35:
                        support = max(support, corr)
                score = wgt * support
                if score > best_score:
                    best, best_score = cl, score
            win_res = float(np.median(resid[best]))
            for cl in clusters:
                if cl is best:
                    continue
                for k in cl:
                    adj = round(resid[k] - win_res)
                    if adj != 0:
                        phi[b:, k] -= adj

    def _bridge(self, D, a, b, w=16):
        f0 = self.f0
        if a - w < 0 or b + w >= D.shape[0]:
            return []
        pa = D[a - w:a].mean(axis=0)
        pb = D[b + 1:b + 1 + w].mean(axis=0)
        m = int(2.5 / f0)
        lo, hi = m + 5, _N_R - m - 5
        if hi - lo < 3.0 / f0:
            return []
        va = pa[lo:hi] - pa[lo:hi].mean()
        ls = np.arange(-m, m + 1)
        cc = np.full(ls.size, -1.0)
        for i, l in enumerate(ls):
            vb = pb[lo + l:hi + l]
            vb = vb - vb.mean()
            den = np.sqrt((va * va).sum() * (vb * vb).sum())
            if den > 1e-9:
                cc[i] = float((va * vb).sum() / den)
        peaks, guard = [], int(0.35 / f0)
        for k in np.argsort(cc)[::-1]:
            if cc[k] < 0.6:
                break
            if any(abs(int(ls[k]) - s) < guard for s, _ in peaks):
                continue
            peaks.append((int(ls[k]), cc[k]))
            if len(peaks) >= 4:
                break
        return [(-s * f0, corr) for s, corr in peaks]

    @property
    def N(self):
        """本段当前最优 N：权威层 + 增量层近段延伸。"""
        return self.N_inc


class PhaseFringeCounter:
    """带符号空间相位条纹计数器（原 FringeCounter 的 drop-in 替代）。"""

    def __init__(self, r0s=COUNT_R0S, profile_len=None, ring_mode="cnn"):
        self._init_r0s = tuple(r0s)
        # 采样环模式：“cnn”=强制用外部传入的 CNN r0×3（默认，半径铺得宽、
        # φ(s) 外推基线长、实测最准）；“cv”=忽略 r0_hint，由解调幅值自选环。
        # 与项目前端 center_mode(cnn/cv) 开关联动。
        self.ring_mode = str(ring_mode).lower()
        self.coast_enabled = False   # 振动期相位滑行: 实验性, 干净段验证回归(28.8 vs 16.46),
                                     # 逐帧阈值守不住干净噪声 -> 勿启用; 振动改用"分段排除"
        self.reset()

    # ── 兼容接口 ──
    def reset(self):
        self.r0s = self._init_r0s
        self.series = {r0: [] for r0 in self.r0s}
        self.T_series = []
        self.t_series = []
        self._warm = []
        self._maxbuf = None            # 运行最大值亮度图（掩膜重建）
        self._cxy = []
        self._r0_warm = []             # 热身期收集的 CNN r0 hint（全分辨率 px）
        self._cfg = None               # 当前 _Config
        self._N_closed = 0.0           # 已关闭段的 N 之和（链接）
        self._frame = 0
        self._next_milestone = _WARMUP + _MILESTONE
        self._stride_cnt = 0
        # 时序稳定度/ burst 状态
        self._sb = 0.0
        self.q_stab = 1.0
        self.burst = False
        self._burst_on_n = 0
        self._burst_off_n = 0

    def set_ring_mode(self, mode):
        """切换采样环模式："cnn"(强制 CNN 三环) / "cv"(幅值自选)。

        下一段定标(_make_config)时生效；已定标的当前段不受影响。
        """
        m = str(mode).lower()
        self.ring_mode = "cv" if m == "cv" else "cnn"
        return self.ring_mode

    @property
    def n_frames(self):
        return len(self.T_series)

    @property
    def N_signed(self):
        """带方向净条纹数（>0 = 条纹向内/吞）。"""
        if self._cfg is None:
            return 0.0
        return float(self._cfg.N)

    @property
    def total_N(self):
        """(|净条纹数|, 各通道带符号累计)。往复吞吐已自动抵消。"""
        if self._cfg is None or self._cfg.N_k is None:
            return 0.0, [0] * len(self.r0s)
        return abs(self.N_signed), [round(float(v), 2)
                                    for v in self._cfg.N_k]

    def get_error_stats(self):
        """导出内部统计量供误差预算使用（与实时链路同一方法，无方法不一致）。

        返回 None 表示计数器尚未定标或数据不足。
        返回 dict:
          N_per_channel : K 个通道各自累计 N（refit 后，raw unwrapped phase diff / 2π）
          N_point       : Theil-Sen 截距外推 N（最优估计）
          N_inc         : 增量层 N（含 coast）
          N_auth        : 权威层 N（refit 结果）
          auth_at       : 权威层重算时的帧序号
          s_norm        : K 个通道的归一化 s 坐标
          amp_k         : K 个通道的解调幅值
          n_frames      : 本段总帧数
          coast_frames  : 本段滑行帧数
          r0s_full      : 采样环全分辨率像素半径
          ring_mode     : 采样环模式
        """
        if self._cfg is None:
            return None
        cfg = self._cfg
        N_k = cfg.N_k
        if N_k is None or len(N_k) < 2:
            return None
        return {
            'N_per_channel': [round(float(v), 4) for v in N_k],
            'N_point': round(float(self.N_signed), 4),
            'N_inc': round(float(cfg.N_inc), 4),
            'N_auth': round(float(cfg.N_auth), 4),
            'auth_at': cfg.auth_at,
            's_norm': [round(float(s), 4) for s in cfg.s_norm],
            'amp_k': [round(float(a), 4) for a in cfg.amp_k],
            'n_frames': len(cfg.P),
            'coast_frames': cfg.coast,
            'r0s_full': cfg.r0s_full,
            'ring_mode': self.ring_mode,
        }

    # ── 主入口 ──
    def _update_stability(self):
        """逐帧时序稳定度 + 滞回 burst (供 center 门 / 告警 / 分段标记)。

        badness = max(相位抖动, 光心跳变, 幅值塌缩) 三分量[0,1]; EMA 平滑得 _sb;
        q_stab = 1 - _sb (calm≈1, burst≈0)。滞回: _sb 超 _BURST_ON 持续 _BURST_ON_N
        帧开, 低于 _BURST_OFF 持续 _BURST_OFF_N 帧关。干净段 _sb 难越 0.4→不误报。
        """
        jit = getattr(self._cfg, "_last_jit", 0.0)
        amp = getattr(self._cfg, "_last_amp", 0.0)
        cj = getattr(self._cfg, "_last_cjump", 0.0)
        b_j = float(np.clip((jit - _JIT_CALM) / (_JIT_BAD - _JIT_CALM), 0, 1))
        b_cj = float(np.clip((cj - _CJ_CALM) / (_CJ_BAD - _CJ_CALM), 0, 1))
        b_am = float(np.clip((_AMP_GOOD - amp) / (_AMP_GOOD - _AMP_BAD), 0, 1))
        b = max(b_j, b_cj, b_am)
        self._sb += (b - self._sb) * _STAB_EMA
        self.q_stab = float(1.0 - min(self._sb, 1.0))
        if not self.burst:
            self._burst_on_n = self._burst_on_n + 1 if self._sb > _BURST_ON else 0
            if self._burst_on_n >= _BURST_ON_N:
                self.burst = True
                self._burst_off_n = 0
        else:
            self._burst_off_n = self._burst_off_n + 1 if self._sb < _BURST_OFF else 0
            if self._burst_off_n >= _BURST_OFF_N:
                self.burst = False
                self._burst_on_n = 0

    def update(self, red, cx, cy, T: float = 0.0, t: float = 0.0,
               r0_hint=None, q_hint=None):
        """red: [0,1] float32 全分辨率红通道；cx, cy: 光心（全分辨率 px）。

        r0_hint: 可选，CNN 三头网的 r0×3（全分辨率 px）。提供时热身期收集其
        中值作为相位计数采样环（实测：它们半径铺得宽、φ(s) 外推基线长，
        比幅值自选通道更准）；不提供时回退到幅值自选。
        """
        H, W = red.shape
        small = cv2.resize(red, (W // _DS, H // _DS),
                           interpolation=cv2.INTER_AREA)
        if self._maxbuf is None:
            self._maxbuf = small.copy()
        else:
            np.maximum(self._maxbuf, small, out=self._maxbuf)
        self._frame += 1
        self._q_hint = q_hint
        self.T_series.append(T)
        self.t_series.append(t)

        if self._cfg is None:
            self._warm.append(small.astype(np.float16))
            self._cxy.append((cx / _DS, cy / _DS))
            if r0_hint is not None and len(r0_hint) >= 2:
                self._r0_warm.append(sorted(float(r) for r in r0_hint))
            if len(self._warm) >= _WARMUP:
                self._make_config(np.stack(self._warm).astype(np.float32))
                self._warm = []
            return

        prof = self._cfg.profile(small)
        self._cfg._q_hint = q_hint
        self._cfg.step(prof)

        # 中心跟随（信任外部 CNN 光心，轻 EMA 去亚像素抖动）：
        # 累计偏移超阈才平移映射（几何连续，相位无跳）。
        cxs, cys = cx / _DS, cy / _DS
        # 时序诊断: 本帧 CNN 光心相对运行 EMA 的跳变(振动 kick 时大)
        self._cfg._last_cjump = float(np.hypot(cxs - self._cx_ema, cys - self._cy_ema))
        self._update_stability()
        self._cx_ema += (cxs - self._cx_ema) * _C_EMA
        self._cy_ema += (cys - self._cy_ema) * _C_EMA
        if (abs(self._cx_ema - self._cfg.cx) > _C_REBUILD
                or abs(self._cy_ema - self._cfg.cy) > _C_REBUILD):
            self._cfg.set_center(self._cx_ema, self._cy_ema)

        # series 抽稀（下游 MC 误差链路兼容）
        self._stride_cnt += 1
        if self._stride_cnt % COUNT_STRIDE == 1:
            for r0, j in zip(self.r0s, self._cfg.jk):
                self.series[r0].append(float(prof[j]))

        # 权威层重算（全历史，含再定锚）
        if len(self._cfg.P) % _REFIT == 0:
            self._cfg.refit()

        # 里程碑：热替换掩膜（运行最大值图随条纹移动逐渐完备）。
        if self._frame >= self._next_milestone:
            self._next_milestone += _MILESTONE
            self._cfg.mask_small = self._mask_from(self._maxbuf)
            old_valid = float(self._cfg.mp.mean())
            self._cfg.set_center(self._cfg.cx, self._cfg.cy)   # 重采新掩膜
            self._cfg.refit()
            print(f"[PhaseFringeCounter] mask refresh @frame {self._frame}: "
                  f"polar valid {old_valid:.2f} -> {float(self._cfg.mp.mean()):.2f}")

    # ── 定标 ──
    def _make_config(self, frames, mask_override=None):
        cx = float(np.median([c[0] for c in self._cxy]))
        cy = float(np.median([c[1] for c in self._cxy]))
        Hs, Ws = frames.shape[1:]
        if mask_override is not None:
            mask = mask_override
        else:
            bright = np.maximum(np.percentile(frames, 90, axis=0),
                                self._maxbuf)
            mask = self._mask_from(bright)

        r_max = min(cx, cy, Ws - 1 - cx, Hs - 1 - cy) - 4
        theta = np.linspace(0, 2 * np.pi, _N_THETA, endpoint=False)
        r_j = np.sqrt(np.linspace(_R_MIN ** 2, r_max ** 2, _N_R))
        cos_t = np.cos(theta).reshape(-1, 1)
        sin_t = np.sin(theta).reshape(-1, 1)

        def maps(u):
            uu = np.asarray(u, dtype=float).reshape(-1, 1)
            mx = (cx + cos_t * uu * r_j).astype(np.float32)
            my = (cy + sin_t * uu * r_j).astype(np.float32)
            mp = cv2.remap(mask.astype(np.float32), mx, my,
                           cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0) > 0.99
            return mx, my, mp

        # 椭圆度（分扇区频率 -> 2theta 拟合）
        mx, my, mp = maps(np.ones(_N_THETA))
        ref = frames[len(frames) // 2]
        v = cv2.remap(ref, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        v[~mp] = np.nan
        per = _N_THETA // 12
        th_c, fr_c = [], []
        with np.errstate(all="ignore"):
            for si in range(12):
                p = np.nanmedian(v[si * per:(si + 1) * per], axis=0)
                if not np.isfinite(p).any():
                    continue
                p = _fill_nan(p)
                f, snr = _fft_peak(p - _mavg_rows(p, 55))
                if f > 0 and snr > 4:
                    th_c.append((si + 0.5) * 2 * np.pi / 12)
                    fr_c.append(f)
        u = np.ones(_N_THETA)
        if len(fr_c) >= 6:
            th_c, fr_c = np.array(th_c), np.array(fr_c)
            M = np.stack([np.ones_like(th_c), np.cos(2 * th_c),
                          np.sin(2 * th_c)], axis=1)
            A, B, C = np.linalg.lstsq(M, fr_c, rcond=None)[0]
            if np.hypot(B, C) / A >= _ELLIPSE_MIN_REL:
                ff = A + B * np.cos(2 * theta) + C * np.sin(2 * theta)
                u = np.sqrt(A / np.clip(ff, 1e-6, None))
                u /= u.mean()
        mx, my, mp = maps(u)

        # 剖面栈 -> f0 / 通道
        profs = []
        for f in frames:
            pv = cv2.remap(f, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=np.nan)
            pv[~mp] = np.nan
            with np.errstate(all="ignore"):
                profs.append(_fill_nan(np.nanmedian(pv, axis=0)))
        P = np.array(profs)
        med_prof = np.median(P, axis=0)
        f0, snr = _fft_peak(med_prof - _mavg_rows(med_prof, 55))
        if f0 <= 0:
            f0 = 0.025
        trend_win = max(9, int(round(1.5 / f0)) | 1)
        D = P - _mavg_rows(P, trend_win)
        freqs = np.fft.fftfreq(_N_R)
        Hb = ((freqs >= _BAND[0] * f0) & (freqs <= _BAND[1] * f0)).astype(float)
        Z = np.fft.ifft(np.fft.fft(D, axis=1) * Hb[None, :], axis=1)
        amp = np.median(np.abs(Z), axis=0)
        level = np.median(P, axis=0)
        cover = mp.mean(axis=0)

        margin = int(round(1.0 / f0))
        jk = None
        # cnn 模式(默认)：强制 CNN r0×3 全部作计数通道（仅越界钳制+去重，
        # 不做饱和/覆盖过滤）。实测过滤会丢内环 -> 只剩大半径环 ->
        # φ(s) 外推基线短 -> 漏计；CNN r0 本就是它跟踪的真实干涉环，直接信任。
        # cv 模式：跳过此块，走下方幅值自选。
        if self.ring_mode == "cnn" and len(self._r0_warm) >= max(3, _WARMUP // 4):
            r0_med = np.median(np.array(self._r0_warm), axis=0) / _DS  # ds px
            cj = sorted(set(int(np.clip(np.argmin(np.abs(r_j - r)),
                                        1, _N_R - 2)) for r in r0_med))
            if len(cj) >= 2:
                jk = np.array(cj)
                sel_src = "cnn-r0(forced)"
                print(f"[PhaseFringeCounter] force CNN rings: r0_med(ds)="
                      f"{np.round(r0_med, 1).tolist()} -> jk={cj} "
                      f"level={np.round(level[cj], 1).tolist()} "
                      f"cover={np.round(cover[cj], 2).tolist()}", flush=True)
        # 回退 / cv 模式：幅值自选（强制半径铺开）
        if jk is None:
            ok = np.ones(_N_R, bool)
            ok[:margin] = False
            ok[int(0.65 * _N_R):] = False
            ok &= level < _SAT_LEVEL
            ok &= cover > 0.6
            if ok.any():
                ok &= amp > 0.45 * amp[ok].max()
            cand = np.where(ok)[0]
            if cand.size < _K:
                cand = np.sort(np.argsort(amp[:int(0.65 * _N_R)])[::-1]
                               [:max(_K, 10)])
            groups = np.array_split(cand, _K)
            jk = np.array(sorted(int(g[np.argmax(amp[g])])
                                 for g in groups if g.size))
            sel_src = "amp-auto"

        cfg = _Config((cos_t * u.reshape(-1, 1) * r_j).astype(np.float32),
                      (sin_t * u.reshape(-1, 1) * r_j).astype(np.float32),
                      mask, cx, cy, f0, jk,
                      (r_j[jk] ** 2) / (r_j[-1] ** 2), amp[jk],
                      tuple(int(round(r_j[j] * _DS)) for j in jk))
        self._cfg = cfg
        cfg.coast_enabled = True    # Q 驱动惯性滑行
        cfg.commit_inertia = True   # 坏帧用惯性推进
        cfg.force_coast = True      # Q<0.5 时 coast
        self._cx_ema = cx
        self._cy_ema = cy
        self.r0s = cfg.r0s_full
        self.series = {r0: self.series.get(r0, []) for r0 in self.r0s}
        print(f"[PhaseFringeCounter] config @frame {self._frame}: "
              f"center=({cx:.1f},{cy:.1f}) f0={f0:.4f} (snr={snr:.0f}) "
              f"mask_valid={mask.mean():.2f} sel={sel_src} "
              f"r0s(full px)={self.r0s}")
        # 回放定标样本帧，期间条纹变化不丢
        for p in P:
            cfg.step(p.astype(np.float32))
        cfg.refit()

    def _mask_from(self, bright):
        local = cv2.medianBlur((np.clip(bright, 0, 1) * 255).astype(np.uint8),
                               31).astype(np.float32) / 255.0
        return ~((bright < _GRID_DARK * local) | (bright < _VIGNETTE))

# drop-in 别名：import 侧可直接 `from ... import FringeCounter`
FringeCounter = PhaseFringeCounter

