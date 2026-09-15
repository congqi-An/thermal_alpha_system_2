"""α 计算: α = N·λ/2 / (L0·ΔT)"""
import numpy as np

from ..config import LASER_WAVELENGTH_M, SAMPLE_LENGTH_M, REFERENCE_ALPHA


def delta_L_from_N(N: float, lam: float = LASER_WAVELENGTH_M) -> float:
    """伸长量 ΔL (m)"""
    return N * lam / 2


def alpha_from_N(N: float, dT: float, L0: float = SAMPLE_LENGTH_M,
                 lam: float = LASER_WAVELENGTH_M) -> float:
    """线胀系数 α (×10^-6 /K)"""
    if dT <= 0:
        return 0.0
    return N * (lam / 2) / (L0 * dT) * 1e6


def alpha_stats(seg_alphas):
    """返回 (mean, std) ×10^-6/K"""
    a = np.asarray(seg_alphas, np.float64)
    if len(a) == 0:
        return 0.0, 0.0
    return float(a.mean()), float(a.std()) if len(a) > 1 else 0.0


def error_vs_reference(alpha: float, ref: float = REFERENCE_ALPHA) -> float:
    """相对误差 %"""
    if ref <= 0:
        return 0.0
    return abs(alpha - ref) / ref * 100
