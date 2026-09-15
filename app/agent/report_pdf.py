# -*- coding: utf-8 -*-
"""实验报告 PDF 生成 — reportlab 排版 + matplotlib 公式/图表

设计 (竞赛报告规格):
- A4 (210x297mm), 四边距 20mm; 样式基于 getSampleStyleSheet() 定制:
  标题 18pt 居中段后 20pt / 一级标题 14pt 加粗段前后 10pt / 正文 11pt 1.5 倍行距 /
  表格单元格 10pt 居中 padding 6pt
- 内容: 实验标题 → 目的 → 原理(公式图) → 仪器(表) → 步骤(9阶段) →
  数据处理与结果(参数/结果/计算表 + 8 张实验过程图) → 结论(LLM) → VLM 摘要表
- 8 张图全部来自实验过程: 粗调光点照片 / 细调干涉条纹帧 / T(t) / N(t) /
  N-ΔT 离散 / α 柱状 / φ(s) 外推 / 通道分散度
- 公式: matplotlib 渲染 PNG (优先 usetex 真 LaTeX, 无 pdflatex 时 mathtext 降级)
- 中文字体: SimHei, 缺失回退 Microsoft YaHei
"""
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ── 实验常量 ──
LAMBDA_NM = 632.8            # He-Ne 波长 nm
HALF_LAMBDA_NM = LAMBDA_NM / 2   # 每条纹对应伸长 316.4nm
L0_MM = 150.0
REF_ALPHA = 20.8             # 黄铜 H62 参考值 ×10^-6/K

# ── 中文字体 (SimHei → Microsoft YaHei 回退) ──
_FONT_CANDIDATES = [r"C:/Windows/Fonts/simhei.ttf",
                    r"C:/Windows/Fonts/msyh.ttf",
                    r"C:/Windows/Fonts/msyh.ttc"]
_FONT_PATH = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
_FONT_NAME = ("SimHei" if _FONT_PATH and "simhei" in _FONT_PATH.lower() else "YaHei")

_latex_ok = None


def latex_available() -> bool:
    """探测 pdflatex (MiKTeX/TeX Live). 服务进程可能继承安装前旧 PATH, 补常见路径."""
    global _latex_ok
    if _latex_ok is None:
        if shutil.which("pdflatex") is None:
            cands = [os.path.expandvars(r"%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"),
                     r"C:\Program Files\MiKTeX\miktex\bin\x64",
                     r"C:\texlive\2024\bin\windows"]
            hit = next((c for c in cands
                        if os.path.exists(os.path.join(c, "pdflatex.exe"))), None)
            if hit:
                os.environ["PATH"] = hit + os.pathsep + os.environ.get("PATH", "")
        _latex_ok = shutil.which("pdflatex") is not None
    return _latex_ok


def _register_fonts():
    if not _FONT_PATH:
        raise FileNotFoundError("中文字体缺失 (simhei/msyh)，无法生成 PDF 报告")
    try:
        pdfmetrics.getFont(_FONT_NAME)
    except KeyError:
        # .ttc 需指定 subfontIndex=0
        pdfmetrics.registerFont(TTFont(_FONT_NAME, _FONT_PATH, subfontIndex=0))


def _setup_plot_fonts():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.usetex"] = False


# ═══════════════════════════════════════════════════════════
# 公式渲染 (usetex 真 LaTeX / mathtext 降级)
# ═══════════════════════════════════════════════════════════

def render_formula(latex_src: str, out_png: str, fontsize: int = 20) -> bool:
    """LaTeX 公式 → 透明底 PNG。返回是否用了真 LaTeX."""
    use_tex = latex_available()
    old_tex = plt.rcParams.get("text.usetex", False)
    old_pre = plt.rcParams.get("text.latex.preamble", "")

    def _save(tex_on: bool):
        plt.rcParams["text.usetex"] = tex_on
        if tex_on:
            plt.rcParams["text.latex.preamble"] = old_pre + \
                r" \usepackage{amsmath} \usepackage{amssymb}"
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, latex_src, fontsize=fontsize)
        fig.savefig(out_png, dpi=200, transparent=True,
                    bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    try:
        _save(use_tex)
    except Exception as e:
        print(f"  [report] usetex 失败, 降级 mathtext: {e}")
        use_tex = False
        _save(False)
    finally:
        plt.rcParams["text.usetex"] = old_tex
        plt.rcParams["text.latex.preamble"] = old_pre
    return use_tex


def _formula_image(story, png_path, width_mm=70):
    w_px, h_px = ImageReader(png_path).getSize()
    w = width_mm * mm
    h = w * h_px / max(w_px, 1)
    tbl = Table([[Image(png_path, width=w, height=h)]], colWidths=[170 * mm])
    tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(tbl)


# ═══════════════════════════════════════════════════════════
# 实验图表 (全部来自实测)
# ═══════════════════════════════════════════════════════════

def _minutes(ts, t0):
    return [(t - t0) / 60.0 for t in ts]


def _hhmmss_to_epoch(s, t0):
    """HH:MM:SS → epoch, 取与 t0 最接近的当日/前后日候选 (规避凌晨 hh<6 启发式偏移)"""
    try:
        hh, mm_, ss = [int(x) for x in str(s).split(":")]
        day = int(t0 // 86400) * 86400
        cands = [day + hh * 3600 + mm_ * 60 + ss + d * 86400
                 for d in (-1, 0, 1)]
        return min(cands, key=lambda e: abs(e - t0))
    except Exception:
        return None


def _fig_temp(temp_history, segments, t0, out_png):
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    if temp_history and len(temp_history) >= 2:
        ts, vs = [], []
        for p in temp_history:
            ep = _hhmmss_to_epoch(p.get("t"), t0) if isinstance(p.get("t"), str) else p.get("t")
            if ep is not None:
                ts.append(ep); vs.append(p["v"])
        if len(ts) >= 2:
            ax.plot(_minutes(ts, t0), vs, color="#c0392b", lw=1.4)
            ax.set_xlabel("时间 (min)"); ax.set_ylabel("温度 PV (°C)")
    if not ax.lines:
        # 温控离线: 用段表 T1/T2 按段线性插值 (视频模拟时温度实际在升)
        if segments:
            xs, ys, t_prev = [], [], 0.0
            for rec in segments:
                t2m = (rec.get("t2", t0) - t0) / 60.0
                xs += [t_prev, t2m]; ys += [rec["T1"], rec["T2"]]
                t_prev = t2m
            ax.plot(xs, ys, color="#c0392b", lw=1.4)
            ax.text(0.02, 0.92, "温控离线：曲线由各段标称 T1/T2 线性插值",
                    transform=ax.transAxes, fontsize=8, color="#888")
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("温度 (°C)")
    for rec in segments:
        if rec.get("t2"):
            xm = (rec["t2"] - t0) / 60.0
            ax.axvline(xm, color="#888", ls=":", lw=0.7)
            ax.annotate(f'段{rec["seg"]}: {rec["T2"]:.0f}°C',
                        (xm, ax.get_ylim()[1]), fontsize=7, color="#555",
                        xytext=(2, -2), textcoords="offset points", va="top")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _fig_n(n_history, segments, t0, out_png):
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    if n_history and len(n_history) >= 2:
        ts = [p[0] for p in n_history]
        ns = [abs(p[1]) for p in n_history]
        ax.plot(_minutes(ts, t0), ns, color="#1a5276", lw=1.4)
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("条纹计数 N (条)")
        for rec in segments:
            if rec.get("t2"):
                xm = (rec["t2"] - t0) / 60.0
                ax.axvline(xm, color="#888", ls=":", lw=0.7)
                ax.annotate(f'{rec["N"]:.1f}', (xm, rec["N"]), fontsize=7,
                            color="#555", xytext=(3, 0), textcoords="offset points")
    else:
        ax.text(0.5, 0.5, "无 N(t) 时序记录", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("条纹计数 N (条)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _fig_ndt(n_history, temp_history, segments, t0, out_png):
    """图4: 条纹数 N 与温度 T 的关系 (累计 N 散点 + 线性拟合 → α)

    多段扫描每段计数器清零, 逐段 N 是片段值而非累计值——直接用片段 N 做
    N-T 散点呈锯齿状, 跨段拟合穿越复位断崖, 斜率无物理意义。
    修正: 多段用累计 N (段1 N₁, 段1+2 N₁+N₂, …) 对 T2 线性拟合;
          单段/自由模式无段间清零, 用 n_history 时间对齐散点."""
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    pts = []          # (T, cumulative_N)
    label_note = ""

    if segments and len(segments) >= 1:
        cum = 0.0
        for rec in sorted(segments, key=lambda r: r.get("seg", 0)):
            n = rec.get("N", 0)
            if n > 0:
                cum += n
                pts.append((rec.get("T2", rec.get("T1", 0.0)), cum))
        if len(pts) < 2:
            # 单段/自由模式: 累计点不足, 降级用 n_history 时序散点
            pts = []
            if n_history and len(n_history) >= 2:
                if temp_history:
                    temps = []
                    for p in temp_history:
                        ep = (_hhmmss_to_epoch(p.get("t"), t0)
                              if isinstance(p.get("t"), str) else p.get("t"))
                        if ep is not None:
                            temps.append((ep, p["v"]))
                    if len(temps) >= 2:
                        ts_arr = np.array([q[0] for q in temps])
                        vs_arr = np.array([q[1] for q in temps])
                        for tn, nn in n_history:
                            if abs(nn) > 0.1:
                                tv = float(np.interp(tn, ts_arr, vs_arr))
                                pts.append((tv, abs(nn)))
                if not pts:
                    for i, (tn, nn) in enumerate(n_history):
                        if abs(nn) > 0.1 and i % 10 == 0:  # n_history 降采样
                            pts.append((i, abs(nn)))
                    label_note = "温控离线：帧序号代替温度"

    if pts and len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        ax.scatter(xs, ys, color="#2874a6", s=30, zorder=3)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            k, b = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, k * xx + b, color="#c0392b", ls="--", lw=1.2)
        alpha_fit = k * LAMBDA_NM * 1e-9 / (2 * L0_MM * 1e-3) * 1e6
        ax.text(0.03, 0.88,
                f"dN/dT = {k:.3f} 条/°C → α ≈ {alpha_fit:.1f}"
                f"$\\times10^{{-6}}$/K", transform=ax.transAxes, fontsize=9)
        if label_note:
            ax.text(0.03, 0.05, label_note, transform=ax.transAxes,
                    fontsize=8, color="#888")
    else:
        ax.text(0.5, 0.5, "无 N–T 关系数据", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
    ax.set_xlabel("温度 T (°C)"); ax.set_ylabel("累计条纹计数 N (条)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _fig_phi(error_stats, out_png):
    """φ(s) 外推图: 各通道相位 N_k vs 归一化 s, Theil-Sen 拟合外推 s=0"""
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    s = np.asarray(error_stats.get("s_norm") or [], float)
    nk = np.asarray(error_stats.get("N_per_channel") or [], float)
    if len(s) >= 2 and len(nk) == len(s):
        ax.scatter(s, nk, color="#7d3c98", s=45, zorder=3)
        # Theil-Sen 斜率 (所有点对斜率的中位数)
        iu = np.triu_indices(len(s), 1)
        if iu[0].size:
            slopes = (nk[iu[1]] - nk[iu[0]]) / (s[iu[1]] - s[iu[0]] + 1e-9)
            k = float(np.median(slopes))
            b = float(np.median(nk - k * s))
            xx = np.linspace(0, s.max(), 50)
            ax.plot(xx, k * xx + b, color="#c0392b", ls="--", lw=1.2)
            ax.axvline(0, color="#888", ls=":", lw=0.8)
            ax.scatter([0], [b], color="#c0392b", zorder=4, s=40)
            ax.text(0.02, 0.88, f"外推截距 N(s=0) ≈ {b:.2f}",
                    transform=ax.transAxes, fontsize=9)
        ax.set_xlabel(r"归一化 $s\ (=r^2/r_{\mathrm{max}}^2)$")
        ax.set_ylabel("相位累计 $N_k$ (条)")
    else:
        ax.text(0.5, 0.5, "无相位解调统计", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_xlabel("归一化 $s$"); ax.set_ylabel("相位累计 $N_k$ (条)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _fig_disp(disp_history, t0, out_png):
    """图7(原图8): 条纹稳定度/相位分散时序 — 诊断振动.
    稳定度 1=稳 / 0=振动 (含相位抖动分量); burst 区间红色阴影."""
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    if disp_history and len(disp_history) >= 2:
        xs = _minutes([p[0] for p in disp_history], t0)
        st = [float(p[1]) for p in disp_history]
        burst = [bool(p[2]) for p in disp_history]
        ax.plot(xs, st, color="#16a085", lw=1.2)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("条纹稳定度 (1=稳 / 0=振动)")
        ax.fill_between(xs, 0, 1.05, where=burst, color="#c0392b", alpha=0.12,
                        interpolate=True)
        n_burst = sum(burst)
        if n_burst:
            ax.text(0.03, 0.93, f"检测到 {n_burst} 帧振动 (红色阴影)",
                    transform=ax.transAxes, fontsize=9, color="#c0392b")
        else:
            ax.text(0.03, 0.93, "全程无振动告警",
                    transform=ax.transAxes, fontsize=9, color="#555")
    else:
        ax.text(0.5, 0.5, "无稳定度时序（未测量）", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_xlabel("时间 (min)"); ax.set_ylabel("条纹稳定度")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def make_figures(ctx, tmpdir) -> dict:
    """生成全部实验图表, 返回 {key: png_path}. 照片/帧直接引用原文件."""
    _setup_plot_fonts()
    t0 = ctx.get("t_scan_start")
    if not t0:
        if ctx.get("n_history"):
            t0 = ctx["n_history"][0][0]
        elif ctx.get("temp_history"):
            ft = ctx["temp_history"][0].get("t")
            # 修复: HH:MM:SS 字符串不可作时间轴原点 (仅 _hhmmss_to_epoch 可解析),
            # 直接取当前时间, 由 _hhmmss_to_epoch 就近日候选解析
            t0 = time.time() if isinstance(ft, str) else ft
        else:
            t0 = time.time()
    paths = {}
    # 图1/图2: 实验过程照片/帧 (原图直嵌, 不重绘)
    if ctx.get("coarse_photo") and os.path.exists(ctx["coarse_photo"]):
        paths["coarse"] = ctx["coarse_photo"]
    if ctx.get("fine_frame") and os.path.exists(ctx["fine_frame"]):
        paths["fine"] = ctx["fine_frame"]
    # 图3-7: matplotlib 重绘 (图6 α柱状已删除)
    p3 = str(Path(tmpdir) / "fig_temp.png")
    _fig_temp(ctx.get("temp_history") or [], ctx.get("segments") or [], t0, p3)
    paths["temp"] = p3
    p4 = str(Path(tmpdir) / "fig_n.png")
    _fig_n(ctx.get("n_history") or [], ctx.get("segments") or [], t0, p4)
    paths["n"] = p4
    p5 = str(Path(tmpdir) / "fig_ndt.png")
    _fig_ndt(ctx.get("n_history") or [], ctx.get("temp_history") or [],
             ctx.get("segments") or [], t0, p5)
    paths["ndt"] = p5
    p6 = str(Path(tmpdir) / "fig_phi.png")
    _fig_phi(ctx.get("error_stats") or {}, p6); paths["phi"] = p6
    p7 = str(Path(tmpdir) / "fig_disp.png")
    _fig_disp(ctx.get("disp_history") or [], t0, p7); paths["disp"] = p7
    return paths


# ═══════════════════════════════════════════════════════════
# 样式与表格
# ═══════════════════════════════════════════════════════════

def _build_styles():
    base = getSampleStyleSheet()
    st = {}
    st["title"] = ParagraphStyle("RTitle", parent=base["Title"], fontName=_FONT_NAME,
                                 fontSize=18, alignment=TA_CENTER, spaceAfter=20,
                                 leading=26)
    st["h1"] = ParagraphStyle("RH1", parent=base["Heading1"], fontName=_FONT_NAME,
                              fontSize=14, spaceBefore=10, spaceAfter=10, leading=20)
    # wordWrap='CJK': 中文字符逐字换行, 避免两端对齐时中英混排拉伸出大段空格
    st["body"] = ParagraphStyle("RBody", parent=base["BodyText"], fontName=_FONT_NAME,
                                fontSize=11, leading=16.5, alignment=TA_JUSTIFY,
                                spaceAfter=6, wordWrap="CJK")
    st["meta"] = ParagraphStyle("RMeta", parent=base["Normal"], fontName=_FONT_NAME,
                                fontSize=10, alignment=TA_CENTER, textColor=colors.grey,
                                spaceAfter=12, wordWrap="CJK")
    st["caption"] = ParagraphStyle("RCap", parent=base["Normal"], fontName=_FONT_NAME,
                                   fontSize=10, alignment=TA_CENTER, spaceBefore=2,
                                   spaceAfter=10, wordWrap="CJK")
    st["cell"] = ParagraphStyle("RCell", parent=base["Normal"], fontName=_FONT_NAME,
                                fontSize=10, alignment=TA_CENTER, leading=13,
                                wordWrap="CJK")
    return st


def _safe(text: str) -> str:
    """SimHei 缺上下标/特殊字形 → reportlab <super>/<sub> 标记或 ASCII 防乱码.
    覆盖: ² ³ ⁻ ⁺ (上标), −(U+2212 负号, SimHei 缺→ASCII '-'), 下标 ₀₁₂."""
    return (str(text)
            .replace("⁻⁶", "<super>-6</super>")
            .replace("²", "<super>2</super>")
            .replace("³", "<super>3</super>")
            .replace("⁻", "<super>-</super>")
            .replace("⁺", "<super>+</super>")
            .replace("−", "-")
            .replace("₀", "<sub>0</sub>")
            .replace("₁", "<sub>1</sub>")
            .replace("₂", "<sub>2</sub>"))


def _table(headers, rows, col_widths, st):
    data = [[Paragraph(_safe(h), st["cell"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(_safe(c), st["cell"]) for c in row])
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def _fig_image(png_path, width_mm=160):
    w_px, h_px = ImageReader(png_path).getSize()
    w = width_mm * mm
    h = w * h_px / max(w_px, 1)
    return Image(png_path, width=w, height=h)


# ═══════════════════════════════════════════════════════════
# 主入口: 组装内容流并 build
# ═══════════════════════════════════════════════════════════

def build_report_pdf(ctx: dict, out_path) -> dict:
    """组装实验报告并生成 PDF.

    ctx 键: title, generated_at, segments, alpha_avg, alpha_std, error_pct,
            params, steps, vlm_history, conclusion,
            n_history, temp_history, t_scan_start,
            coarse_photo, fine_frame, error_stats, latex_degraded
    返回 {"latex_used": bool}.
    """
    _register_fonts()
    st = _build_styles()
    from ..config import ROOT_DIR
    tmp_base = Path(ROOT_DIR) / ".report_tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="report_", dir=str(tmp_base))
    try:
        # ── 公式图 ──
        latex_used = render_formula(
            r"$\alpha = \dfrac{N \lambda}{2 L_0 \Delta T}$",
            f"{tmpdir}/eq_main.png")
        render_formula(r"$2h\cos\theta = k\lambda$", f"{tmpdir}/eq_bright.png")
        render_formula(r"$\Delta L = N \cdot \dfrac{\lambda}{2}$",
                       f"{tmpdir}/eq_dl.png")

        figs = make_figures(ctx, tmpdir)

        story = []
        # 1. 标题 + 时间戳
        story.append(Paragraph(ctx.get("title", "热膨胀系数测量实验报告"), st["title"]))
        story.append(Paragraph(
            f"生成时间：{ctx.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
            st["meta"]))

        # 2. 一、实验目的
        story.append(Paragraph("一、实验目的", st["h1"]))
        story.append(Paragraph(
            "了解迈克尔逊干涉仪的基本原理，并采用干涉法测量黄铜（H62）样品的线膨胀系数 α。"
            "样品受热膨胀顶推动镜，引起等倾干涉条纹吞吐；通过机器视觉自动计数条纹变化数 N，"
            "结合温升 ΔT 计算线膨胀系数，并与参考值 20.8×10<super>-6</super>/K 对比，"
            "验证条纹吞吐与热膨胀的定量对应关系。", st["body"]))

        # 3. 二、实验原理
        story.append(Paragraph("二、实验原理", st["h1"]))
        story.append(Paragraph(
            "调节迈克尔逊干涉仪使动镜 M1 与定镜虚像 M2' 平行，形成等效空气薄膜，"
            "产生等倾干涉同心圆环。亮纹满足光程差条件：", st["body"]))
        _formula_image(story, f"{tmpdir}/eq_bright.png", 55)
        story.append(Paragraph(
            "样品膨胀推动 M1 使膜厚 h 改变，h 每变化 λ/2，固定半径处吞吐一条条纹。"
            "计数条纹变化数 N，结合温升 ΔT 与样品初始长度 L<sub>0</sub>，得线膨胀系数：",
            st["body"]))
        _formula_image(story, f"{tmpdir}/eq_main.png", 75)
        story.append(Paragraph(_safe(
            f"本实验 λ = {LAMBDA_NM} nm（He-Ne 激光），每吞吐一条对应样品伸长 "
            f"λ/2 = {HALF_LAMBDA_NM} nm。条纹计数采用带符号空间相位解调：对干涉图像做 "
            "s=r² 椭圆极坐标展开，FFT 窄带提取相位，多通道 Theil-Sen 拟合外推至中心，"
            "方向敏感（吞 +1 / 吐 −1），升温首尾的吞吐往复自动抵消，且具有亚条纹分辨率。"),
            st["body"]))

        # 4. 三、实验仪器
        story.append(Paragraph("三、实验仪器", st["h1"]))
        inst_rows = [
            ["激光器", "He-Ne", "相干光源，波长 632.8 nm"],
            ["干涉仪", "迈克尔逊", "动镜由样品热膨胀顶推"],
            ["工业相机", "海康 MV-CS050-60GC", "GigE 采集干涉条纹 (2448×2048)"],
            ["温控器", "ANTHONE LU-926U", "RS485 Modbus，PID 加热 + PT100 测温"],
            ["样品", "黄铜 H62", f"L₀ = {L0_MM:.0f} mm"],
            ["智能分析", "CNN + 轻量VLM Agent", "光心/ROI/质量感知 + 实验指导"],
        ]
        story.append(_table(["名称", "型号", "用途"], inst_rows,
                            [35 * mm, 60 * mm, 75 * mm], st))
        story.append(Spacer(1, 6))

        # 5. 四、实验步骤
        story.append(Paragraph("四、实验步骤", st["h1"]))
        steps = ctx.get("steps") or []
        for i, (name, tm) in enumerate(steps, 1):
            t_str = f"（{tm}）" if tm else ""
            story.append(Paragraph(f"{i}. {name}{t_str}", st["body"]))
        if not steps:
            story.append(Paragraph("（本次报告无分阶段指导记录）", st["body"]))

        # 6. 五、数据处理与结果
        story.append(Paragraph("五、数据处理与结果", st["h1"]))
        segs = ctx.get("segments") or []

        story.append(Paragraph("5.1 实验参数", st["body"]))
        params = ctx.get("params") or []
        if params:
            prow = [[params[i][0], params[i][1],
                     *(params[i + 1] if i + 1 < len(params) else ["", ""])]
                    for i in range(0, len(params), 2)]
            story.append(_table(["参数", "值", "参数", "值"], prow,
                                [38 * mm, 47 * mm, 38 * mm, 47 * mm], st))
            story.append(Spacer(1, 6))

        story.append(Paragraph("5.2 分析结果", st["body"]))
        res_rows = [[rec["seg"], f'{rec["T1"]:.2f}', f'{rec["T2"]:.2f}',
                     f'{rec["N"]:.2f}', f'{rec["dT"]:.2f}',
                     f'{rec["alpha"]:.2f}'] for rec in segs]
        a_avg = ctx.get("alpha_avg", 0.0); a_std = ctx.get("alpha_std", 0.0)
        err = ctx.get("error_pct", 0.0)
        # 修复: 汇总信息合并进 α 列 (原实现把 α均值 放 ΔT 列、误差 放 α 列, 列语义错位)
        res_rows.append(["—", "—", "—", "—", "—",
                         f"{a_avg:.2f} ± {a_std:.2f} (误差 {err:.2f}%)"])
        story.append(_table(["段", "T1 (°C)", "T2 (°C)", "N (条)",
                             "ΔT (°C)", "α (×10⁻⁶/K)"],
                            res_rows,
                            [16 * mm, 28 * mm, 28 * mm, 28 * mm,
                             35 * mm, 35 * mm], st))
        story.append(Spacer(1, 6))

        story.append(Paragraph("5.3 计算过程", st["body"]))
        _formula_image(story, f"{tmpdir}/eq_dl.png", 55)
        calc_rows = []
        for rec in segs:
            dl_nm = rec["N"] * HALF_LAMBDA_NM
            dT = rec["dT"] if rec["dT"] >= 0.05 else float("nan")
            alpha_txt = (f"{dl_nm / (L0_MM * dT):.2f}"
                         if dT == dT else "ΔT 过小, 记无效")
            calc_rows.append([rec["seg"], f'{rec["N"]:.2f}',
                              f"{dl_nm:.0f} nm = {dl_nm / 1e3:.3f} μm",
                              f'{rec["dT"]:.2f}', alpha_txt])
        story.append(_table(["段", "N (条)", "ΔL = N×316.4 nm", "ΔT (°C)",
                             "α = ΔL/(L₀·ΔT)"],
                            calc_rows,
                            [14 * mm, 24 * mm, 56 * mm, 28 * mm, 48 * mm], st))
        story.append(Spacer(1, 6))

        # 6.4 实验过程图表 (全部来自实测; 图6 α柱状已删除)
        story.append(Paragraph("5.4 实验过程图表", st["body"]))
        # 图1/图2 并排同页: 粗调光点照片 + 细调干涉条纹帧 (各 85mm, 缺失时占位)
        coarse_img = (_fig_image(figs["coarse"], 85) if "coarse" in figs else None)
        fine_img = (_fig_image(figs["fine"], 85) if "fine" in figs else None)
        cell_c = coarse_img or Paragraph("（未上传粗调光点照片）", st["cell"])
        cell_f = fine_img or Paragraph("（未存档细调条纹帧）", st["cell"])
        pair = Table([[cell_c, cell_f]], colWidths=[85 * mm, 85 * mm])
        pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(KeepTogether([
            pair,
            Paragraph("图 1 光路粗调光点（左）与光路细调干涉条纹（右）", st["caption"]),
        ]))
        story.append(_fig_image(figs["temp"]))
        story.append(Paragraph("图 2 温度变化曲线 T(t)（实测 PV，竖线为各段结算时刻）",
                               st["caption"]))
        story.append(_fig_image(figs["n"]))
        story.append(Paragraph("图 3 条纹计数曲线 N(t)（每段计数独立累计，段末结算）",
                               st["caption"]))
        story.append(_fig_image(figs["ndt"]))
        story.append(Paragraph("图 4 条纹数 N 与温度 T 的关系（线性拟合斜率 → α）",
                               st["caption"]))
        if "phi" in figs and os.path.exists(figs["phi"]):
            story.append(_fig_image(figs["phi"]))
            story.append(Paragraph("图 5 空间相位 φ(s) 外推（各采样环通道 + Theil-Sen 拟合）",
                                   st["caption"]))
        if "disp" in figs and os.path.exists(figs["disp"]):
            story.append(_fig_image(figs["disp"]))
            story.append(Paragraph("图 6 条纹稳定度/相位分散时序（诊断振动，红色=振动区间）",
                                   st["caption"]))

        # 7. VLM 分析摘要表
        story.append(Paragraph("附：VLM 视觉分析摘要", st["h1"]))
        vlm = ctx.get("vlm_history") or []
        if vlm:
            vrows = [[v.get("time", ""), v.get("scene", ""),
                      f'{v.get("quality_cnn", 0):.2f}',
                      (v.get("analysis") or "")[:80]] for v in vlm]
            story.append(_table(["时间", "场景", "CNN质量", "分析摘要"],
                                vrows, [22 * mm, 30 * mm, 20 * mm, 98 * mm],
                                st))
        else:
            story.append(Paragraph("本次实验无 VLM 深度分析记录。", st["body"]))
        story.append(Spacer(1, 6))

        # 8. 六、实验结论
        story.append(Paragraph("六、实验结论", st["h1"]))
        story.append(Paragraph(_safe(ctx.get("conclusion") or "（结论生成失败）"),
                               st["body"]))
        if ctx.get("latex_degraded"):
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                "注：本机未检测到 LaTeX 发行版，公式由 mathtext 降级渲染。",
                st["meta"]))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                                leftMargin=20 * mm, rightMargin=20 * mm,
                                topMargin=20 * mm, bottomMargin=20 * mm,
                                title=ctx.get("title", "实验报告"))
        doc.build(story)
        return {"latex_used": latex_used}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
