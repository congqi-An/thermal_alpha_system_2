"""全局共享状态"""
import threading
from typing import Optional, Any

# 硬件/算法实例
camera: Optional[Any] = None
temp_ctrl: Optional[Any] = None
fringe_counter: Optional[Any] = None
scan_runner: Optional[Any] = None
quality_net: Optional[Any] = None
current_quality: float = 0.0  # 默认 0 (相机离线时不应误导 Agent); 测量期被稳定度覆盖
quality_gate_enabled: bool = True  # 质量门控开关: True=CNN质量头参与计数决策(中心门按q排除);
                                   # False=不参与(所有帧光心都进EMA, 回归无质量头的原行为)
# 时序稳定度/振动 burst (PhaseFringeCounter 算, 测量期更新): 替代恒1的CNN q 作"质量"
stability: float = 1.0
burst: bool = False
# N 振动控制门 (NRateGate, 惯性覆盖): 门后 N 供显示/结算, 防止大震动时 N 跳变 >2 条/s
n_gate: Optional[Any] = None
gated_N: float = 0.0          # 门后带方向 N
gated_N_abs: float = 0.0      # |门后 N| (替代 total_N[0] 供 α 结算)
vlm: Optional[Any] = None

# ── 反应层告警投递 (SafetyGuard 检测到 Q 跳变后由 LLM 生成文案, 经 /ws/measure 推送) ──
alert_seq: int = 0                        # 单调递增, 前端用去重
last_alert: Optional[dict] = None         # {seq, text, level, ts}

# 帧分析三头 CNN (FrameAnalysisNet: 光心+ROI+质量 一次推理, 优先使用)
frame_analysis_net: Optional[Any] = None
frame_analysis_device: str = "cpu"

# 光心+ROI 双头 CNN (CenterR0Net: FrameAnalysisNet 不可用时回退)
center_r0_net: Optional[Any] = None
center_r0_device: str = "cpu"
center_r0_method: str = "cnn"   # "cnn" 自判定 / "fixed" 固定 COUNT_R0S

# 光心定位 CNN (CenterNet, 最终回退)
center_cnn: Optional[Any] = None
center_cnn_device: str = "cpu"
center_method: str = "cnn"        # "cnn" / "cv" (运行时可切)


# Qwen3-VL 4B Agent (Ollama): 9 阶段指导 + 对话工具 + 反应层
agent: Optional[Any] = None

# 测量/扫描状态
measure_active: bool = False
measure_lock = threading.Lock()
scan_active: bool = False
video_auto_measure: bool = False   # 视频模拟源自动进入的测量 (卸载时退出)

# 温控实时数据
current_T: float = 25.0
current_sv: float = 0.0
current_mv: float = 0.0
temp_online: bool = False
temp_history: list = []          # [{t, v}] PV 历史

# 最新帧 (grab 线程写, ws 读)
latest_frame: Optional[Any] = None
_frame_lock = threading.Lock()

# 光心追踪运行时状态 (grab 线程写)
current_center: Optional[tuple] = None
center_history: list = []       # 滑动平均窗口 [(cx, cy)...]
current_r0s: Optional[tuple] = None   # 自适应 r0 (显示用, 计数用头帧锁定)
_r0_cnt: int = 0                # auto_r0_detect 触发计数器
# 光心/r0 实际来源 (CNN-CV 交叉校验后的真实使用方)
center_source: str = "none"     # "cnn" / "cv" / "none"
r0_source: str = "none"         # "cnn" / "cv-std" / "none"
# 光心/r0 判定模式 (前端按钮切换): cnn=纯 CNN / cv=纯 CV
center_mode: str = "cnn"

# 视频模拟源 (VideoFileSource): 加载后 grab 循环优先从视频取帧, 模拟实拍
video_source: Optional[Any] = None

# 报告图表数据 (scan_runner.start 初始化)
n_history: list = []              # 整轮扫描 N/T 时序 [(time, N_abs), ...] 段内计数 reset 不清空
disp_history: list = []           # 条纹稳定度/振动时序 [(time, stability, burst), ...] 诊断振动
_last_disp_t: float = 0.0         # disp_history 降采样时间戳
