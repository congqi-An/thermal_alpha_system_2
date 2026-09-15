"""基于计算机视觉和轻量VLM的干涉法测量热膨胀智能实验系统 — 配置"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── 激光 / 样品 ──
LASER_WAVELENGTH_NM = 632.8
LASER_WAVELENGTH_M = 632.8e-9
SAMPLE_LENGTH_MM = 150.0          # 黄铜样品长度 mm
SAMPLE_LENGTH_M = 0.15
REFERENCE_ALPHA = 20.8           # 黄铜 H62 参考值 x10^-6/K

# ── 相机 (海康 MV-CS050-60GC) ──
CAMERA_EXPOSURE_US = 20000        # 提高曝光(原5000太暗, mean~0.02)
CAMERA_GAIN_DB = 0.0
CAMERA_FPS = 20.0
CAMERA_GAMMA = 0.7                # Gamma(暗部非线性增亮, 0.7常用)
CAMERA_BLACKLEVEL = 0             # 黑电平

# ── Modbus 温控 (ANTHONE LU-926U) ──
MODBUS_PORT = "COM7"
MODBUS_BAUD = 9600
MODBUS_SLAVE = 1

# ── 自动温度扫描 ──
SCAN_T_START = 30.0
SCAN_T_END = 50.0
SCAN_STEP = 5.0
SCAN_STABILIZE_S = 3.0           # 段末到温确认: PV>=T2 保持此时长即结算 (短, 防保温吞吐虚计)
SCAN_ENTER_CONFIRM_S = 3.0       # 首段进段确认: PV>=T_start 保持此时长才开计数
SCAN_STABLE_BAND = 0.3           # |PV-SV| < 此值视为趋稳 (PID整定后实测稳态±0.2, 由0.5收紧)

# ── 条纹计数回退半径 ──
COUNT_R0S = (150, 200, 250)      # 固定径向位置

# ── Web ──
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
VIDEO_JPEG_QUALITY = 70
VIDEO_MAX_FPS = 15.0

MEASURE_SAVE_DIR = ROOT_DIR / "measurements"

# ── 条纹质量评估 (CNN) ──
QUALITY_MODEL_PATH = ROOT_DIR / "checkpoints" / "best_quality.pt"
QUALITY_THRESHOLD = 0.4          # 低于此分的帧剔除(不参与过零计数)

# ── 帧分析三头 CNN (FrameAnalysisNet: 光心+ROI+质量 一次推理) ──
# 2026-08-01: 质量头已用真实标签+合成退化重训 (q_retrained, 骨干 bit-identical),
# 线上实时质量分由此生效 (center 门控/Agent 安全检测/前端读数)。基座备份: frame_analysis_net.pt
FRAME_ANALYSIS_NET_PATH = ROOT_DIR / "checkpoints" / "frame_analysis_net.q_retrained.pt"

# ── 光心+ROI 双头 CNN (CenterR0Net: FrameAnalysisNet 不可用时回退) ──
CENTER_R0_NET_PATH = ROOT_DIR / "checkpoints" / "center_r0_net.pt"
CENTER_R0_METHOD = "cnn"        # "cnn" 用 CNN 自判定; "fixed" 用固定 COUNT_R0S

# ── 光心定位 (CenterNet, 裁剪推理, 最终回退) ──
CENTER_METHOD = "cnn"             # "cnn" 用 CNN 光心(默认); "cv" 回退 find_radial_center
CENTER_CNN_PATH = ROOT_DIR / "checkpoints" / "center_cnn.pt"

# ── 多模态 VLM (经 Ollama 托管, 不占本地加载开销) ──
VLM_ENABLED = True               # True 启用 VLM (Ollama 提供, 不占本地显存)
# ── Qwen3-VL 4B Agent (Ollama, Windows 原生) ──
LLM_SERVER_URL = "http://localhost:11434"   # Ollama 地址
LLM_MODEL_NAME = "qwen3-vl:4b"              # Ollama 模型标识 (2026-08-03 由 openbmb/minicpm-v4.6 切换)
AGENT_TEMP_LIMIT = 60.0                    # 安全温度上限 °C (2026-08-07 由 80 收紧)
