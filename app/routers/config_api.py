"""系统配置 REST API — 集中展示所有可设参数"""
from fastapi import APIRouter

from .. import config, state

router = APIRouter(prefix="/api/config", tags=["config"])

# 海康相机常用可调参数(通过 /api/camera/set 通用设置)
CAMERA_PARAMS = [
    {"name": "ExposureTime", "type": "float", "unit": "us", "desc": "曝光时间"},
    {"name": "Gain", "type": "float", "unit": "dB", "desc": "增益"},
    {"name": "Gamma", "type": "float", "desc": "Gamma(暗部增亮)"},
    {"name": "BlackLevel", "type": "float", "desc": "黑电平"},
    {"name": "BalanceWhiteAuto", "type": "enum", "desc": "白平衡(0关/1次/2连续)"},
    {"name": "BalanceRatioRed", "type": "float", "desc": "红白平衡比"},
    {"name": "BalanceRatioGreen", "type": "float", "desc": "绿白平衡比"},
    {"name": "BalanceRatioBlue", "type": "float", "desc": "蓝白平衡比"},
    {"name": "Contrast", "type": "float", "desc": "对比度"},
    {"name": "Sharpness", "type": "float", "desc": "锐度"},
    {"name": "Saturation", "type": "float", "desc": "饱和度"},
    {"name": "AcquisitionFrameRate", "type": "float", "unit": "fps", "desc": "帧率"},
    {"name": "Width", "type": "int", "desc": "ROI 宽"},
    {"name": "Height", "type": "int", "desc": "ROI 高"},
    {"name": "OffsetX", "type": "int", "desc": "ROI 偏移X"},
    {"name": "OffsetY", "type": "int", "desc": "ROI 偏移Y"},
    {"name": "BinningHorizontal", "type": "int", "desc": "水平 Binning"},
    {"name": "BinningVertical", "type": "int", "desc": "垂直 Binning"},
    {"name": "ReverseX", "type": "enum", "desc": "水平翻转"},
    {"name": "ReverseY", "type": "enum", "desc": "垂直翻转"},
    {"name": "PixelFormat", "type": "enum", "desc": "像素格式"},
    {"name": "TriggerMode", "type": "enum", "desc": "触发模式(0连续/1触发)"},
]

# 温控 LU-926U 常用可调寄存器(通过 /api/temp/write 通用设置)
MODBUS_REGS = [
    {"name": "SV1", "desc": "设定温度°C", "unit": "°C"},
    {"name": "P1", "desc": "P 比例带"},
    {"name": "P2", "desc": "I 积分时间"},
    {"name": "rt", "desc": "D 微分时间"},
    {"name": "cHy", "desc": "控制死区"},
    {"name": "ctL", "desc": "控制周期"},
    {"name": "crL", "desc": "控制模式(1ONOFF/2自整定/3手动/4PID)"},
    {"name": "Act", "desc": "动作方向(0制冷/1加热)"},
    {"name": "Sn1", "desc": "传感器类型"},
    {"name": "Poi1", "desc": "分辨率(0:1°C/1:0.1°C)"},
    {"name": "SMV", "desc": "手动输出%"},
]


@router.get("")
async def get_config():
    """返回所有配置参数 + 可调参数列表"""
    return {
        "camera": {
            "exposure_us": config.CAMERA_EXPOSURE_US,
            "gain_db": config.CAMERA_GAIN_DB,
            "fps": config.CAMERA_FPS,
            "gamma": config.CAMERA_GAMMA,
            "blacklevel": config.CAMERA_BLACKLEVEL,
            "model": "MV-CS050-60GC",
        },
        "modbus": {
            "port": config.MODBUS_PORT,
            "baud": config.MODBUS_BAUD,
            "slave": config.MODBUS_SLAVE,
            "device": "LU-926U",
        },
        "scan": {
            "T_start": config.SCAN_T_START,
            "T_end": config.SCAN_T_END,
            "step": config.SCAN_STEP,
            "stabilize_s": config.SCAN_STABILIZE_S,
            "stable_band": config.SCAN_STABLE_BAND,
        },
        "algo": {
            "L0_mm": config.SAMPLE_LENGTH_MM,
            "lam_nm": config.LASER_WAVELENGTH_NM,
            "ref_alpha": config.REFERENCE_ALPHA,
            "count_r0s": list(config.COUNT_R0S),
            "quality_threshold": config.QUALITY_THRESHOLD,
        },
        "vlm": {
            "model": config.LLM_MODEL_NAME,
            "max_new": 400,
            "downsample": "16x",
        },
        "center": {
            "method": state.center_method,           # 当前光心定位法 cnn/cv
            "cnn_loaded": state.center_cnn is not None,
            "available": ["cnn", "cv"] if state.center_cnn is not None else ["cv"],
        },
        "quality_gate": {
            "enabled": state.quality_gate_enabled,   # 质量门控开关
            "threshold": config.QUALITY_THRESHOLD,
        },
        "camera_params": CAMERA_PARAMS,
        "modbus_regs": MODBUS_REGS,
    }
