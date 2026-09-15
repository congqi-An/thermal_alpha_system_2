"""海康 MVS SDK 相机封装层 — 基于官方 MvImport (SDK 4.8+)

依赖：MVS SDK 4.8 Development 组件
    MvImport 目录已从 MVS/Development/Samples/Python/ 复制到 app/camera/

用法：
    with HikCamera() as cam:
        cam.start()
        frame = cam.grab()  # numpy RGB 图像 (2448×2048 for MV-CS050-60GC)
        cam.stop()
"""

import sys
from pathlib import Path
from typing import Optional, Any
import numpy as np
import threading

# 确保 MvImport 在路径中
_MVIMPORT_DIR = Path(__file__).resolve().parent / "MvImport"
if str(_MVIMPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_MVIMPORT_DIR.parent))

from MvImport.MvCameraControl_class import (
    MvCamera,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_DEVICE_INFO,
    MV_FRAME_OUT_INFO_EX,
    MVCC_INTVALUE,
    MVCC_FLOATVALUE,
    MV_GIGE_DEVICE,
    MV_OK,
    MV_ACCESS_Exclusive,
    MV_EXPOSURE_AUTO_MODE_OFF,
    MV_GAIN_MODE_OFF,
    MV_TRIGGER_MODE_OFF,
    create_string_buffer,
    cast,
    POINTER,
)


class MvsError(Exception):
    """MVS SDK 错误"""
    pass


class MvsNotInstalled(Exception):
    """MVS SDK 未安装"""
    pass


class HikCamera:
    """海康 GigE 相机 (使用官方 MvImport SDK)"""

    def __init__(self):
        self._cam = MvCamera()
        self._opened = False
        self._grabbing = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._width: int = 0
        self._height: int = 0
        self._grab_buf: Optional[Any] = None      # 复用 grab 缓冲 (每帧新建 15MB 有 GC 压力)
        self._grab_buf_size: int = 0

    # ── 上下文管理 ──
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        if self._grabbing:
            self.stop()
        self.close()

    # ── 设备连接 ──
    def open(self) -> bool:
        """枚举并打开相机"""
        # 1. 枚举 GigE 设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)
        if ret != MV_OK:
            raise MvsError(f"设备枚举失败: 0x{ret:08X}")
        if deviceList.nDeviceNum == 0:
            raise MvsError(
                "未发现 GigE 相机。请检查:\n"
                "  1. 相机上电且网口指示灯闪烁\n"
                "  2. 电脑与相机在同一网段\n"
                "  3. 防火墙允许 UDP 3956, 49152-65535"
            )
        print(f"[HikCamera] 发现 {deviceList.nDeviceNum} 台 GigE 设备")

        # 2. 选择第一台设备创建句柄
        stDeviceInfo = cast(
            deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        ret = self._cam.MV_CC_CreateHandle(stDeviceInfo)
        if ret != MV_OK:
            raise MvsError(f"创建句柄失败: 0x{ret:08X}")

        # 3. 打开设备
        ret = self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            raise MvsError(f"打开设备失败: 0x{ret:08X}")

        # 4. 获取分辨率
        w_val, h_val = MVCC_INTVALUE(), MVCC_INTVALUE()
        self._cam.MV_CC_GetIntValue('Width', w_val)
        self._cam.MV_CC_GetIntValue('Height', h_val)
        self._width = w_val.nCurValue
        self._height = h_val.nCurValue
        self._opened = True
        print(f"[HikCamera] 已连接: {self._width}×{self._height}")
        return True

    def close(self):
        """关闭设备并释放资源"""
        self._opened = False
        try:
            if self._cam is not None:
                self._cam.MV_CC_CloseDevice()
                self._cam.MV_CC_DestroyHandle()
        except Exception:
            pass

    # ── 配置 ──
    def configure(self, exposure_us: int = 5000, gain_db: float = 0.0, fps: float = 20.0):
        """一键配置相机参数"""
        # 关闭自动曝光和自动增益
        try:
            self._cam.MV_CC_SetEnumValue('ExposureAuto', MV_EXPOSURE_AUTO_MODE_OFF)
        except Exception:
            pass
        try:
            self._cam.MV_CC_SetEnumValue('GainAuto', MV_GAIN_MODE_OFF)
        except Exception:
            pass
        # 连续采集模式
        try:
            self._cam.MV_CC_SetEnumValue('TriggerMode', MV_TRIGGER_MODE_OFF)
        except Exception:
            pass

        self.set_exposure(exposure_us)
        self.set_gain(gain_db)
        self.set_framerate(fps)

    def set_exposure(self, exposure_us: int):
        """设置曝光时间 (微秒) — 先关自动曝光, 再设"""
        try:
            self._cam.MV_CC_SetEnumValue('ExposureAuto', MV_EXPOSURE_AUTO_MODE_OFF)
            ret = self._cam.MV_CC_SetFloatValue('ExposureTime', float(exposure_us))
            print(f"[HikCamera] ExposureTime set={exposure_us} ret={ret}")
            return ret == MV_OK
        except Exception as e:
            print(f"[HikCamera] set_exposure err: {e}")
            return False

    def set_gain(self, gain_db: float) -> bool:
        """设置增益 (dB)"""
        try:
            return self._cam.MV_CC_SetFloatValue('Gain', float(gain_db)) == MV_OK
        except Exception as e:
            print(f"[HikCamera] set_gain: {e}")
            return False

    def set_framerate(self, fps: float):
        """设置帧率"""
        try:
            self._cam.MV_CC_SetFloatValue('AcquisitionFrameRate', float(fps))
        except Exception:
            pass  # 部分相机不支持直接设置帧率

    def set_gamma(self, gamma: float):
        """设置 Gamma (暗部增亮, 0.7~1.0)"""
        try:
            self._cam.MV_CC_SetFloatValue('Gamma', float(gamma))
        except Exception as e:
            print(f"[HikCamera] set_gamma: {e}")

    def set_blacklevel(self, level: float):
        """设置黑电平"""
        try:
            self._cam.MV_CC_SetFloatValue('BlackLevel', float(level))
        except Exception as e:
            print(f"[HikCamera] set_blacklevel: {e}")

    def set_auto_exposure(self, on: bool):
        """自动曝光开关"""
        try:
            from MvImport.MvCameraControl_class import MV_EXPOSURE_AUTO_MODE_CONTINUOUS, MV_EXPOSURE_AUTO_MODE_OFF
            val = MV_EXPOSURE_AUTO_MODE_CONTINUOUS if on else MV_EXPOSURE_AUTO_MODE_OFF
            self._cam.MV_CC_SetEnumValue('ExposureAuto', val)
        except Exception as e:
            print(f"[HikCamera] set_auto_exposure: {e}")

    # ── 通用参数设置(任意海康参数) ──
    def set_float(self, name: str, value: float) -> bool:
        try:
            return self._cam.MV_CC_SetFloatValue(name, float(value)) == MV_OK
        except Exception as e:
            print(f"[HikCamera] set_float {name}: {e}"); return False

    def set_enum(self, name: str, value) -> bool:
        try:
            # SDK 的 MV_CC_SetEnumValue 期望 int 枚举值, 传入 float 可能出错
            return self._cam.MV_CC_SetEnumValue(name, int(value)) == MV_OK
        except Exception as e:
            print(f"[HikCamera] set_enum {name}: {e}"); return False

    def set_int(self, name: str, value: int) -> bool:
        try:
            return self._cam.MV_CC_SetIntValue(name, int(value)) == MV_OK
        except Exception as e:
            print(f"[HikCamera] set_int {name}: {e}"); return False

    def get_float(self, name: str):
        try:
            v = MVCC_FLOATVALUE(); self._cam.MV_CC_GetFloatValue(name, v); return v.nCurValue
        except Exception:
            return None

    def get_int(self, name: str):
        try:
            v = MVCC_INTVALUE(); self._cam.MV_CC_GetIntValue(name, v); return v.nCurValue
        except Exception:
            return None

    # ── 取流控制 ──
    def start(self):
        """开始取流"""
        if self._grabbing:
            return
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            raise MvsError(f"开始取流失败: 0x{ret:08X}")
        self._grabbing = True
        print("[HikCamera] 取流已开始")

    def stop(self):
        """停止取流"""
        if not self._grabbing:
            return
        self._cam.MV_CC_StopGrabbing()
        self._grabbing = False

    def grab(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        """获取一帧 RGB 图像（阻塞）

        Returns:
            numpy RGB 图像 (H×W×3, uint8)，超时返回 None
        """
        if not self._grabbing:
            raise MvsError("相机未开始取流，请先调用 start()")

        frame_info = MV_FRAME_OUT_INFO_EX()
        buf_size = self._width * self._height * 3 + 2048
        if self._grab_buf is None or self._grab_buf_size != buf_size:
            self._grab_buf = create_string_buffer(buf_size)
            self._grab_buf_size = buf_size
        buf = self._grab_buf

        ret = self._cam.MV_CC_GetImageForRGB(buf, buf_size, frame_info, timeout_ms)
        if ret != MV_OK:
            return None

        data_len = frame_info.nFrameLen
        if data_len == 0 or data_len > buf_size:
            return None

        raw = np.frombuffer(buf, dtype=np.uint8, count=data_len)
        expected = self._width * self._height * 3
        if data_len >= expected:
            rgb = raw[:expected].reshape((self._height, self._width, 3)).copy()
        elif data_len == self._width * self._height:
            gray = raw.reshape((self._height, self._width))
            rgb = np.dstack([gray, gray, gray])
        else:
            return None

        with self._lock:
            self._latest_frame = rgb
        return rgb

    # ── 属性 ──
    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def is_grabbing(self) -> bool:
        return self._grabbing

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None


# ── 测试入口 ──
if __name__ == "__main__":
    import cv2
    print("=== HikCamera Test (MvImport) ===")
    with HikCamera() as cam:
        cam.configure(exposure_us=5000, gain_db=0.0, fps=10.0)
        cam.start()
        print(f"Camera: {cam.width}×{cam.height}")

        for i in range(30):
            frame = cam.grab(timeout_ms=1000)
            if frame is not None:
                bgr = frame[:, :, ::-1]
                cv2.imshow("HikCamera", bgr)
                print(f"  Frame {i+1}: {frame.shape}, mean={frame.mean():.1f}")
            else:
                print(f"  Frame {i+1}: timeout")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.stop()
        cv2.destroyAllWindows()
