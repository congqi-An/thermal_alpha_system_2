"""视频文件模拟源 — 把本地视频当相机输入, 复用整条分析链路

用途: 无硬件时加载实验视频 (如 D:\\copy_data\\vedio\\*.avi), 走与实拍
完全相同的 grab 链路: 光心/r0 判定 + 质量 Q + 条纹计数 + 信号震荡图。
温度模块不参与 (N 计数不依赖 T, dT/α 结算无意义, 只看 N 和信号)。

接口对齐 grab 循环的用法: grab() 按视频原生 fps 节奏出帧 (未到时间
返回 None), 输出 RGB (与 HikCamera.grab 一致, 红通道=frame[:,:,0])。
"""
import time
import threading
from pathlib import Path

import cv2


class VideoFileSource:
    """视频回放源 (线程安全: grab 仅被 grab 线程调用, 控制来自 API 线程)"""

    def __init__(self, path: str, loop: bool = True, speed: float = 1.0):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"视频不存在: {path}")
        self.path = str(p)
        self.name = p.name
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        if self.fps <= 0 or self.fps > 240:
            self.fps = 20.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.loop = loop
        self.speed = max(0.1, min(8.0, speed))
        self.frame_idx = 0
        self.active = False          # play/pause 开关
        self.finished = False        # 非循环模式播完
        self.looped = False          # 本帧发生循环回卷 (grab 线程消费后清)
        self._next_t = 0.0
        self._lock = threading.Lock()

    # ── 控制 (API 线程调用) ──
    def play(self):
        with self._lock:
            if self.finished:        # 播完后再播: 从头开始, 标记回卷让计数器重置
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                self.finished = False
                self.looped = True
            self.active = True
            self._next_t = 0.0       # 立即出下一帧

    def pause(self):
        self.active = False

    def rewind(self):
        with self._lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.finished = False
            self.looped = True    # 标记回卷: grab 线程据此重置计数器 (与 play-after-finished 一致)

    def release(self):
        self.active = False
        with self._lock:
            try:
                self.cap.release()
            except Exception:
                pass

    # ── 取帧 (grab 线程调用) ──
    def grab(self):
        """按 fps×speed 节奏出帧; 未到时间/暂停/播完返回 None。输出 RGB。"""
        if not self.active:
            return None
        now = time.monotonic()
        if now < self._next_t:
            return None
        with self._lock:
            ret, frame_bgr = self.cap.read()
            if not ret:
                if self.loop and self.n_frames > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0
                    self.looped = True
                    ret, frame_bgr = self.cap.read()
                if not ret:
                    self.active = False
                    self.finished = True
                    return None
            self.frame_idx += 1
        # 节奏推进: 以读帧完成时刻为基准 (解码可能耗时几十ms, 若用读前时刻
        # 会导致慢解码时连续出帧); 正常按 interval 递增, 首帧/落后超过一帧
        # (处理慢、暂停恢复) 时从完成时刻重新起步, 不突发追帧
        done = time.monotonic()
        interval = 1.0 / (self.fps * self.speed)
        if self._next_t <= 0 or done - self._next_t > interval:
            self._next_t = done + interval
        else:
            self._next_t += interval
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def status(self) -> dict:
        return {
            "loaded": True, "name": self.name, "path": self.path,
            "playing": self.active, "finished": self.finished,
            "frame_idx": self.frame_idx, "n_frames": self.n_frames,
            "fps": round(self.fps, 1), "speed": self.speed, "loop": self.loop,
            "size": [self.width, self.height],
        }
