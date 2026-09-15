# -*- coding: utf-8 -*-
"""NRateGate — N 振动控制门 (惯性覆盖, 速率限制)

目标: 大震动时计数器 N 会跳变, 但任何一秒内 N 变化应 ≤ max_rate (默认 2 条/秒)。

机制:
  正常帧(测量自身变化 ≤ cap): N 跟随测量, 该帧 N 变化进入"干净帧窗口"(最近 win 帧均值)。
  振动帧(测量自身变化 > cap):  N 不跟测量, 按干净帧窗口的平均变化推进(惯性覆盖)。
  恢复期(N 落后测量):           限速爬坡收敛, 不污染干净窗口。

性质:
  - |dN/dt| ≤ max_rate 恒成立。
  - 振动尖峰被覆盖(不进入 N), 振动结束 N 以 ≤max_rate 收敛回测量。
  - 正常慢速(热膨胀 ~0.2 条/s)不受影响; 快速扫描若超 max_rate 会被限速(滞后, 不丢总计数)。
"""
import numpy as np


class NRateGate:
    def __init__(self, max_rate: float = 2.0, win: int = 30):
        self.max_rate = max_rate       # 条/秒 硬上限
        self.win = win                 # 干净帧窗口大小 (帧)
        self.N = None                  # 门后 N (带方向)
        self.t = None
        self.n_prev = None             # 上一帧测量 (检测跳变)
        self.clean = []                # 最近干净帧的 N 变化 (条/帧)
        self.covered = 0               # 覆盖(振动)帧数
        self.capped = 0                # 恢复期限速帧数

    def reset(self):
        self.__init__(self.max_rate, self.win)

    def update(self, n_meas: float, t: float) -> float:
        n_meas = float(n_meas)
        if self.N is None:
            self.N = n_meas
            self.t = t
            self.n_prev = n_meas
            return self.N
        dt = max(t - self.t, 1e-6)
        cap = self.max_rate * dt
        d_meas = n_meas - self.n_prev       # 测量自身的变化
        self.n_prev = n_meas
        if abs(d_meas) > cap:               # 测量跳变 → 振动帧, 惯性覆盖
            self.covered += 1
            if self.clean:
                rate = float(np.mean(self.clean)) / dt   # 平均条/帧 → 条/秒
                self.N += float(np.clip(rate * dt, -cap, cap))
            # 无干净样本: 保持 N 不动
        else:                               # 正常/恢复帧
            d = n_meas - self.N
            if abs(d) <= cap:               # N 已跟上: 跟随测量, 更新干净窗口
                self.N = n_meas
                self.clean.append(d_meas)
                if len(self.clean) > self.win:
                    self.clean.pop(0)
            else:                           # N 落后(刚过振动): 限速爬坡, 不污染窗口
                self.capped += 1
                self.N += float(np.clip(d, -cap, cap))
        self.t = t
        return self.N
