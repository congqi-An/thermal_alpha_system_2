"""反应层安全守卫 — 无 LLM 的纯规则层, 毫秒级响应

由 main.py _reaction_loop 每 2s 驱动, 检测硬件安全边界并触发事件。
安全联锁不依赖 LLM 决策, 直接执行硬件保护动作。
"""
import time
from typing import Optional

from .. import state
from ..config import AGENT_TEMP_LIMIT, QUALITY_THRESHOLD
from .event_bus import EventBus, EventType


class SafetyGuard:
    """反应层: 纯规则安全守卫"""

    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        # 质量低于阈值的起始时间 (用于判断持续时间)
        self._quality_low_since: Optional[float] = None
        # 上一次质量值 (用于突降检测)
        self._last_quality: float = 1.0
        # 冷却: 避免同一事件短时间内重复触发
        self._cooldowns: dict[str, float] = {}
        self._cooldown_s: float = 10.0  # 同类事件最小间隔

    def check_all(self):
        """每轮反应层调用, 执行所有安全检查 (纯规则, <1ms)"""
        self._check_temperature_safety()
        self._check_quality()

    # ─── 温度安全 (紧急) ─────────────────────────────────────────

    def _check_temperature_safety(self):
        """温度超限 → 立即停止加热 + 停止扫描 + 紧急事件"""
        if state.current_T > AGENT_TEMP_LIMIT:
            # 硬停: 不经过 LLM, 直接切断
            if state.temp_ctrl is not None and state.temp_online:
                try:
                    state.temp_ctrl.regulate_to(25.0)  # 设回室温
                except Exception:
                    pass
            # 必须同时停扫描: 否则 ScanRunner 下一段仍 set_sv(T2) 抬温, 硬停被顶回
            if state.scan_active and state.scan_runner is not None:
                try:
                    state.scan_runner.stop()
                except Exception:
                    pass
            if self._cooldown_ok("temp_safety"):
                self.bus.push(EventType.TEMP_SAFETY, data={
                    "temperature": round(state.current_T, 1),
                    "limit": AGENT_TEMP_LIMIT,
                    "action": "已强制降温至25°C",
                })
                print(f"  [SAFETY] 温度超限! T={state.current_T:.1f}°C > {AGENT_TEMP_LIMIT}°C, 已强制降温")

    # ─── 条纹质量 ────────────────────────────────────────────────

    def _check_quality(self):
        """质量突降 → 条纹丢失(紧急); 质量持续 < 0.4 达 3s → 提醒 (60s 内最多一次)。

        Q 偏低提醒: 连续 3 秒内 Q < 0.4 → QUALITY_DROP 事件
        (反应层主循环捕获后调 LLM 生成文案, 注入对话)。

        门控: Q 类告警仅在测量相关阶段开启——
          * 引导阶段 >= 4 (光路细调起, 相机开始看条纹)
          * 或实际测量/扫描中 (measure_active / scan_active)
        实验一开始(光路未调/相机未对准) Q 天然低, 不应误报"实验系统提醒"。
        """
        q = state.current_quality
        now = time.time()

        if not self._quality_alerts_enabled():
            self._quality_low_since = None   # 关闭期间不累积计时
            self._last_quality = q           # 基线跟随, 开启时不误判突降
            return

        # 突降检测: 质量瞬间下降 > 0.4 且落至 0.3 以下 → 条纹可能消失
        if self._last_quality - q > 0.4 and q < 0.3:
            if self._cooldown_ok("fringe_lost"):
                self.bus.push(EventType.FRINGE_LOST, data={
                    "quality": round(q, 3),
                    "prev_quality": round(self._last_quality, 3),
                    "drop": round(self._last_quality - q, 3),
                })
                print(f"  [SAFETY] 条纹疑似消失! 质量从 {self._last_quality:.2f} 突降至 {q:.2f}")

        # 持续低质量: Q < 0.4 连续 3 秒 → 提醒 (60s 冷却, 一分钟内最多一次)
        if q < QUALITY_THRESHOLD:
            if self._quality_low_since is None:
                self._quality_low_since = now
            duration = now - self._quality_low_since
            if duration > 3.0 and self._cooldown_ok("quality_drop", 60.0):
                self.bus.push(EventType.QUALITY_DROP, data={
                    "quality": round(q, 3),
                    "threshold": QUALITY_THRESHOLD,
                    "duration_s": round(duration, 1),
                })
                print(f"  [SAFETY] Q 持续偏低! q={q:.2f} < {QUALITY_THRESHOLD} 已 {duration:.0f}s")
        else:
            self._quality_low_since = None  # 恢复

        self._last_quality = q

    @staticmethod
    def _quality_alerts_enabled() -> bool:
        """Q 类告警是否开启: 引导阶段 >=4(光路细调起) 或 测量/扫描中."""
        if state.measure_active or state.scan_active:
            return True
        agent = getattr(state, "agent", None)
        guide = getattr(agent, "guide", None)
        if guide is not None and guide.active:
            return guide.idx >= 3    # 阶段4 光路细调 (0-indexed idx 3)
        return False

    # ─── 冷却机制 ────────────────────────────────────────────────

    def _cooldown_ok(self, key: str, duration: Optional[float] = None) -> bool:
        """检查事件冷却, 避免短时间内重复触发。duration 可覆盖默认 _cooldown_s。"""
        now = time.time()
        last = self._cooldowns.get(key, 0)
        limit = duration if duration is not None else self._cooldown_s
        if now - last < limit:
            return False
        self._cooldowns[key] = now
        return True

    def reset(self):
        """重置状态 (Agent 重启时调用)"""
        self._quality_low_since = None
        self._last_quality = 1.0
        self._cooldowns.clear()
