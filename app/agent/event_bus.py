"""反应层使用的异步事件总线。"""
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """事件类型"""
    # 安全/紧急
    TEMP_SAFETY = "temp_safety"            # 温度超限 (紧急, 插队)
    FRINGE_LOST = "fringe_lost"            # 条纹突然消失 (紧急)
    # 异常/警告
    QUALITY_DROP = "quality_drop"          # 质量持续低于阈值


class Priority(int, Enum):
    """事件优先级 (数值越小越优先)"""
    CRITICAL = 0    # 安全联锁, 立即处理
    HIGH = 1        # 异常告警, 下一轮处理
    NORMAL = 2      # 普通事件


# 事件类型 → 默认优先级映射
_TYPE_PRIORITY = {
    EventType.TEMP_SAFETY: Priority.CRITICAL,
    EventType.FRINGE_LOST: Priority.CRITICAL,
    EventType.QUALITY_DROP: Priority.HIGH,
}


@dataclass(order=True)
class Event:
    """可按优先级与入队顺序排序的事件。"""
    priority: int                                   # 原始优先级 (参与排序)
    seq: int = field(default=0)                     # 同级 FIFO 序号 (参与排序)
    type: EventType = field(compare=False, default=EventType.QUALITY_DROP)
    data: dict = field(compare=False, default_factory=dict)


class EventBus:
    """异步事件总线，支持优先级队列。"""

    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = 0  # 序号 (同优先级内 FIFO)

    def push(self, event_type: EventType, data: Optional[dict] = None,
             priority: Optional[Priority] = None):
        """推送事件 (仅限事件循环内调用: asyncio.Queue 绑定运行循环,
        在无事件循环的线程调用 put_nowait 会抛 RuntimeError)"""
        if priority is None:
            priority = _TYPE_PRIORITY.get(event_type, Priority.NORMAL)
        self._seq += 1
        event = Event(
            priority=int(priority),
            seq=self._seq,
            type=event_type,
            data=data or {},
        )
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def pop_all(self) -> list[Event]:
        """非阻塞: 一次性取出所有待处理事件"""
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

# 全局单例
event_bus = EventBus()
