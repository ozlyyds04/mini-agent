"""运行时事件流。

CLI、trace、测试都通过订阅事件来观察运行时，而不是让主循环直接 print。
好处：主循环保持纯粹（不关心展示），且"工具调用 trace"这个题目要求变成
一个自然的副产品。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass
class Event:
    """一条运行时事件。"""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = field(default_factory=now_iso)
    trace_id: str = ""


#: 常用事件类型（字符串而非枚举，方便扩展）
RUN_START = "run.start"
RUN_END = "run.end"
STEP_START = "step.start"
USER_MESSAGE = "user.message"
LLM_REQUEST = "llm.request"
LLM_DELTA = "llm.delta"
THINKING = "llm.thinking"
TOOL_CALL = "tool.call"
TOOL_RESULT = "tool.result"
ASYNC_SUBMIT = "tool.async.submit"
ASYNC_DONE = "tool.async.done"
CONTEXT_BUILT = "context.built"
COMPRESSION = "context.compression"
MEMORY_RECALL = "memory.recall"
FINAL = "assistant.final"
ERROR = "run.error"
CANCELLED = "run.cancelled"


class EventBus:
    """极简同步事件总线（不引入 asyncio，保持零依赖与可测试）。"""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, handler: Callable[[Event], None]) -> Callable[[], None]:
        self._subscribers.append(handler)

        def unsubscribe() -> None:
            if handler in self._subscribers:
                self._subscribers.remove(handler)

        return unsubscribe

    def publish(self, event: Event) -> None:
        for handler in list(self._subscribers):
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - 订阅者异常不能影响主流程
                continue

    def emit(self, kind: str, trace_id: str = "", **payload: Any) -> Event:
        event = Event(kind=kind, payload=payload, trace_id=trace_id)
        self.publish(event)
        return event
