"""异步工具：提交即返回，结果在 step 边界以"通知"形式注入。

为什么需要
----------
有些工具（生成报告、跑测试、长时间检索）不能让用户干等。但结果又很重要。
朴素做法是阻塞等待，体验差；直接把结果打断正在进行的 LLM 请求，又会破坏
上下文的一致性（同一轮里混入新信息，模型容易答非所问）。

本实现
------
1. ``submit()`` 立刻返回一个 ``AsyncTask``，主循环马上给模型一个"已受理"的
   工具结果（保证 tool_call / tool_result 配对，避免 400）；
2. 后台线程算完把结果放进队列；
3. 主循环在**下一个 step 边界** ``drain()``，把结果作为一条系统通知注入
   （role="user"，带 ``[异步工具完成]`` 标记），不打断正在进行的请求；
4. 如果本次 run 已经结束，结果留在队列里，**下次 run 开始时自动补投**——
   这就是"结果依然重要"。

注：用 role="user" 而不是 role="tool"，是因为同一条 assistant 消息的
tool_call_id 已经被"已受理"结果占用，重复使用会违反协议。
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..trace.tracer import Tracer
from ..tools.base import Tool, ToolContext, ToolResult
from .events import ASYNC_DONE, ASYNC_SUBMIT, EventBus

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class AsyncTask:
    """一个异步工具任务。"""

    task_id: str
    tool_name: str
    tool_call_id: str
    status: str = "running"          # running | done | error
    result: ToolResult | None = None
    error: str = ""
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""

    def observation(self) -> str:
        """注入上下文时使用的文本。"""
        if self.result is None:
            body = self.error or "（无结果）"
        else:
            body = self.result.content if self.result.ok else f"失败：{self.result.error}"
        return (
            f"[异步工具完成] task_id={self.task_id} tool={self.tool_name} "
            f"耗时开始于 {self.started_at}\n{body}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.result.ok if self.result else False,
        }


class AsyncToolRunner:
    """异步工具的提交与完成通知。"""

    def __init__(
        self,
        tracer: Tracer | None = None,
        bus: EventBus | None = None,
        max_workers: int = 2,
    ) -> None:
        self.tracer = tracer
        self.bus = bus
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mini-agent-async")
        self._finished: "queue.Queue[AsyncTask]" = queue.Queue()
        self._tasks: dict[str, AsyncTask] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 提交
    def submit(
        self,
        tool: Tool,
        ctx: ToolContext,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> AsyncTask:
        task = AsyncTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            tool_name=tool.name,
            tool_call_id=tool_call_id,
        )
        with self._lock:
            self._tasks[task.task_id] = task

        if self.bus is not None:
            self.bus.emit(ASYNC_SUBMIT, task_id=task.task_id, tool=tool.name)

        def work() -> None:
            try:
                raw = tool(ctx, **arguments)
                task.result = raw if isinstance(raw, ToolResult) else ToolResult.success(str(raw))
                task.status = "done" if task.result.ok else "error"
                task.error = task.result.error
            except Exception as exc:  # noqa: BLE001 - 异步任务必须自己兜住异常
                task.status = "error"
                task.error = f"{type(exc).__name__}: {exc}"
                task.result = ToolResult.failure(task.error)
            finally:
                task.finished_at = _now_iso()
                self._finished.put(task)
                if self.tracer is not None:
                    self.tracer.event(
                        "tool.async.done",
                        task_id=task.task_id,
                        tool=task.tool_name,
                        status=task.status,
                    )

        self._pool.submit(work)
        return task

    # ---------------------------------------------------------------- 取结果
    def drain(self) -> list[AsyncTask]:
        """取走所有已完成但尚未上报的任务。"""
        done: list[AsyncTask] = []
        while True:
            try:
                task = self._finished.get_nowait()
            except queue.Empty:
                break
            done.append(task)
        if done and self.bus is not None:
            for task in done:
                self.bus.emit(ASYNC_DONE, task_id=task.task_id, tool=task.tool_name, status=task.status)
        return done

    def pending(self) -> list[AsyncTask]:
        with self._lock:
            return [task for task in self._tasks.values() if task.status == "running"]

    def drained_tasks(self) -> list[AsyncTask]:
        """已结束的任务（含已上报的），给 CLI ``/tasks`` 用。"""
        with self._lock:
            return [task for task in self._tasks.values() if task.status != "running"]

    def wait_all(self, timeout: float = 30.0) -> None:
        """主要给测试与脚本用：等待全部任务结束。"""
        self._pool.shutdown(wait=True)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mini-agent-async")

    def shutdown(self) -> None:
        try:
            self._pool.shutdown(wait=False)
        except RuntimeError:
            pass
