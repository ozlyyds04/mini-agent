"""工具执行器：校验 → 超时 → 异常 → 统一 ToolResult。

核心设计：**工具失败不打断主循环**。
工具名未知、参数不合法、执行抛异常、执行超时，全部被翻译成
``ToolResult(ok=False, error=...)`` 回喂给模型，让它自己修正后重试。
这是"基本异常处理"之外真正能体现 Agent 工程经验的地方——直接抛异常会
让一次小失误废掉整轮对话。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from ..errors import ToolValidationError
from ..trace.tracer import Tracer
from ..tools.base import Tool, ToolContext, ToolResult
from ..tools.registry import ToolRegistry
from .events import TOOL_CALL, TOOL_RESULT, EventBus

logger = logging.getLogger(__name__)


class ToolExecutor:
    """负责把一次 ToolCall 变成 ToolResult。"""

    def __init__(
        self,
        registry: ToolRegistry,
        tracer: Tracer | None = None,
        bus: EventBus | None = None,
        max_workers: int = 4,
    ) -> None:
        self.registry = registry
        self.tracer = tracer
        self.bus = bus
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mini-agent-tool")

    # ---------------------------------------------------------------- 主入口
    def execute(self, call: Any, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        tool = self.registry.get(getattr(call, "name", ""))

        # 1) 未知工具：把可用清单回喂，模型通常下一次就能改对
        if tool is None:
            return self._finish(
                call,
                ToolResult.failure(
                    f"未知工具 {call.name!r}。可用工具：{', '.join(self.registry.names())}"
                ),
                started,
            )

        # 2) 参数校验（缺必填 / 未知参数 / 枚举越界 / 类型强制失败）
        try:
            arguments = tool.normalize_arguments(dict(getattr(call, "arguments", {}) or {}))
        except ToolValidationError as exc:
            return self._finish(call, ToolResult.failure(str(exc)), started)
        except Exception as exc:  # noqa: BLE001 - 校验器自身异常也不能打断循环
            return self._finish(
                call, ToolResult.failure(f"参数校验失败：{type(exc).__name__}: {exc}"), started
            )

        self._emit(TOOL_CALL, tool=tool.name, arguments=arguments, call_id=getattr(call, "id", ""))

        # 3) 执行（带超时；工具内部的异常也转成失败结果）
        result = self._invoke(tool, arguments, ctx)
        return self._finish(call, result, started)

    # ---------------------------------------------------------------- 内部
    def _invoke(self, tool: Tool, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        span_ctx = self.tracer.span("tool.call", tool=tool.name) if self.tracer else _NullSpan()
        with span_ctx as span:
            try:
                future: Future = self._pool.submit(tool, ctx, **arguments)
                raw = future.result(timeout=tool.timeout_s)
            except FutureTimeoutError:
                span.fields["timeout"] = True
                return ToolResult.failure(
                    f"工具 {tool.name} 执行超时（>{tool.timeout_s}s）。"
                    "请尝试更小范围的调用，或改用其他工具。"
                )
            except Exception as exc:  # noqa: BLE001 - 工具异常 = 可恢复的观测
                logger.warning("工具 %s 执行异常: %s", tool.name, exc)
                return ToolResult.failure(
                    f"工具 {tool.name} 执行异常：{type(exc).__name__}: {exc}"
                )

            # 允许工具返回 str / dict / ToolResult
            if isinstance(raw, ToolResult):
                return raw
            if isinstance(raw, str):
                return ToolResult.success(raw)
            return ToolResult.success(str(raw))

    def _finish(self, call: Any, result: ToolResult, started: float) -> ToolResult:
        duration_ms = (time.perf_counter() - started) * 1000
        result.meta.setdefault("duration_ms", round(duration_ms, 2))
        result.meta.setdefault("tool", getattr(call, "name", ""))
        self._emit(
            TOOL_RESULT,
            tool=getattr(call, "name", ""),
            ok=result.ok,
            duration_ms=result.meta["duration_ms"],
            error=result.error,
            preview=(result.content or result.error)[:200],
        )
        return result

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.bus is not None:
            self.bus.emit(kind, **payload)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


class _NullSpan:
    """tracer 不存在时的空 span，避免调用点写 if。"""

    fields: dict[str, Any] = {}

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False
