"""工具执行器测试：异常 / 超时 / 校验失败都必须变成可观测的 ToolResult。"""

from __future__ import annotations

import time
import unittest
from typing import Literal

from agent.errors import ToolValidationError
from agent.llm.types import ToolCall
from agent.runtime.events import TOOL_CALL, TOOL_RESULT, EventBus
from agent.runtime.executor import ToolExecutor
from agent.tools.base import ToolContext, ToolResult, tool
from agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------- 测试工具
@tool(name="echo", description="回显输入")
def echo(text: str) -> str:
    """回显。

    Args:
        text: 要回显的文本
    """
    return f"echo:{text}"


@tool(name="boom", description="总是抛异常")
def boom(reason: str = "boom") -> str:
    raise RuntimeError(reason)


@tool(name="slow", description="慢工具", timeout_s=0.05)
def slow(seconds: float = 0.5) -> str:
    time.sleep(seconds)
    return "done"


@tool(name="with_ctx", description="需要会话上下文")
def with_ctx(ctx: ToolContext, key: str = "x") -> str:
    return f"session={ctx.session_id},key={key}"


@tool(name="pick", description="枚举参数")
def pick(action: Literal["add", "list"] = "list") -> str:
    return action


def make_executor(bus: EventBus | None = None) -> ToolExecutor:
    registry = ToolRegistry([echo, boom, slow, with_ctx, pick])
    return ToolExecutor(registry, bus=bus)


class TestToolExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = EventBus()
        self.events = []
        self.bus.subscribe(lambda event: self.events.append(event))
        self.executor = make_executor(self.bus)
        self.ctx = ToolContext(session=None, session_id="s-1")

    def tearDown(self) -> None:
        self.executor.shutdown()

    def call(self, name: str, **arguments) -> ToolResult:
        return self.executor.execute(ToolCall(id="c1", name=name, arguments=arguments), self.ctx)

    def test_success(self) -> None:
        result = self.call("echo", text="hi")
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "echo:hi")
        self.assertIn("duration_ms", result.meta)

    def test_unknown_tool_lists_available(self) -> None:
        result = self.call("nope")
        self.assertFalse(result.ok)
        self.assertIn("nope", result.error)
        self.assertIn("echo", result.error)  # 回喂可用清单，模型可自愈

    def test_missing_required_argument(self) -> None:
        result = self.call("echo")
        self.assertFalse(result.ok)
        self.assertIn("缺少必填参数", result.error)
        self.assertIn("text", result.error)

    def test_unknown_argument_rejected(self) -> None:
        result = self.call("echo", text="hi", bogus=1)
        self.assertFalse(result.ok)
        self.assertIn("未知参数", result.error)

    def test_enum_violation(self) -> None:
        result = self.call("pick", action="delete")
        self.assertFalse(result.ok)
        self.assertIn("枚举", result.error)

    def test_type_coercion(self) -> None:
        result = self.call("slow", seconds="0.01")
        self.assertTrue(result.ok, result.error)

    def test_tool_exception_becomes_failure(self) -> None:
        result = self.call("boom", reason="kaboom")
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.error)

    def test_timeout(self) -> None:
        result = self.call("slow", seconds=0.5)
        self.assertFalse(result.ok)
        self.assertIn("超时", result.error)

    def test_context_injected(self) -> None:
        result = self.call("with_ctx", key="k1")
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "session=s-1,key=k1")

    def test_events_emitted(self) -> None:
        self.call("echo", text="hi")
        kinds = [event.kind for event in self.events]
        self.assertIn(TOOL_CALL, kinds)
        self.assertIn(TOOL_RESULT, kinds)
        result_event = next(event for event in self.events if event.kind == TOOL_RESULT)
        self.assertTrue(result_event.payload["ok"])

    def test_failure_observation_contains_hint(self) -> None:
        result = self.call("nope")
        observation = result.to_observation()
        self.assertIn("[tool_error]", observation)
        self.assertIn("修正", observation)

    def test_validation_error_type_is_tool_error(self) -> None:
        with self.assertRaises(ToolValidationError):
            echo.normalize_arguments({})


if __name__ == "__main__":
    unittest.main()
