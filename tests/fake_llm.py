"""可编程的假 LLM：让全部单测离线运行，不依赖网络与 API key。

用法
----
按脚本依次返回（最常用）::

    llm = FakeLLM([
        tool_response([("todo", {"action": "add", "content": "买牛奶"})]),
        text_response("已经记下了"),
    ])

按输入内容路由::

    llm = FakeLLM(router=lambda messages, tools: text_response("天气晴"))

调用记录会被保存，便于断言"到底把什么塞进了 context"::

    assert "待办" in llm.calls[0]["messages"][0]["content"]
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Iterable, Sequence

from agent.llm.types import LLMResponse, ToolCall
from agent.errors import LLMHTTPError


# ---------------------------------------------------------------- 构造工具
def text_response(content: str = "好的", thinking: str = "", **kwargs: Any) -> LLMResponse:
    """一条"直接回复"的模型输出。"""
    return LLMResponse(
        content=content,
        thinking=thinking,
        model=kwargs.pop("model", "fake-model"),
        finish_reason=kwargs.pop("finish_reason", "stop"),
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        **kwargs,
    )


def tool_response(
    calls: Sequence[tuple[str, dict[str, Any]] | ToolCall],
    content: str = "",
    thinking: str = "",
) -> LLMResponse:
    """一条"要求调用工具"的模型输出。

    ``calls`` 可以写成 ``[("calculator", {"expression": "1+1"})]`` 这种简写。
    """
    tool_calls: list[ToolCall] = []
    for item in calls:
        if isinstance(item, ToolCall):
            tool_calls.append(item)
            continue
        name, arguments = item
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=dict(arguments),
                raw_arguments=json.dumps(arguments, ensure_ascii=False),
            )
        )
    return LLMResponse(
        content=content,
        thinking=thinking,
        tool_calls=tool_calls,
        model="fake-model",
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    )


# ---------------------------------------------------------------- FakeLLM
class FakeLLM:
    """可编程的 LLM 客户端替身。

    优先级：``router`` > ``script``（依次消费）> ``default``。
    """

    def __init__(
        self,
        script: Iterable[LLMResponse] | None = None,
        router: Callable[[list[dict], list[dict] | None], LLMResponse] | None = None,
        default: LLMResponse | None = None,
        model: str = "fake-model",
    ) -> None:
        self.script: list[LLMResponse] = list(script or [])
        self.router = router
        self.default = default
        self.model = model
        #: 每次调用的请求快照，测试里用来断言 context 内容
        self.calls: list[dict[str, Any]] = []

    # -- 记录 ---------------------------------------------------------
    def _record(self, messages: list[dict], tools: list[dict] | None) -> None:
        self.calls.append({"messages": messages, "tools": tools})

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_messages(self) -> list[dict]:
        return self.calls[-1]["messages"]

    def last_system_prompt(self) -> str:
        for message in self.last_messages():
            if message.get("role") == "system":
                return message.get("content") or ""
        return ""

    # -- 主入口 -------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._record(messages, tools)
        started = time.perf_counter()
        if self.router is not None:
            response = self.router(messages, tools)
        elif self.script:
            response = self.script.pop(0)
        elif self.default is not None:
            response = self.default
        else:
            raise AssertionError(
                f"FakeLLM 脚本已耗尽，但仍收到第 {len(self.calls)} 次调用；"
                "请补充 script 或设置 default。"
            )
        response.latency_ms = (time.perf_counter() - started) * 1000
        return response


# ---------------------------------------------------------------- FlakyLLM
class FlakyLLM(FakeLLM):
    """按序列抛出异常，用于验证重试与降级。"""

    def __init__(self, failures: Iterable[Exception], then: LLMResponse | None = None) -> None:
        super().__init__()
        self.failures = list(failures)
        self.then = then or text_response("重试成功")

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any) -> LLMResponse:
        self._record(messages, tools)
        if self.failures:
            raise self.failures.pop(0)
        return self.then


# ---------------------------------------------------------------- 便捷构造
def http_429() -> LLMHTTPError:
    return LLMHTTPError(429, "rate limited")


def http_500() -> LLMHTTPError:
    return LLMHTTPError(500, "internal error")


class AssertingLLM(FakeLLM):
    """断言"必须被调用"的场景（如压缩必须触发一次摘要）。"""

    def __init__(self, expected_calls: int, response: LLMResponse) -> None:
        super().__init__()
        self.expected_calls = expected_calls
        self.response = response

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any) -> LLMResponse:
        self._record(messages, tools)
        return self.response
