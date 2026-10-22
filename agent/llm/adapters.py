"""DeepSeek 协议适配：请求组装与响应解析。

只有一家厂商，所以这里刻意保持"薄"：不做厂商分支、不塞私有参数，
只负责把内部数据结构翻译成 DeepSeek 的 chat/completions 请求体。
如果将来要接第二家，扩展点就是继承 :class:`Adapter` 覆盖 :meth:`payload` /
:meth:`provider_extras`，主循环不用改。
"""

from __future__ import annotations

from typing import Any

from ..config import Config, ModelProfile
from .parser import parse_chat_completion
from .types import LLMResponse


class Adapter:
    """DeepSeek（OpenAI-compatible）请求组装器。"""

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def profile(self) -> ModelProfile:
        return self.config.profile

    # -- 请求 ---------------------------------------------------------
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": "mini-agent/1.0",
        }

    def provider_extras(self) -> dict[str, Any]:
        """私有参数扩展点。默认不加任何字段，保证协议兼容性优先。"""
        return {}

    def payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            # 保守用法：只发 auto。我们从不需要强制调用——参数校验 + 错误回喂
            # 已经足够稳，而强制调用会让模型在"该直接回答"时也去调工具。
            body["tool_choice"] = "auto"
        if stream:
            # DeepSeek 支持 stream_options，但打开后部分网关会多返回空 choices；
            # 这里关闭，token 用量由 trace 用估算值兜底。
            body["stream_options"] = {"include_usage": False}
        body.update(self.provider_extras())
        return body

    # -- 响应 ---------------------------------------------------------
    def parse(self, raw: dict[str, Any]) -> LLMResponse:
        return parse_chat_completion(raw)


def build_adapter(config: Config) -> Adapter:
    return Adapter(config)
