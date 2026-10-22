"""多厂商适配层：请求组装与响应解析的差异收敛在这里。

目标：主循环只依赖 :class:`LLMClient` 的 ``chat()``，永远不出现
``if provider == "glm"``。厂商差异通过 :class:`ProviderProfile`（能力画像）
+ :class:`Adapter`（协议组装）表达。
"""

from __future__ import annotations

from typing import Any

from ..config import Config, ProviderProfile
from .parser import parse_chat_completion
from .types import LLMResponse


class Adapter:
    """OpenAI-compatible 协议的请求组装器。"""

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def profile(self) -> ProviderProfile:
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
        """厂商私有参数（子类覆盖）。默认不加任何私有字段，保证兼容性优先。"""
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
            # 保守用法：只发 auto。GLM 只支持 auto，OpenAI 系支持 required，
            # 但我们从不需要强制调用（参数校验 + 错误回喂已经足够稳）。
            body["tool_choice"] = "auto"
        if stream:
            # 部分厂商不支持 stream_options，默认关闭以保证兼容性；
            # 代价是流式下拿不到 token 用量，由 trace 用估算值兜底。
            body["stream_options"] = {"include_usage": False}
        body.update(self.provider_extras())
        return body

    # -- 响应 ---------------------------------------------------------
    def parse(self, raw: dict[str, Any]) -> LLMResponse:
        return parse_chat_completion(raw)


class DeepSeekAdapter(Adapter):
    """DeepSeek：推理模型的 reasoning_content 不能回填，这里不做额外处理。"""


class GLMAdapter(Adapter):
    """智谱 GLM：交错思考 + 工具调用时，reasoning_content 必须原样回传。

    这个行为由 ``ProviderProfile.backfill_thinking`` 控制（见 config.py），
    adapter 这里只负责请求体上的差异。
    """

    def provider_extras(self) -> dict[str, Any]:
        # GLM-4.5 系列支持 thinking 开关；不强依赖，交给模型默认行为。
        return {}


class ArkAdapter(Adapter):
    """火山方舟（豆包）：协议同构，深度思考同样返回 reasoning_content。"""


_ADAPTERS = {
    "deepseek": DeepSeekAdapter,
    "glm": GLMAdapter,
    "ark": ArkAdapter,
}


def build_adapter(config: Config) -> Adapter:
    return _ADAPTERS.get(config.provider, Adapter)(config)
