"""LLM 交互的统一数据结构。

一次模型调用在协议层的产物只有三样东西，这里把它们显式建模，避免
"到处传 dict、字段名靠记忆"的常见问题：

* :class:`ToolCall`   —— 模型要求调用的工具（id / name / arguments）
* :class:`Message`    —— 上下文中的一条消息（含 thinking、tool_calls、tool_call_id）
* :class:`LLMResponse`—— 一次调用的归一化结果（思考、工具调用、最终文本）

之所以需要这层，是因为 Anthropic 的 ``tool_use``/``tool_result`` block 与
OpenAI 的 ``tool_calls``/``role:"tool"`` 语义不同，上层循环不应该感知差异。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------- ToolCall
@dataclass
class ToolCall:
    """模型请求的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: 原始参数文本（解析失败时用于诊断与回喂模型）
    raw_arguments: str = ""

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> "ToolCall":
        fn = raw.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args, raw_text = raw_args, json.dumps(raw_args, ensure_ascii=False)
        else:
            raw_text = raw_args or ""
            args = _safe_json_object(raw_text)
        return cls(
            id=str(raw.get("id") or ""),
            name=str(fn.get("name") or ""),
            arguments=args,
            raw_arguments=raw_text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            arguments=dict(data.get("arguments") or {}),
            raw_arguments=str(data.get("raw_arguments") or ""),
        )


def _safe_json_object(text: str) -> dict[str, Any]:
    """尽力把模型给的参数文本解析成对象（脏数据兜底的第一层）。

    真实的失败形态：参数被截断、外面包了 ```json 围栏、用了单引号、尾逗号。
    这里只做"便宜的修复"，修不好就返回 {}，由上层把错误回喂给模型让它重来。
    """
    if not text or not text.strip():
        return {}
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        # ```json ... ``` 或 ``` ... ```
        body = stripped.split("\n", 1)[-1]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        candidates.append(body.strip())

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            continue

    # 第三级修复：去尾逗号 + 单引号换双引号（只对"看起来像对象"的文本做）
    repaired = candidates[-1]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = repaired.replace("'", '"')
    try:
        value = json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------- Message
_ROLE_VALUES = {"system", "user", "assistant", "tool"}


@dataclass
class Message:
    """上下文中的一条消息。"""

    role: str
    content: str = ""
    #: Agent 的思考过程（DeepSeek reasoning_content / GLM reasoning_content）。
    #: 是否回填给下一轮由 ProviderProfile.backfill_thinking 决定。
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: role == "tool" 时必填，必须与 request 中的 tool_call.id 严格配对。
    tool_call_id: str = ""
    #: 工具名（部分厂商的 tool 消息需要 name 字段）
    name: str = ""
    #: 内部标记：这条消息是否已被压缩摘要覆盖（不参与序列化到 LLM）
    compressed: bool = False

    def __post_init__(self) -> None:
        if self.role not in _ROLE_VALUES:
            raise ValueError(f"invalid message role: {self.role!r}")

    # -- 输出给 LLM ---------------------------------------------------
    def to_openai(self, backfill_thinking: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.role == "assistant":
            payload["content"] = self.content or None
            if self.tool_calls:
                payload["tool_calls"] = [c.to_openai() for c in self.tool_calls]
            if backfill_thinking and self.thinking:
                payload["reasoning_content"] = self.thinking
        elif self.role == "tool":
            payload["content"] = self.content
            payload["tool_call_id"] = self.tool_call_id
            if self.name:
                payload["name"] = self.name
        else:
            payload["content"] = self.content
        return payload

    # -- 持久化 -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.thinking:
            data["thinking"] = self.thinking
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=str(data.get("role") or "user"),
            content=str(data.get("content") or ""),
            thinking=str(data.get("thinking") or ""),
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls") or []],
            tool_call_id=str(data.get("tool_call_id") or ""),
            name=str(data.get("name") or ""),
        )

    # -- 便捷构造 -----------------------------------------------------
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        thinking: str = "",
        tool_calls: list[ToolCall] | None = None,
    ) -> "Message":
        return cls(
            role="assistant",
            content=content,
            thinking=thinking,
            tool_calls=list(tool_calls or []),
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: str = "") -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


# ---------------------------------------------------------------- LLMResponse
@dataclass
class LLMResponse:
    """一次 LLM 调用的归一化结果。"""

    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def text(self) -> str:
        return self.content.strip()
