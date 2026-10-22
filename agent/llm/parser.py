"""LLM 输出的归一化解析。

把 DeepSeek 的原始响应翻译成内部统一的 :class:`LLMResponse`，同时吸收掉
真实调用里会遇到的脏数据。上层循环只面对归一化结果，不需要知道协议长什么样。

处理的真实差异与脏数据
----------------------
1. 思考过程在 ``reasoning_content``（DeepSeek 的推理模型会返回）。
2. ``content`` 可能是字符串，也可能是 block 数组（网关偶尔会包一层）。
3. 流式 ``tool_calls`` 按 ``index`` 分片：首片带 id/name，后续只带 arguments 片段，
   必须拼完才能 parse，中途 parse 必然失败。
4. 模型偶尔把工具调用**写进 content**（```json 围栏或裸 JSON），需要兜底抽取。
5. ``arguments`` 可能是截断/单引号/尾逗号的脏 JSON。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from .types import LLMResponse, ToolCall

#: 从 content 里抠工具调用 JSON 的围栏正则
_FENCED_JSON = re.compile(r"```(?:json|tool_call|tool|python)?\s*(\{.*?\})\s*```", re.DOTALL)

#: 思考过程可能出现的字段名，按优先级
_THINKING_KEYS = ("reasoning_content", "reasoning", "thinking", "reasoning_text", "thought")


# ---------------------------------------------------------------- 思考过程
def extract_thinking(message: dict[str, Any]) -> str:
    """从 message 里提取思考过程文本。"""
    for key in _THINKING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            for inner in ("text", "content", "thinking"):
                text = value.get(inner)
                if isinstance(text, str) and text.strip():
                    return text
        if isinstance(value, list):
            parts = [
                str(item.get("text") or item.get("thinking") or "")
                for item in value
                if isinstance(item, dict)
            ]
            merged = "".join(parts).strip()
            if merged:
                return merged
    return ""


# ---------------------------------------------------------------- content
def normalize_content(content: Any) -> str:
    """content 可能是 str / None / block 数组。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # {"type":"text","text":"..."} / {"type":"thinking",...}
                if item.get("type") in ("thinking", "reasoning"):
                    continue
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------- 脏 JSON
def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        body = lines[1:] if len(lines) > 1 else []
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        return "\n".join(body).strip()
    return stripped


def _extract_first_object(text: str) -> dict[str, Any] | None:
    """扫描出第一个括号平衡的 JSON 对象（应对"前面有解释文字"的场景）。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    return None


def _try_json(text: str) -> Any:
    """三级修复：原样 -> 去围栏 -> 补尾逗号/单引号后重试。"""
    candidates = [text, _strip_fence(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    repaired = _strip_fence(text)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)          # 去尾逗号
    repaired = repaired.replace("'", '"')                        # 单引号
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------- 兜底抽取
def extract_text_tool_calls(content: str) -> list[ToolCall]:
    """当模型没有走原生 tool_calls 时，从自然语言里抽取工具调用。

    支持的形态::

        {"name": "calculator", "arguments": {"expression": "1+1"}}
        {"tool": "todo", "args": {"action": "list"}}
        ```json
        {"name": "search", "parameters": {"query": "..."}}
        ```
    """
    if not content or "{" not in content:
        return []

    candidates: list[str] = []
    fenced = _FENCED_JSON.findall(content)
    if fenced:
        candidates.extend(fenced)
    candidates.append(content)

    for raw in candidates:
        parsed = _try_json(raw)
        if parsed is None:
            parsed = _extract_first_object(raw)
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name") or parsed.get("tool") or parsed.get("tool_name")
        arguments = parsed.get("arguments") or parsed.get("parameters") or parsed.get("args") or {}
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(arguments, str):
            decoded = _try_json(arguments)
            arguments = decoded if isinstance(decoded, dict) else {}
        if not isinstance(arguments, dict):
            continue
        return [
            ToolCall(
                id="",
                name=name.strip(),
                arguments=arguments,
                raw_arguments=json.dumps(arguments, ensure_ascii=False),
            )
        ]
    return []


# ---------------------------------------------------------------- 非流式
def parse_chat_completion(raw: dict[str, Any]) -> LLMResponse:
    """解析非流式的 chat/completions 响应。"""
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content = normalize_content(message.get("content"))
    thinking = extract_thinking(message) or extract_thinking(choice)
    tool_calls = [ToolCall.from_openai(item) for item in message.get("tool_calls") or []]

    # 兜底：模型把工具调用写进了 content
    if not tool_calls and content:
        tool_calls = extract_text_tool_calls(content)

    usage = raw.get("usage") or {}
    return LLMResponse(
        content=content,
        thinking=thinking,
        tool_calls=tool_calls,
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        model=str(raw.get("model") or ""),
        finish_reason=str(choice.get("finish_reason") or ""),
        raw=raw,
    )


# ---------------------------------------------------------------- 流式
class StreamAccumulator:
    """流式响应的累积器：边收边吐文本，最后产出完整的 :class:`LLMResponse`。

    关键点：**工具调用必须收完再 parse**。首片只有 id/name，参数在后续片里
    逐字到达，中途 ``json.loads`` 必然抛异常。
    """

    def __init__(self) -> None:
        self._content: list[str] = []
        self._thinking: list[str] = []
        self._tools: dict[int, dict[str, Any]] = {}
        self._finish_reason = ""
        self._model = ""
        self._usage: dict[str, int] = {}

    def feed(self, chunk: dict[str, Any]) -> str:
        """喂一个 SSE chunk，返回本次新增的可见文本（用于打字机效果）。"""
        self._model = chunk.get("model") or self._model
        if chunk.get("usage"):
            usage = chunk["usage"]
            self._usage = {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }

        choices = chunk.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            self._finish_reason = str(choice["finish_reason"])

        # 思考过程
        thinking_delta = extract_thinking(delta)
        if thinking_delta:
            self._thinking.append(thinking_delta)

        # 可见文本
        text = normalize_content(delta.get("content"))
        if text:
            self._content.append(text)

        # 工具调用分片
        for item in delta.get("tool_calls") or []:
            index = int(item.get("index") or 0)
            slot = self._tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if item.get("id") and not slot["id"]:
                slot["id"] = str(item["id"])
            function = item.get("function") or {}
            if function.get("name") and not slot["name"]:
                slot["name"] = str(function["name"])
            fragment = function.get("arguments")
            if fragment:
                slot["arguments"] += str(fragment)
        return text

    @property
    def thinking_text(self) -> str:
        return "".join(self._thinking)

    def finalize(self) -> LLMResponse:
        tool_calls: list[ToolCall] = []
        for index in sorted(self._tools):
            slot = self._tools[index]
            raw_arguments = slot["arguments"]
            parsed = _try_json(raw_arguments) if raw_arguments else {}
            if not isinstance(parsed, dict):
                parsed = {}
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=parsed,
                    raw_arguments=raw_arguments,
                )
            )
        content = "".join(self._content)
        if not tool_calls and content:
            tool_calls = extract_text_tool_calls(content)
        return LLMResponse(
            content=content,
            thinking="".join(self._thinking),
            tool_calls=[c for c in tool_calls if c.name],
            usage=self._usage,
            model=self._model,
            finish_reason=self._finish_reason,
        )


# ---------------------------------------------------------------- SSE 工具
def parse_sse_line(line: str) -> dict[str, Any] | None:
    """解析一行 SSE。返回 None 表示 [DONE] 或无法解析。"""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def iter_sse_chunks(lines: Iterable[bytes | str]) -> Iterable[dict[str, Any]]:
    """把 urllib 的逐行输出变成 chunk 流。"""
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        chunk = parse_sse_line(line)
        if chunk is not None:
            yield chunk
