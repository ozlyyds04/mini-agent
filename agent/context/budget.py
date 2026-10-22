"""Token 预算估算与四档水位。

**这是估算，不是 tokenizer**。目的不是复刻某个厂商的 BPE，而是提供一个
"同一坐标系下比较大小"的稳定标尺：压缩前/后能比、不同 session 能比、
水位阈值（60% / 80% / 92%）能落地。刻意不引入 tiktoken —— 零第三方依赖是本项目的
卖点，而且真实 token 数只有请求发出去后 usage 才准确。

经验公式（按字符类型分档）
--------------------------
* CJK 字符（汉字 / 假名 / 谚文 / 全角标点）≈ 1 token / 字；
* ASCII 字母数字 ≈ 1 token / 4 字符；
* ASCII 标点与空白 ≈ 1 token / 2 字符；
* 其余（emoji 等）≈ 1 token / 字符。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ..llm.types import Message

#: 每条消息固定的协议开销（role 字段、分隔符等）
MESSAGE_OVERHEAD = 4

#: CJK 相关区间：这些字符在主流 BPE 里基本是 1 token 上下
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),   # CJK 标点
    (0x3040, 0x30FF),   # 日文假名
    (0x3400, 0x4DBF),   # CJK 扩展 A
    (0x4E00, 0x9FFF),   # CJK 基本区
    (0xAC00, 0xD7AF),   # 谚文
    (0xF900, 0xFAFF),   # CJK 兼容
    (0xFF00, 0xFFEF),   # 全角字符
)


def _is_cjk(code: int) -> bool:
    for low, high in _CJK_RANGES:
        if low <= code <= high:
            return True
    return False


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（见模块 docstring 的经验公式）。"""
    if not text:
        return 0
    cjk = 0
    ascii_alnum = 0
    ascii_other = 0
    other = 0
    for char in text:
        code = ord(char)
        if _is_cjk(code):
            cjk += 1
        elif code < 128:
            if char.isalnum():
                ascii_alnum += 1
            else:
                ascii_other += 1
        else:
            other += 1
    total = cjk + ascii_alnum / 4 + ascii_other / 2 + other
    return max(1, math.ceil(total))


def _estimate_tool_calls(tool_calls: Any) -> int:
    """tool_calls 按 JSON 序列化后的长度粗算（arguments 通常很短）。"""
    if not tool_calls:
        return 0
    try:
        if isinstance(tool_calls, list):
            payload = json.dumps(
                [call.to_dict() if hasattr(call, "to_dict") else call for call in tool_calls],
                ensure_ascii=False,
            )
        else:  # pragma: no cover - 防御性分支
            payload = json.dumps(tool_calls, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover
        payload = str(tool_calls)
    return estimate_tokens(payload)


def estimate_messages_tokens(messages: list[dict[str, Any]] | list[Message]) -> int:
    """估算一组消息（``Message`` 或 OpenAI dict）的 token 数。"""
    total = 0
    for message in messages:
        total += MESSAGE_OVERHEAD
        if isinstance(message, Message):
            total += estimate_tokens(message.content)
            if message.name:
                total += estimate_tokens(message.name)
            if message.tool_call_id:
                total += estimate_tokens(message.tool_call_id)
            total += _estimate_tool_calls(message.tool_calls)
        elif isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                total += estimate_tokens(content)
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str):
                total += estimate_tokens(reasoning)
            name = message.get("name")
            if isinstance(name, str) and name:
                total += estimate_tokens(name)
            total += _estimate_tool_calls(message.get("tool_calls"))
        else:  # pragma: no cover - 防御性分支
            total += estimate_tokens(str(message))
    return total


# ---------------------------------------------------------------- 水位
@dataclass
class ContextBudget:
    """上下文预算与四档水位。

    * ``0``：安全区，只做 L0（tool 结果截断），不动历史；
    * ``1``：L1，把最旧的若干轮折叠成增量摘要；
    * ``2``：L2，递归重写摘要，产物更短；
    * ``3``：L3，不再调模型，直接硬裁剪 + 归档。
    """

    total: int
    l1_ratio: float = 0.60
    l2_ratio: float = 0.80
    l3_ratio: float = 0.92

    @classmethod
    def from_config(cls, config: Any) -> "ContextBudget":
        total = int(getattr(config, "context_budget_tokens", 8000) or 8000)
        return cls(
            total=max(1, total),
            l1_ratio=float(getattr(config, "compress_l1_ratio", 0.60)),
            l2_ratio=float(getattr(config, "compress_l2_ratio", 0.80)),
            l3_ratio=float(getattr(config, "compress_l3_ratio", 0.92)),
        )

    def used(self, tokens: int) -> float:
        """返回占用比例（total 非法时按 1.0 处理，宁可早压缩也别超窗）。"""
        if self.total <= 0:  # pragma: no cover - from_config 已保证 >=1
            return 1.0
        return max(0.0, tokens / self.total)

    def level(self, ratio: float) -> int:
        """比例 -> 水位等级 0/1/2/3（阈值来自 Config 的 compress_*_ratio）。"""
        if ratio >= self.l3_ratio:
            return 3
        if ratio >= self.l2_ratio:
            return 2
        if ratio >= self.l1_ratio:
            return 1
        return 0

    def limit(self, ratio: float) -> int:
        """某个水位对应的 token 上限（例如 builder 的硬预算）。"""
        return max(0, int(self.total * ratio))
