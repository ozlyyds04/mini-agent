"""检索工具：mock 实现，但接口是真的（可替换成真实检索 API）。

当前用内置 ``CORPUS`` + 中文 2-gram 字符重叠做本地打分，**不联网、零依赖**。
接真实检索时只改 ``_backend()``：返回 ``[{"title", "body"}, ...]`` 即接管检索，
返回 ``None`` 则继续走内置语料；格式化、截断、失败语义都不用动。
"""

from __future__ import annotations

import re
from typing import Any

from .base import ToolResult, tool

#: 内置知识库（mock）。每条含 title / body / tags，主题对应本项目的核心设计点。
CORPUS: list[dict[str, Any]] = [
    {
        "title": "Agent 主循环",
        "body": "主循环用 while 驱动模型思考与工具执行，把每次工具结果作为观测回喂，"
        "直到模型给出最终回答或触发 max_steps 上限，防止无限自调用。",
        "tags": ["agent", "主循环", "while", "max_steps"],
    },
    {
        "title": "工具调用协议",
        "body": "OpenAI-compatible 的 function calling：模型返回 tool_calls，执行器校验参数后"
        "调用工具，再把 ToolResult 以 role=tool 的消息回喂，tool_call_id 必须配对。",
        "tags": ["tool_call", "function calling", "schema", "配对"],
    },
    {
        "title": "Session 隔离",
        "body": "每个 session 拥有独立的 messages、todos 与 summary；工具通过 ToolContext 拿当前会话，"
        "模块里禁止全局可变状态，因此两个 session 天然互不影响。",
        "tags": ["session", "隔离", "context", "todo"],
    },
    {
        "title": "Context 压缩",
        "body": "分层压缩：L0 截断超长工具结果；L1/L2 把较早消息折叠成滚动摘要并归档原文；"
        "L3 达到硬水位时直接裁剪最旧消息，同时永远保留第一条 user 消息与 pinned 事实。",
        "tags": ["context", "压缩", "摘要", "budget"],
    },
    {
        "title": "Memory 召回",
        "body": "记忆召回把历史对话的摘要与关键事实按相关性挑回上下文；pinned 事实不参与压缩，"
        "确保用户目标、关键 ID 在长对话里不丢。",
        "tags": ["memory", "召回", "pinned", "摘要"],
    },
    {
        "title": "异步工具",
        "body": "耗时工具立即返回 task_id，真正的结果稍后以事件回投给运行时；这样主循环不必长时间阻塞，"
        "用户体验和 TTFT 都更好。",
        "tags": ["异步", "async", "task_id", "事件"],
    },
    {
        "title": "TTFT 与流式输出",
        "body": "TTFT 是首个 token 的到达时间，流式输出能显著降低感知延迟；观测时同时记录首 token 与总耗时。",
        "tags": ["ttft", "流式", "延迟", "stream"],
    },
    {
        "title": "OpenHands",
        "body": "OpenHands 是开源 Agent 框架，把动作定义成代码（CodeAct），用沙箱执行后把结果回喂模型，"
        "强调可观测与可回放。",
        "tags": ["openhands", "codeact", "框架", "沙箱"],
    },
    {
        "title": "重试与退避",
        "body": "429 与 5xx 属于可重试错误，按指数退避加抖动重试；4xx 重试没有意义，直接失败并给出原因。",
        "tags": ["重试", "退避", "429", "http"],
    },
    {
        "title": "可观测性",
        "body": "每一步记录 prompt token、completion token、延迟与工具结果，trace 可回放，"
        "便于定位是模型选择错了工具，还是工具实现有 bug。",
        "tags": ["trace", "日志", "观测", "token"],
    },
]

_GRAM_CLEAN = re.compile(r"[\W_]+", re.UNICODE)


def _grams(text: str) -> set[str]:
    """中文无需分词：去掉标点空白后按 2-gram 切分。"""
    clean = _GRAM_CLEAN.sub("", (text or "").lower())
    if len(clean) < 2:
        return {clean} if clean else set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def _score(query: str, doc: dict[str, Any]) -> float:
    """命中数加权：标题 x3、标签 x2、正文 x1；整词命中标题再给奖励。"""
    query_grams = _grams(query)
    if not query_grams:
        return 0.0
    score = 3.0 * len(query_grams & _grams(doc["title"]))
    score += 2.0 * len(query_grams & _grams("".join(doc["tags"])))
    score += 1.0 * len(query_grams & _grams(doc["body"]))
    normalized_query = _GRAM_CLEAN.sub("", query.lower())
    if normalized_query and normalized_query in _GRAM_CLEAN.sub("", doc["title"].lower()):
        score += 5.0
    # 长 query 天然命中更多 gram，做轻度归一化，避免长问句霸榜。
    return score / max(1.0, len(query_grams) ** 0.5)


def _snippet(text: str, limit: int = 200) -> str:
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "…"


def _backend(query: str, top_k: int) -> list[dict[str, Any]] | None:
    """真实检索 API 的替换点：返回命中列表即接管检索；默认 ``None`` 走本地语料。"""
    return None


def search(query: str, top_k: int = 3) -> ToolResult:
    """在内置知识库中检索与关键词最相关的文档片段。

    本地 mock 检索（不联网）：中文按 2-gram 做标题/标签/正文的字符重叠加权。
    没有命中属于正常的空结果，返回成功而不是失败。

    Args:
        query: 检索关键词或自然语言问题，例如 ``session 隔离``。
        top_k: 返回条数上限，默认 3，范围 1-10。
    """
    keyword = (query or "").strip()
    if not keyword:
        return ToolResult.failure("query 不能为空，请提供要检索的关键词。")
    try:
        limit = max(1, min(int(top_k), 10))
    except (TypeError, ValueError):
        return ToolResult.failure("top_k 必须是整数，例如 3。")

    hits = _backend(keyword, limit)
    if hits is None:
        ranked = sorted(((_score(keyword, doc), doc) for doc in CORPUS), key=lambda item: (-item[0], item[1]["title"]))
        hits = [doc for score, doc in ranked if score > 0.0][:limit]
    else:
        hits = hits[:limit]

    if not hits:
        return ToolResult.success("没有找到相关内容。")
    blocks = [f"[{index}] {doc['title']}\n{_snippet(doc['body'])}\n" for index, doc in enumerate(hits, start=1)]
    return ToolResult.success("\n".join(blocks), count=len(hits))


TOOL = tool(
    name="search",
    description="在内置知识库中检索与关键词最相关的文档片段（本地 mock，不联网）。",
)(search)
