"""上下文组装：把 Session 状态拼成一次 LLM 请求的 messages。

组装顺序（与 CONTRACT.md 2 节一致）
----------------------------------
1. 一条 system 消息：角色设定、当前本地时间、工具清单、会话状态（轮次 + 待办）、
   pinned 事实、**可选**的会话原始目标、较早对话的摘要、调用方追加的 extra_system；
2. 最近的若干轮消息（``keep_recent_turns``，从 user 消息处切轮，避免切碎一轮）；
3. 丢弃 ``compressed=True`` 的消息；
4. **修复 tool_call / tool_result 配对**：assistant 的每个 ``tool_call.id`` 都必须能在
   紧随其后的 ``role="tool"`` 消息里找到，否则丢弃该 assistant 消息；孤立的 tool
   结果同样丢弃。宁可少给模型一条消息，也不能让厂商直接 400；
5. L0：超长 tool 结果按 60% + 20% 截断（最便宜的一层压缩）；
6. 硬预算：超过 ``context_budget_tokens * compress_l3_ratio`` 时从最旧的非 system
   消息开始丢，**永远保留最后一条 user 消息**。

关键取舍
--------
* 本模块**不修改** Session（截断会 ``dataclasses.replace`` 出新对象），所以
  ``build()`` 可以随时调用（观测、token 估算、测试），不会产生副作用；
* tool 结果截断放在"选窗口"之后、硬预算之前：先只截真正要发出去的消息，
  不浪费在旧消息上；
* 硬预算兜底是压缩失败（L1/L2 摘不出来）时的最后一道保险。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..config import Config
from ..llm.types import Message
from ..session.models import Session, now_iso
from ..session.store import SessionStore
from .budget import ContextBudget, estimate_messages_tokens, estimate_tokens

#: system prompt 里 pinned 原始目标的长度上限（防止第一条 user 又臭又长）
_PINNED_GOAL_LIMIT = 400

_SYSTEM_ROLE = (
    "你是 mini-agent，一个从零实现的最小可用 Agent。你可以调用工具来完成任务。\n"
    "工作原则：\n"
    "- 需要外部信息或计算时优先调用工具，不要凭空编造；\n"
    "- 工具失败时先读错误信息、修正参数再试，不要重复同一个失败调用；\n"
    "- 回答简洁，结论先行。"
)


# ---------------------------------------------------------------- 纯函数工具
def truncate_tool_result(content: str, max_chars: int) -> str:
    """L0 压缩：保留前 60% + 后 20%，中间用显式标记代替。"""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    head = max(1, int(max_chars * 0.6))
    tail = max(0, int(max_chars * 0.2))
    omitted = len(content) - head - tail
    if omitted <= 0:  # pragma: no cover - max_chars 极小时才会发生
        return content
    marker = f"\n...[已截断 {omitted} 字符]...\n"
    return content[:head] + marker + (content[-tail:] if tail else "")


def recent_window_start(messages: list[Message], keep_recent_turns: int) -> int:
    """返回"最近 keep_recent_turns 轮"的起始下标（按 user 消息划分轮次）。

    返回 ``0`` 表示所有消息都落在保留窗口内（没有可折叠 / 可裁剪的前缀）。
    ``keep_recent_turns <= 0`` 时至少保留最后一轮，避免把当前问题也裁掉。
    """
    keep = max(1, int(keep_recent_turns))
    user_positions = [
        index for index, message in enumerate(messages) if message.role == "user"
    ]
    if not user_positions or len(user_positions) <= keep:
        return 0
    return user_positions[-keep]


def repair_tool_pairs(messages: list[Message]) -> list[Message]:
    """让 assistant(tool_calls) 与随后的 tool 结果严格配对。

    OpenAI-compatible 协议要求 ``role="tool"`` 消息紧跟在发起调用的 assistant 之后，
    且 ``tool_call_id`` 必须被引用过。常见的 400 来源就是：窗口裁剪/压缩把 tool 结果
    删了但留着 assistant，或者反过来。这里的策略是 **宁缺勿错配**：
    配对不完整就整条 assistant（含它的 tool 结果）丢弃。
    """
    repaired: list[Message] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        if message.role == "assistant" and message.tool_calls:
            needed_ids = [call.id for call in message.tool_calls]
            needed = set(needed_ids)
            scan = index + 1
            tool_messages: list[Message] = []
            while scan < total and messages[scan].role == "tool":
                tool_messages.append(messages[scan])
                scan += 1
            found = {item.tool_call_id for item in tool_messages}
            if needed <= found:
                repaired.append(message)
                repaired.extend(
                    item for item in tool_messages if item.tool_call_id in needed
                )
            # 否则整条 assistant 与它的 tool 结果一起丢弃
            index = scan
        elif message.role == "tool":
            # 孤立 tool 结果（前面的 assistant 被裁掉/压缩掉了）：丢弃
            index += 1
        else:
            repaired.append(message)
            index += 1
    return repaired


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


# ---------------------------------------------------------------- ContextBuilder
class ContextBuilder:
    """把一个 Session 组装成 OpenAI 格式的 messages 列表。"""

    def __init__(self, config: Config, store: SessionStore | None = None) -> None:
        self.config = config
        self.store = store

    @property
    def budget(self) -> ContextBudget:
        """按当前 Config 现算（测试里可以随时改 context_budget_tokens）。"""
        return ContextBudget.from_config(self.config)

    # -- 对外 -----------------------------------------------------------
    def build(
        self,
        session: Session,
        tools_description: str = "",
        extra_system: str = "",
    ) -> list[dict[str, Any]]:
        """返回 ``[system, ...messages]`` 的 OpenAI 格式列表。"""
        selected = self.select_messages(session)
        system_text = self.render_system(
            session, tools_description=tools_description, extra_system=extra_system
        )
        selected = self._apply_hard_budget(system_text, selected)
        backfill = self.config.profile.backfill_thinking
        return [{"role": "system", "content": system_text}] + [
            message.to_openai(backfill_thinking=backfill) for message in selected
        ]

    def measure(self, session: Session, tools_description: str = "") -> int:
        """估算"如果现在发请求"的 token 数（system + 全部未压缩可见消息）。

        压缩水位用它来判定，因此这里故意把**整个可见历史**都算进去，
        而不是只看 ``keep_recent_turns`` 窗口——否则旧消息堆积永远触发不了压缩。
        """
        system_text = self.render_system(session, tools_description=tools_description)
        visible = [
            message for message in session.visible_messages() if not message.compressed
        ]
        return estimate_tokens(system_text) + estimate_messages_tokens(visible)

    # -- 消息选择 -------------------------------------------------------
    def select_messages(self, session: Session) -> list[Message]:
        """取最近若干轮 -> 丢弃 compressed -> L0 截断 -> 修复工具配对。"""
        visible = [
            message for message in session.visible_messages() if not message.compressed
        ]
        start = recent_window_start(visible, int(self.config.keep_recent_turns))
        windowed = visible[start:]
        truncated = [self._truncate_tool_result(message) for message in windowed]
        return repair_tool_pairs(truncated)

    def _truncate_tool_result(self, message: Message) -> Message:
        if message.role != "tool":
            return message
        limit = int(self.config.tool_result_max_chars)
        if limit <= 0 or len(message.content) <= limit:
            return message
        return replace(message, content=truncate_tool_result(message.content, limit))

    # -- system prompt ---------------------------------------------------
    def render_system(
        self,
        session: Session,
        tools_description: str = "",
        extra_system: str = "",
    ) -> str:
        parts: list[str] = [
            _SYSTEM_ROLE,
            f"## 当前本地时间\n{now_iso()}",
            "## 可用工具\n" + (tools_description.strip() or "（本轮没有可用工具）"),
            session.render_state_block(),
        ]

        pinned = session.pinned_facts()
        if pinned:
            parts.append(
                "## 固定事实（pinned，永不压缩）\n"
                + "\n".join(f"- {item}" for item in pinned)
            )

        # 第一条 user 消息是 pinned：一旦它不在本次窗口里（被压缩/被裁剪），
        # 就以原文回灌到 system，保证"早期目标"永远答得上来。
        first_user = self._first_user_message(session)
        if first_user is not None and self._first_user_outside_window(session):
            parts.append(
                "## 会话原始目标（pinned，第一条用户消息）\n"
                + _clip(first_user.content, _PINNED_GOAL_LIMIT)
            )

        if session.summary.strip():
            parts.append("## 较早对话的摘要\n" + session.summary.strip())

        if extra_system.strip():
            parts.append(extra_system.strip())

        return "\n\n".join(parts)

    @staticmethod
    def _first_user_message(session: Session) -> Message | None:
        for message in session.messages:
            if message.role == "user":
                return message
        return None

    def _first_user_outside_window(self, session: Session) -> bool:
        """第一条 user 消息是否已不在本次可见窗口内（被压缩/被裁剪）。"""
        first_index = session.first_user_index()
        if first_index < 0:
            return False
        if first_index < session.summary_upto:
            return True
        if session.messages[first_index].compressed:
            return True
        visible = session.visible_messages()
        offsets = [
            offset
            for offset, message in enumerate(visible)
            if not message.compressed
        ]
        user_offsets = [
            offset for offset in offsets if visible[offset].role == "user"
        ]
        keep = max(1, int(self.config.keep_recent_turns))
        if len(user_offsets) <= keep:
            return False
        window_start_abs = session.summary_upto + user_offsets[-keep]
        return first_index < window_start_abs

    # -- 硬预算 ---------------------------------------------------------
    def _apply_hard_budget(self, system_text: str, messages: list[Message]) -> list[Message]:
        """最后一道保险：超 L3 水位就从最旧的非 system 消息开始丢。

        唯一硬约束：**最后一条 user 消息必须保留**（否则模型不知道要回答什么）。
        """
        limit = self.budget.limit(self.config.compress_l3_ratio)
        if limit <= 0:
            return messages
        working = list(messages)
        while working:
            used = estimate_tokens(system_text) + estimate_messages_tokens(working)
            if used <= limit:
                break
            last_user = max(
                (index for index, message in enumerate(working) if message.role == "user"),
                default=-1,
            )
            drop_at = next(
                (index for index in range(len(working)) if index != last_user), None
            )
            if drop_at is None:
                break
            del working[drop_at]
            # 丢掉 assistant 后可能留下孤立的 tool 结果，再修一次配对
            working = repair_tool_pairs(working)
        return working
