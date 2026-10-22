"""分层上下文压缩：L0 截断（builder 内）→ L1 增量摘要 → L2 递归重写 → L3 硬裁剪。

分层原则：**先便宜后贵**，而且越贵（越丢信息）的层越晚触发。

* ``L0``（永远开启，在 builder 里）：tool 结果字符串截断，零成本、无损于"要不要发"；
* ``L1``（60% 水位）：把保留窗口之外的旧消息折叠成增量摘要，与已有 summary 合并；
* ``L2``（80% 水位）：递归重写——已有摘要 + 新折叠区间一起重写成更短的版本（≤600 字）；
* ``L3``（92% 水位）：不再调 LLM，直接硬裁剪（只留最近 ``keep_recent_turns`` 轮 + pinned），
  这是摘要失败/来不及摘要时的兜底。

三条硬约束
----------
1. **失败不抛异常**：摘要调用超时/报错/返回空，全部收敛成
   ``CompressionReport(ok=False, note=原因)``，由 builder 的硬预算继续兜底；
2. **可逆**：折叠前先把原文写入
   ``<data_dir>/artifacts/<session_id>/archived-<n>.jsonl``，归档成功才推进
   ``summary_upto``；路径记进 ``session.meta["archives"]``；
3. **pinned 不动**：``session.meta["pinned"]`` 的固定事实由 builder 放进 system prompt；
   第一条 user 消息在折叠后用原文回灌，保证"早期目标"永远在上下文里。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..llm.types import Message
from ..session.models import Session, now_iso
from ..session.store import SessionStore
from .budget import ContextBudget, estimate_messages_tokens, estimate_tokens
from .builder import ContextBuilder, recent_window_start

logger = logging.getLogger(__name__)

#: L1 摘要长度上限（字）
L1_SUMMARY_LIMIT = 400
#: L2 递归重写后的长度上限（字）
L2_SUMMARY_LIMIT = 600
#: 送进摘要 prompt 的单条消息截断长度（防止摘要请求自己又爆 context）
_TRANSCRIPT_ITEM_LIMIT = 600

_SECTION_NAMES = (
    "【用户目标】",
    "【已完成的动作与结论】",
    "【关键事实与ID】",
    "【未完成/待确认】",
    "【当前待办】",
)


# ---------------------------------------------------------------- 报告
@dataclass
class CompressionReport:
    """一次压缩的结构化结果（供 runtime / UI / 测试断言）。"""

    level: int = 0
    before_tokens: int = 0
    after_tokens: int = 0
    folded_messages: int = 0
    ok: bool = True
    note: str = ""
    archives: list[str] = field(default_factory=list)

    @property
    def ratio_before_after(self) -> tuple[float, float]:
        """便捷属性：压缩前后 token 数。"""
        return (float(self.before_tokens), float(self.after_tokens))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "folded_messages": self.folded_messages,
            "ok": self.ok,
            "note": self.note,
            "archives": list(self.archives),
        }


# ---------------------------------------------------------------- 工具函数
def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _summary_prompt(level: int, existing_summary: str) -> str:
    """固定模板：要求结构化五段，并硬性限制字数。"""
    limit = L2_SUMMARY_LIMIT if level >= 2 else L1_SUMMARY_LIMIT
    lines = [
        "你是会话上下文压缩器。请把下面的对话片段压缩成一份供后续继续任务使用的结构化摘要。",
        "",
        "必须包含以下五段（没有内容就写“（无）”，缺段也要保持可读）：",
        *_SECTION_NAMES,
        "",
        "硬性要求：",
        f"1. 全文不超过 {limit} 字；",
        "2. 数字、文件名、ID、URL、工具名必须原样保留，绝不编造；",
        "3. 用陈述句，不要复述对话原文，不要输出表格；",
        "4. 只输出摘要正文。",
    ]
    if existing_summary.strip():
        if level >= 2:
            lines += [
                "",
                "这是递归压缩：请把“已有摘要”和“新增对话”一起重写成更短的版本，"
                "合并重复信息，保留最关键的目标 / 结论 / 事实 / 未决事项。",
                "",
                "已有摘要：",
                existing_summary.strip(),
            ]
        else:
            lines += [
                "",
                "已有摘要（请与新对话合并，不要丢弃其中仍然有效的信息）：",
                existing_summary.strip(),
            ]
    return "\n".join(lines)


def _render_transcript(messages: list[Message]) -> str:
    """把待折叠消息渲染成摘要 prompt 的用户内容。"""
    lines = ["以下是需要压缩的对话（按时间顺序）："]
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            calls = "; ".join(
                f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
                for call in message.tool_calls
            )
            body = _clip(message.content, _TRANSCRIPT_ITEM_LIMIT)
            lines.append(f"[assistant] {body} 〔调用工具〕{calls}")
        elif message.role == "tool":
            name = message.name or message.tool_call_id
            lines.append(f"[tool:{name}] {_clip(message.content, _TRANSCRIPT_ITEM_LIMIT)}")
        else:
            lines.append(f"[{message.role}] {_clip(message.content, _TRANSCRIPT_ITEM_LIMIT)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Compressor
class Compressor:
    """按 token 水位分层压缩一个 Session。

    ``llm`` 可以是任何带 ``chat(messages=[...]) -> response`` 的客户端
    （生产是 :class:`agent.llm.client.LLMClient`，测试是 ``FakeLLM``）。
    """

    def __init__(
        self,
        config: Config,
        llm: Any | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.store = store
        self.builder = ContextBuilder(config, store)

    @property
    def budget(self) -> ContextBudget:
        """按当前 Config 现算（测试里可以随时改 context_budget_tokens）。"""
        return ContextBudget.from_config(self.config)

    # -- 对外入口 -------------------------------------------------------
    def maybe_compress(self, session: Session) -> CompressionReport:
        """按当前占用比例选择水位并执行；任何失败都以 ``ok=False`` 返回。"""
        with session.lock:
            return self._maybe_compress_locked(session)

    def _maybe_compress_locked(self, session: Session) -> CompressionReport:
        before = self.builder.measure(session)
        ratio = self.budget.used(before)
        level = self.budget.level(ratio)
        report = CompressionReport(level=level, before_tokens=before, after_tokens=before)

        if level <= 0:
            report.note = f"上下文占用 {ratio:.0%}，低于 L1 水位，仅 L0 截断"
            return report

        if level >= 3:
            self._hard_trim(session, report)
        else:
            self._fold(session, level, report)
            # 摘要后仍然贴着 L3 水位：追加一次硬裁剪（不再调模型）
            if report.ok and self.budget.used(self.builder.measure(session)) >= self.config.compress_l3_ratio:
                trimmed = self._hard_trim(session, report)
                if trimmed:
                    report.level = 3
                    self._append_note(report, "摘要后仍超 L3 水位，追加硬裁剪")

        report.after_tokens = self.builder.measure(session)
        if report.ok:
            logger.info(
                "compress session=%s level=%s tokens %s -> %s folded=%s",
                session.id,
                report.level,
                report.before_tokens,
                report.after_tokens,
                report.folded_messages,
            )
        else:
            logger.warning("compress session=%s failed: %s", session.id, report.note)
        return report

    # -- L1 / L2：摘要 ---------------------------------------------------
    def _fold(self, session: Session, level: int, report: CompressionReport) -> None:
        if self.llm is None:
            report.ok = False
            report.note = "未注入 LLM，L1/L2 摘要不可用（由 builder 硬预算兜底）"
            return

        start = max(0, min(session.summary_upto, len(session.messages)))
        visible_abs = [
            index
            for index in range(start, len(session.messages))
            if not session.messages[index].compressed
        ]
        region_rel = recent_window_start(
            [session.messages[index] for index in visible_abs],
            int(self.config.keep_recent_turns),
        )
        if region_rel <= 0 or region_rel >= len(visible_abs):
            # 不是失败：可见消息都落在保留窗口内，本来就没有可摘要的旧历史
            report.note = "没有需要摘要的旧消息（全部落在保留窗口内）"
            return

        cutoff = visible_abs[region_rel]
        region = [session.messages[index] for index in visible_abs[:region_rel]]
        # 第一条 user 消息视为 pinned：既不被裁掉，也**不进摘要**（由 builder 原样回灌），
        # 否则模型可能把用户的原始目标改写/丢掉。
        first_user_index = session.first_user_index()
        summary_region = [
            session.messages[index]
            for index in visible_abs[:region_rel]
            if index != first_user_index
        ]
        prompt = _summary_prompt(level, session.summary)
        transcript = _render_transcript(summary_region)

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ]
            )
        except Exception as exc:  # noqa: BLE001 - 摘要失败绝不允许冒泡到主循环
            report.ok = False
            report.note = f"摘要调用失败：{type(exc).__name__}: {exc}"
            return

        summary = str(getattr(response, "content", "") or "").strip()
        if not summary:
            report.ok = False
            report.note = "模型返回了空摘要，本次压缩放弃"
            return

        limit = L2_SUMMARY_LIMIT if level >= 2 else L1_SUMMARY_LIMIT
        if len(summary) > limit:
            summary = summary[:limit].rstrip() + "…"

        # 先归档、后推进游标：归档失败就整体放弃，保证可逆
        try:
            archive_path = self._archive(session, region)
        except OSError as exc:
            report.ok = False
            report.note = f"归档失败，放弃压缩：{exc}"
            return

        for message in region:
            message.compressed = True
        session.summary = summary
        session.summary_upto = cutoff
        self._record_archive(session, archive_path, len(region))
        report.archives.append(str(archive_path))
        session.touch()
        report.folded_messages = len(region)
        self._append_note(
            report, f"L{level} 摘要折叠 {len(region)} 条消息，归档 {archive_path.name}"
        )
        self._persist_summary(session, report)

    # -- L3：硬裁剪 ------------------------------------------------------
    def _hard_trim(self, session: Session, report: CompressionReport) -> int:
        """不调 LLM 的硬裁剪：先裁保留窗口之外，必要时在窗口内继续裁到 L3 水位以下。

        硬性保护：第一条 user 消息（pinned）与**最后一条 user 消息**永远不裁。
        裁掉的内容全部先归档，因此信息仍然可逆。
        """
        start = max(0, min(session.summary_upto, len(session.messages)))
        visible_abs = [
            index
            for index in range(start, len(session.messages))
            if not session.messages[index].compressed
        ]
        if not visible_abs:
            report.note = "没有可裁剪的可见消息"
            return 0

        region_rel = recent_window_start(
            [session.messages[index] for index in visible_abs],
            int(self.config.keep_recent_turns),
        )
        pinned_first = session.first_user_index()
        last_user = max(
            (index for index in visible_abs if session.messages[index].role == "user"),
            default=-1,
        )
        protected = {index for index in (pinned_first, last_user) if index >= 0}

        # 1) 保留窗口之外的旧消息：必裁
        must_trim = [
            index for index in visible_abs[:region_rel] if index not in protected
        ]
        # 2) 窗口内更旧的消息：只有窗口自己就超预算时才"整轮"按需裁剪；
        #    最后一轮（最后一条 user 及其后的 tool 结果）是正在进行的任务，绝不裁。
        window_abs = visible_abs[region_rel:]
        groups: list[list[int]] = []
        current: list[int] = []
        for index in window_abs:
            if session.messages[index].role == "user" and current:
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
        pool_groups = [
            group
            for group in groups[:-1]
            if any(index not in protected for index in group)
        ]

        trim = set(must_trim)
        limit = self.budget.limit(self.config.compress_l3_ratio)
        base_tokens = estimate_tokens(self.builder.render_system(session))
        remaining = [
            session.messages[index] for index in visible_abs if index not in trim
        ]
        while (
            pool_groups
            and base_tokens + estimate_messages_tokens(remaining) > limit
        ):
            for index in pool_groups.pop(0):
                if index not in protected:
                    trim.add(index)
            remaining = [
                session.messages[index] for index in visible_abs if index not in trim
            ]

        if not trim:
            report.note = "可裁剪消息都被 pinned / 保留窗口保护，硬裁剪无需执行"
            return 0

        region = [session.messages[index] for index in visible_abs if index in trim]
        try:
            archive_path = self._archive(session, region)
        except OSError as exc:
            report.ok = False
            report.note = f"归档失败，放弃硬裁剪：{exc}"
            return 0

        for index in trim:
            session.messages[index].compressed = True
        self._record_archive(session, archive_path, len(region))
        report.archives.append(str(archive_path))
        session.meta["l3_trimmed"] = int(session.meta.get("l3_trimmed") or 0) + len(region)
        session.touch()
        if self.store is not None:
            try:
                self.store.append_note(
                    session,
                    "compression",
                    {
                        "level": 3,
                        "folded_messages": len(region),
                        "archive": str(archive_path),
                        "ts": now_iso(),
                    },
                )
            except OSError as exc:  # pragma: no cover - 持久化失败不影响内存状态
                logger.warning("append compression note failed: %s", exc)
        report.folded_messages += len(region)
        self._append_note(report, f"L3 硬裁剪 {len(region)} 条消息，归档 {archive_path.name}")
        return len(region)

    # -- 归档 -------------------------------------------------------------
    def _archive(self, session: Session, messages: list[Message]) -> Path:
        """把原文写入 ``<data_dir>/artifacts/<session_id>/archived-<n>.jsonl``。"""
        directory = Path(self.config.data_dir) / "artifacts" / session.id
        directory.mkdir(parents=True, exist_ok=True)
        index = 1
        while (directory / f"archived-{index}.jsonl").exists():
            index += 1
        path = directory / f"archived-{index}.jsonl"
        payload = "\n".join(
            json.dumps(message.to_dict(), ensure_ascii=False) for message in messages
        )
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(temp_path, path)
        return path

    @staticmethod
    def _record_archive(
        session: Session, archive_path: Path, message_count: int
    ) -> None:
        """把归档路径记进 session.meta（持久化由 save_summary / append_note 负责）。"""
        archives = session.meta.get("archives")
        if not isinstance(archives, list):
            archives = []
            session.meta["archives"] = archives
        archives.append(str(archive_path))
        session.meta["last_archive"] = str(archive_path)
        session.meta["last_archive_messages"] = message_count

    def _persist_summary(self, session: Session, report: CompressionReport) -> None:
        if self.store is None:
            return
        try:
            self.store.save_summary(session)
            self.store.append_note(
                session,
                "compression",
                {
                    "level": report.level,
                    "folded_messages": report.folded_messages,
                    "summary_upto": session.summary_upto,
                    "before_tokens": report.before_tokens,
                    "after_tokens": report.after_tokens,
                    "ts": now_iso(),
                },
            )
        except OSError as exc:  # pragma: no cover - 持久化失败不影响内存状态
            logger.warning("persist compression state failed: %s", exc)

    @staticmethod
    def _append_note(report: CompressionReport, note: str) -> None:
        report.note = f"{report.note}；{note}" if report.note else note
