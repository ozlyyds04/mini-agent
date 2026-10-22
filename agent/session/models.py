"""会话层数据模型：:class:`Session` / :class:`TodoItem`，以及 JSONL 友好的序列化。

设计要点
--------
1. **状态全部挂在 Session 实例上**：messages / todos / summary / summary_upto 都是
   实例字段，本模块没有任何可变的模块级全局状态。多个 session 并存时天然隔离，
   todo 这类"看起来像全局"的工具只操作 ``ctx.session``，因此不会串会话。
2. **滚动摘要游标**：``summary`` 覆盖 ``messages[:summary_upto]``，
   :meth:`Session.visible_messages` 只返回其后的消息。压缩因此只推进一个游标，
   原文仍完整保留在 JSONL 里，随时可回滚 / 审计。
3. **时间统一为带时区的 ISO 字符串**（``2026-10-22T11:33:37+08:00``）。缺少时区的
   时间在跨时区部署、夏令时切换、JSONL 回放排序时都会出错。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..llm.types import Message


def now_iso() -> str:
    """当前本地时间，带时区偏移的 ISO 字符串（秒级精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_session_id(prefix: str = "s") -> str:
    """生成形如 ``s20261022-a1b2c3`` 的会话 id（日期 + 随机后缀）。"""
    day = datetime.now().astimezone().strftime("%Y%m%d")
    return f"{prefix}{day}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------- TodoItem
@dataclass
class TodoItem:
    """一条待办。id 在单个 session 内单调递增，跨 session 不共享。"""

    id: int
    content: str
    done: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "done": self.done,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoItem":
        return cls(
            id=int(data.get("id") or 0),
            content=str(data.get("content") or ""),
            done=bool(data.get("done")),
            created_at=str(data.get("created_at") or ""),
        )

    def render_line(self) -> str:
        """``#1 [ ] 内容`` / ``#2 [x] 内容``。"""
        mark = "x" if self.done else " "
        return f"#{self.id} [{mark}] {self.content}"


# ---------------------------------------------------------------- Session
@dataclass
class Session:
    """一个会话的全部状态（消息、待办、摘要、元信息、锁）。"""

    id: str
    created_at: str = ""
    updated_at: str = ""
    title: str = ""
    messages: list[Message] = field(default_factory=list)
    todos: list[TodoItem] = field(default_factory=list)
    #: 滚动摘要（压缩产物），覆盖 messages[:summary_upto]
    summary: str = ""
    #: messages[:summary_upto] 已被摘要覆盖（或已归档），其后的消息才是"可见"的
    summary_upto: int = 0
    turn_count: int = 0
    #: idle | busy | waiting_tool | paused
    state: str = "idle"
    meta: dict[str, Any] = field(default_factory=dict)
    #: 同一 session 的进程内互斥锁；不参与序列化
    lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    # -- 消息 ---------------------------------------------------------
    def append(self, message: Message) -> Message:
        """追加消息并刷新 updated_at；user 消息计入 turn_count。"""
        self.messages.append(message)
        if message.role == "user":
            self.turn_count += 1
        self.touch()
        return message

    def touch(self) -> None:
        self.updated_at = now_iso()

    def visible_messages(self) -> list[Message]:
        """未被摘要覆盖的消息（``messages[summary_upto:]``）。"""
        start = max(0, min(self.summary_upto, len(self.messages)))
        return self.messages[start:]

    # -- 待办 ---------------------------------------------------------
    def next_todo_id(self) -> int:
        return max((todo.id for todo in self.todos), default=0) + 1

    def add_todo(self, content: str) -> TodoItem:
        todo = TodoItem(
            id=self.next_todo_id(),
            content=content.strip(),
            done=False,
            created_at=now_iso(),
        )
        self.todos.append(todo)
        self.touch()
        return todo

    def done_todo(self, todo_id: int) -> TodoItem | None:
        for todo in self.todos:
            if todo.id == int(todo_id):
                todo.done = True
                self.touch()
                return todo
        return None

    def clear_todos(self, only_done: bool = False) -> int:
        """清空待办；``only_done=True`` 时只删已完成的。返回删除数量。"""
        before = len(self.todos)
        if only_done:
            self.todos = [todo for todo in self.todos if not todo.done]
        else:
            self.todos = []
        removed = before - len(self.todos)
        if removed:
            self.touch()
        return removed

    def pending_todos(self) -> list[TodoItem]:
        return [todo for todo in self.todos if not todo.done]

    def render_todos(self) -> str:
        """给 system prompt 用的待办清单；空时返回 ``（无）``。"""
        if not self.todos:
            return "（无）"
        return "\n".join(todo.render_line() for todo in self.todos)

    def render_state_block(self) -> str:
        """结构化状态（轮次 + 状态 + 待办），供 system prompt 使用。"""
        todo_lines = "\n".join(f"  {line}" for line in self.render_todos().splitlines())
        return (
            "### 会话状态\n"
            f"- 会话 ID: {self.id}\n"
            f"- 已完成轮次: {self.turn_count}\n"
            f"- 状态: {self.state}\n"
            "- 待办:\n"
            f"{todo_lines}"
        )

    # -- pinned 相关（压缩时要用） -----------------------------------
    def first_user_index(self) -> int:
        """第一条 user 消息的绝对下标；没有则 -1。它被视为 pinned。"""
        for index, message in enumerate(self.messages):
            if message.role == "user":
                return index
        return -1

    def pinned_facts(self) -> list[str]:
        """``meta["pinned"]`` 里的固定事实：永不进摘要、永不被裁剪。"""
        raw = self.meta.get("pinned")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    # -- 序列化 -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "messages": [message.to_dict() for message in self.messages],
            "todos": [todo.to_dict() for todo in self.todos],
            "summary": self.summary,
            "summary_upto": self.summary_upto,
            "turn_count": self.turn_count,
            "state": self.state,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        session = cls(
            id=str(data.get("id") or new_session_id()),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            title=str(data.get("title") or ""),
            summary=str(data.get("summary") or ""),
            summary_upto=int(data.get("summary_upto") or 0),
            turn_count=int(data.get("turn_count") or 0),
            state=str(data.get("state") or "idle"),
            meta=dict(data.get("meta") or {}),
        )
        session.messages = [
            Message.from_dict(item)
            for item in data.get("messages") or []
            if isinstance(item, dict)
        ]
        session.todos = [
            TodoItem.from_dict(item)
            for item in data.get("todos") or []
            if isinstance(item, dict)
        ]
        session.summary_upto = max(0, min(session.summary_upto, len(session.messages)))
        return session
