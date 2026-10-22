"""Session 的 JSONL 持久化（append-only 事件流 + 首行 meta 快照）。

文件布局
--------
``<data_dir>/sessions/<session_id>.jsonl``::

    {"type":"meta","session":{...}}
    {"type":"message","ts":"...","message":{...}}
    {"type":"todos","ts":"...","todos":[...]}
    {"type":"summary","ts":"...","summary":"...","summary_upto":12}
    {"type":"note","ts":"...","kind":"tool_trace","payload":{...}}

为什么用 JSONL 而不是一个 JSON 大文件
------------------------------------
* 追加写只需一次 open/append，进程被 kill 也最多丢最后一行，不会整个会话损坏；
* 人肉 ``type`` 字段排查、grep、回滚都方便；
* 坏行可以逐行跳过（``corrupt_lines``），不会因为一行脏数据让整个会话打不开。

首行 meta 会随会话状态变化被重写（temp + ``os.replace`` 原子替换），
这样 :meth:`SessionStore.list_sessions` 只读首行就能拿到最新的 updated_at。
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm.types import Message
from .models import Session, TodoItem, new_session_id, now_iso

#: 只允许这类 id，防止 ``../`` 之类把文件写到 sessions 目录之外
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

#: meta 里保留的 note 数量上限（避免 meta 无限膨胀）
_MAX_KEPT_NOTES = 100


class SessionStore:
    """会话存储。每个 :class:`SessionStore` 实例管理一个数据根目录。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        #: 进程内 session 锁表（按 id 复用同一把 RLock）
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # -- 路径与锁 -----------------------------------------------------
    def path_for(self, session_id: str) -> Path:
        """session_id -> 文件路径（同时做非法 id 校验）。"""
        session_id = str(session_id)
        if not _SAFE_SESSION_ID.match(session_id):
            raise ValueError(f"非法 session_id: {session_id!r}")
        return self.sessions_dir / f"{session_id}.jsonl"

    def lock(self, session_id: str) -> threading.RLock:
        """同一 session id 在进程内返回同一把锁。"""
        key = str(session_id)
        with self._locks_guard:
            existing = self._locks.get(key)
            if existing is None:
                existing = threading.RLock()
                self._locks[key] = existing
            return existing

    # -- 创建 / 查询 ---------------------------------------------------
    def create(self, title: str = "", session_id: str | None = None) -> Session:
        """新建会话并立刻写首行 meta。未指定 id 时生成 ``s<date>-<rand>``。"""
        sid = (session_id or "").strip()
        if sid:
            path = self.path_for(sid)
            with self.lock(sid):
                if path.exists():
                    raise ValueError(f"session 已存在: {sid}")
                return self._create_at(sid, title)
        for _ in range(16):
            sid = new_session_id()
            if not self.path_for(sid).exists():
                return self._create_at(sid, title)
        raise RuntimeError("无法生成不重复的 session_id")  # pragma: no cover

    def _create_at(self, session_id: str, title: str) -> Session:
        path = self.path_for(session_id)
        now = now_iso()
        session = Session(
            id=session_id,
            created_at=now,
            updated_at=now,
            title=title,
            state="idle",
            turn_count=0,
        )
        with self.lock(session_id):
            self._append_line(path, {"type": "meta", "session": self._meta_payload(session)})
        return session

    def exists(self, session_id: str) -> bool:
        try:
            return self.path_for(session_id).is_file()
        except ValueError:
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """只读每个文件首行 meta，按 updated_at 倒序返回摘要信息。"""
        rows: list[dict[str, Any]] = []
        for path in sorted(self.sessions_dir.glob("*.jsonl")):
            row: dict[str, Any] = {
                "id": path.stem,
                "title": "",
                "created_at": "",
                "updated_at": "",
                "turns": 0,
                "state": "idle",
            }
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    first_line = handle.readline()
                info = json.loads(first_line).get("session") or {}
                row.update(
                    {
                        "id": str(info.get("id") or path.stem),
                        "title": str(info.get("title") or ""),
                        "created_at": str(info.get("created_at") or ""),
                        "updated_at": str(info.get("updated_at") or ""),
                        "turns": int(info.get("turn_count") or 0),
                        "state": str(info.get("state") or "idle"),
                    }
                )
            except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
                pass
            if not row["updated_at"]:
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
                    row["updated_at"] = mtime.isoformat(timespec="seconds")
                except OSError:  # pragma: no cover - 文件刚被删
                    pass
            rows.append(row)
        rows.sort(key=lambda item: (item["updated_at"], item["id"]), reverse=True)
        return rows

    # -- 加载（replay） -------------------------------------------------
    def load(self, session_id: str) -> Session:
        """逐行 replay 事件流。坏行跳过并记入 ``session.meta["corrupt_lines"]``。"""
        path = self.path_for(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"session 不存在: {session_id}")
        with self.lock(session_id):
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        session = Session(id=str(session_id))
        corrupt = 0
        meta_seen = False
        last_ts = ""

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                corrupt += 1
                continue
            if not isinstance(row, dict):
                corrupt += 1
                continue
            kind = row.get("type")
            try:
                if kind == "meta":
                    self._apply_meta(session, row.get("session") or {})
                    meta_seen = True
                elif kind == "message":
                    session.messages.append(Message.from_dict(row["message"]))
                    last_ts = str(row.get("ts") or last_ts)
                elif kind == "todos":
                    session.todos = [
                        TodoItem.from_dict(item)
                        for item in row.get("todos") or []
                        if isinstance(item, dict)
                    ]
                    last_ts = str(row.get("ts") or last_ts)
                elif kind == "summary":
                    session.summary = str(row.get("summary") or "")
                    session.summary_upto = int(row.get("summary_upto") or 0)
                    last_ts = str(row.get("ts") or last_ts)
                elif kind == "note":
                    notes = session.meta.setdefault("notes", [])
                    if isinstance(notes, list):
                        notes.append(
                            {
                                "ts": row.get("ts") or "",
                                "kind": str(row.get("kind") or ""),
                                "payload": row.get("payload") or {},
                            }
                        )
                        if len(notes) > _MAX_KEPT_NOTES:
                            del notes[: len(notes) - _MAX_KEPT_NOTES]
                else:
                    # 未知 type 视为脏行（向前兼容时也可以静默忽略，这里选择暴露出来）
                    corrupt += 1
            except (KeyError, TypeError, ValueError, AttributeError):
                corrupt += 1

        if not meta_seen:
            # 没有 meta 行（例如被外部工具截断）时尽力恢复基本字段
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
                session.created_at = session.created_at or mtime.isoformat(timespec="seconds")
                session.updated_at = session.updated_at or session.created_at
            except OSError:  # pragma: no cover
                pass

        # turn_count 以 meta 为准，但用消息数兜底（防止 meta 因崩溃落后于事件流）
        user_messages = sum(1 for message in session.messages if message.role == "user")
        session.turn_count = max(session.turn_count, user_messages)
        session.updated_at = session.updated_at or last_ts or now_iso()
        session.summary_upto = max(0, min(session.summary_upto, len(session.messages)))
        session.meta["corrupt_lines"] = corrupt
        return session

    @staticmethod
    def _apply_meta(session: Session, info: dict[str, Any]) -> None:
        session.id = str(info.get("id") or session.id)
        session.title = str(info.get("title") or session.title)
        session.created_at = str(info.get("created_at") or session.created_at)
        session.updated_at = str(info.get("updated_at") or session.updated_at)
        session.state = str(info.get("state") or session.state)
        session.turn_count = int(info.get("turn_count") or session.turn_count)

    # -- 写入 -----------------------------------------------------------
    def append_message(self, session: Session, message: Message) -> None:
        """追加一条消息并刷新首行 meta（updated_at / turn_count）。

        兼容两种调用姿势：
        * 只调 ``store.append_message(...)`` —— 这里负责 ``session.append`` + 落盘；
        * 先 ``session.append(...)`` 再 ``store.append_message(...)``（runtime/loop 的写法）
          —— 检测到消息已经是 session 最后一条时只落盘，不会重复追加。
        """
        with self.lock(session.id):
            self._ensure_file(session)
            if session.messages and session.messages[-1] is message:
                session.touch()
            else:
                session.append(message)
            self._append_line(
                self.path_for(session.id),
                {"type": "message", "ts": now_iso(), "message": message.to_dict()},
            )
            self._flush_meta(session)

    def save_todos(self, session: Session) -> None:
        """写一份完整待办快照（重放时后写的覆盖先写的）。"""
        with self.lock(session.id):
            self._ensure_file(session)
            self._append_line(
                self.path_for(session.id),
                {
                    "type": "todos",
                    "ts": now_iso(),
                    "todos": [todo.to_dict() for todo in session.todos],
                },
            )
            self._flush_meta(session)

    def save_summary(self, session: Session) -> None:
        """写摘要行（含 summary_upto 游标）。"""
        with self.lock(session.id):
            self._ensure_file(session)
            self._append_line(
                self.path_for(session.id),
                {
                    "type": "summary",
                    "ts": now_iso(),
                    "summary": session.summary,
                    "summary_upto": session.summary_upto,
                },
            )
            self._flush_meta(session)

    def append_note(self, session: Session, kind: str, payload: dict[str, Any]) -> None:
        """写一条观测性 note（如压缩报告、工具轨迹）。"""
        with self.lock(session.id):
            self._ensure_file(session)
            session.touch()
            self._append_line(
                self.path_for(session.id),
                {"type": "note", "ts": now_iso(), "kind": kind, "payload": payload},
            )
            self._flush_meta(session)

    def delete(self, session_id: str) -> None:
        """删除会话文件（不存在则忽略）。"""
        path = self.path_for(session_id)
        with self.lock(session_id):
            try:
                path.unlink()
            except FileNotFoundError:
                return

    # -- 内部工具 -------------------------------------------------------
    @staticmethod
    def _meta_payload(session: Session) -> dict[str, Any]:
        return {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "state": session.state,
            "turn_count": session.turn_count,
        }

    def _ensure_file(self, session: Session) -> None:
        """session 文件不存在时补写首行 meta（容忍直接 new Session 的调用方）。"""
        path = self.path_for(session.id)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        now = now_iso()
        session.created_at = session.created_at or now
        session.updated_at = session.updated_at or now
        self._append_line(path, {"type": "meta", "session": self._meta_payload(session)})

    def _flush_meta(self, session: Session) -> None:
        """把首行 meta 刷新为最新状态（写临时文件后原子替换）。"""
        path = self.path_for(session.id)
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:  # pragma: no cover - 读失败就让事件流继续追加
            return
        meta_line = json.dumps(
            {"type": "meta", "session": self._meta_payload(session)}, ensure_ascii=False
        )
        if lines and self._line_type(lines[0]) == "meta":
            lines[0] = meta_line
        else:
            lines.insert(0, meta_line)
        temp_path = path.with_name(path.name + ".tmp")
        try:
            temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(temp_path, path)
        except OSError:  # pragma: no cover - 极端 IO 失败不阻塞主循环
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _line_type(line: str) -> str:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return ""
        return str(row.get("type") or "") if isinstance(row, dict) else ""

    @staticmethod
    def _append_line(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
