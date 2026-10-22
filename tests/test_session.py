"""会话层单测：隔离、JSONL 回放、坏行容错、排序、todo 语义。

全部离线运行（tempfile 临时目录，不联网、不读 .env）。
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from agent.llm.types import Message, ToolCall
from agent.session.models import now_iso
from agent.session.store import SessionStore


class SessionStoreTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = SessionStore(self.root)


class SessionIsolationTest(SessionStoreTestBase):
    def test_two_sessions_do_not_share_any_state(self) -> None:
        """CONTRACT 用例 1：todo / messages / summary 完全隔离。"""
        session_a = self.store.create("会话 A")
        session_b = self.store.create("会话 B")
        self.assertNotEqual(session_a.id, session_b.id)

        session_a.add_todo("给 A 买牛奶")
        session_b.add_todo("给 B 写周报")
        self.store.save_todos(session_a)
        self.store.save_todos(session_b)

        self.store.append_message(session_a, Message.user("A 的问题"))
        self.store.append_message(session_b, Message.user("B 的问题"))

        session_a.summary = "A 的摘要"
        session_a.summary_upto = 1
        self.store.save_summary(session_a)

        # 内存对象互不影响
        self.assertEqual([todo.content for todo in session_a.todos], ["给 A 买牛奶"])
        self.assertEqual([todo.content for todo in session_b.todos], ["给 B 写周报"])
        self.assertEqual(len(session_a.messages), 1)
        self.assertEqual(len(session_b.messages), 1)
        self.assertEqual(session_b.summary, "")
        self.assertEqual(session_b.summary_upto, 0)

        # 从文件回放后仍然隔离
        loaded_a = self.store.load(session_a.id)
        loaded_b = self.store.load(session_b.id)
        self.assertEqual([todo.content for todo in loaded_a.todos], ["给 A 买牛奶"])
        self.assertEqual([todo.content for todo in loaded_b.todos], ["给 B 写周报"])
        self.assertEqual(loaded_a.messages[0].content, "A 的问题")
        self.assertEqual(loaded_b.messages[0].content, "B 的问题")
        self.assertEqual(loaded_a.summary, "A 的摘要")
        self.assertEqual(loaded_a.summary_upto, 1)
        self.assertEqual(loaded_b.summary, "")
        self.assertEqual(loaded_b.summary_upto, 0)

        # 清掉 A 的待办不影响 B 的持久化状态
        loaded_a.clear_todos()
        self.store.save_todos(loaded_a)
        self.assertEqual(
            [todo.content for todo in self.store.load(session_b.id).todos],
            ["给 B 写周报"],
        )


class SessionPersistenceTest(SessionStoreTestBase):
    def test_create_append_save_todos_load_roundtrip(self) -> None:
        """CONTRACT 用例 2（前半）：创建 -> 追加 -> save_todos -> load 后状态一致。"""
        session = self.store.create("回放测试")
        self.store.append_message(session, Message.user("算 1+1"))
        call = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        self.store.append_message(session, Message.assistant("", tool_calls=[call]))
        self.store.append_message(
            session, Message.tool_result("call_1", "1+1 = 2", "calculator")
        )

        session.add_todo("确认结果")
        self.store.save_todos(session)
        session.summary = "用户要求算 1+1，结果为 2"
        session.summary_upto = 2
        self.store.save_summary(session)

        loaded = self.store.load(session.id)
        self.assertEqual(loaded.title, "回放测试")
        self.assertEqual(loaded.created_at, session.created_at)
        self.assertEqual(loaded.turn_count, 1)
        self.assertEqual(len(loaded.messages), 3)
        self.assertEqual(loaded.messages[1].tool_calls[0].id, "call_1")
        self.assertEqual(
            loaded.messages[1].tool_calls[0].arguments, {"expression": "1+1"}
        )
        self.assertEqual(loaded.messages[2].tool_call_id, "call_1")
        self.assertEqual(loaded.messages[2].name, "calculator")
        self.assertEqual([todo.content for todo in loaded.todos], ["确认结果"])
        self.assertEqual(loaded.summary, "用户要求算 1+1，结果为 2")
        self.assertEqual(loaded.summary_upto, 2)
        self.assertEqual(
            [message.content for message in loaded.visible_messages()], ["1+1 = 2"]
        )
        self.assertEqual(loaded.meta["corrupt_lines"], 0)

    def test_append_message_after_session_append_does_not_duplicate(self) -> None:
        """兼容 runtime/loop 的 ``session.append(...) + store.append_message(...)`` 写法。"""
        session = self.store.create("双写兼容")
        message = Message.user("只应该有一条")
        session.append(message)
        self.store.append_message(session, message)

        self.assertEqual(len(session.messages), 1)
        loaded = self.store.load(session.id)
        self.assertEqual(len(loaded.messages), 1)
        self.assertEqual(loaded.turn_count, 1)

    def test_corrupt_lines_are_skipped_and_counted(self) -> None:
        """CONTRACT 用例 2（后半）：坏行跳过，corrupt_lines > 0。"""
        session = self.store.create("坏行")
        self.store.append_message(session, Message.user("正常消息"))
        path = self.store.path_for(session.id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{这不是合法 JSON\n")
            handle.write('{"type":"unknown_event","ts":"x"}\n')
            handle.write("\n")  # 空行不算坏行

        loaded = self.store.load(session.id)
        self.assertEqual(len(loaded.messages), 1)
        self.assertEqual(loaded.messages[0].content, "正常消息")
        self.assertGreaterEqual(loaded.meta["corrupt_lines"], 2)
        self.assertEqual(loaded.turn_count, 1)

    def test_list_sessions_sorted_by_updated_at_desc(self) -> None:
        """CONTRACT 用例 3：list_sessions 只读首行 meta，按 updated_at 倒序。"""
        older = self.store.create("旧会话")
        newer = self.store.create("新会话")
        # 直接指定 updated_at，避免测试受"同一秒"影响；save_todos 会刷新首行 meta
        older.updated_at = "2026-01-01T00:00:00+08:00"
        self.store.save_todos(older)
        newer.updated_at = "2026-01-02T00:00:00+08:00"
        self.store.save_todos(newer)

        rows = self.store.list_sessions()
        self.assertEqual([row["id"] for row in rows], [newer.id, older.id])
        self.assertEqual(rows[0]["title"], "新会话")
        self.assertEqual(rows[0]["updated_at"], "2026-01-02T00:00:00+08:00")
        self.assertEqual(rows[1]["updated_at"], "2026-01-01T00:00:00+08:00")
        self.assertEqual(rows[0]["turns"], 0)
        self.assertEqual(rows[0]["state"], "idle")

    def test_append_message_updates_meta_turn_count(self) -> None:
        session = self.store.create("轮次")
        self.store.append_message(session, Message.user("q1"))
        self.store.append_message(session, Message.assistant("a1"))
        self.store.append_message(session, Message.user("q2"))
        self.assertEqual(session.turn_count, 2)

        row = next(row for row in self.store.list_sessions() if row["id"] == session.id)
        self.assertEqual(row["turns"], 2)
        self.assertEqual(self.store.load(session.id).turn_count, 2)


class TodoTest(SessionStoreTestBase):
    def test_todo_crud_and_rendering(self) -> None:
        session = self.store.create("待办")
        self.assertEqual(session.render_todos(), "（无）")

        first = session.add_todo("  买牛奶  ")
        second = session.add_todo("写周报")
        self.assertEqual((first.id, second.id), (1, 2))
        self.assertEqual(first.content, "买牛奶")
        self.assertEqual(
            session.render_todos(), "#1 [ ] 买牛奶\n#2 [ ] 写周报"
        )

        self.assertIsNotNone(session.done_todo(1))
        self.assertIsNone(session.done_todo(99))
        self.assertEqual([todo.id for todo in session.pending_todos()], [2])
        self.assertIn("#1 [x] 买牛奶", session.render_todos())

        self.assertEqual(session.clear_todos(only_done=True), 1)
        self.assertEqual([todo.id for todo in session.todos], [2])
        self.assertEqual(session.clear_todos(), 1)
        self.assertEqual(session.todos, [])
        self.assertEqual(session.render_todos(), "（无）")
        self.assertEqual(session.next_todo_id(), 1)

    def test_todo_state_is_rendered_into_state_block(self) -> None:
        session = self.store.create("状态块")
        session.add_todo("整理测试报告")
        block = session.render_state_block()
        self.assertIn("已完成轮次", block)
        self.assertIn("待办", block)
        self.assertIn("#1 [ ] 整理测试报告", block)


class SessionModelTest(SessionStoreTestBase):
    def test_visible_messages_follow_summary_upto(self) -> None:
        session = self.store.create("游标")
        self.store.append_message(session, Message.user("q1"))
        self.store.append_message(session, Message.assistant("a1"))
        self.store.append_message(session, Message.user("q2"))

        self.assertEqual(len(session.visible_messages()), 3)
        session.summary_upto = 2
        self.assertEqual([m.content for m in session.visible_messages()], ["q2"])
        # 越界游标被夹住
        session.summary_upto = 99
        self.assertEqual(session.visible_messages(), [])
        session.summary_upto = -5
        self.assertEqual(len(session.visible_messages()), 3)

    def test_now_iso_is_timezone_aware(self) -> None:
        stamp = datetime.fromisoformat(now_iso())
        self.assertIsNotNone(stamp.utcoffset())

    def test_lock_is_shared_per_session(self) -> None:
        self.assertIs(self.store.lock("abc"), self.store.lock("abc"))
        self.assertIsNot(self.store.lock("abc"), self.store.lock("def"))

    def test_exists_delete_and_illegal_id(self) -> None:
        session = self.store.create("临时")
        self.assertTrue(self.store.exists(session.id))
        self.store.delete(session.id)
        self.assertFalse(self.store.exists(session.id))
        self.store.delete(session.id)  # 幂等
        self.assertEqual(self.store.list_sessions(), [])

        with self.assertRaises(ValueError):
            self.store.path_for("../../evil")
        self.assertFalse(self.store.exists("../../evil"))


if __name__ == "__main__":
    unittest.main()
