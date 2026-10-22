"""记忆召回测试：判定、召回、渲染，以及与主循环的集成。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.config import Config
from agent.context.builder import ContextBuilder
from agent.context.compressor import Compressor
from agent.llm.types import Message
from agent.memory.recall import MemoryRecaller, RecalledSnippet
from agent.runtime.loop import AgentRuntime
from agent.session.models import Session
from agent.session.store import SessionStore
from agent.tools.registry import builtin_registry
from agent.trace.tracer import Tracer
from tests.fake_llm import FakeLLM, text_response


def make_session(messages: list[tuple[str, str]]) -> Session:
    session = Session(id="s1")
    for role, content in messages:
        session.append(Message(role=role, content=content))
    return session


class TestShouldRecall(unittest.TestCase):
    def setUp(self) -> None:
        self.recaller = MemoryRecaller()

    def test_trigger_words(self) -> None:
        for text in ("上次那个方案怎么样", "我之前说过什么", "你还记得吗", "我一般几点开会", "last time we agreed"):
            self.assertTrue(self.recaller.should_recall(text), text)

    def test_no_trigger(self) -> None:
        for text in ("帮我算一下 3*7", "今天广州天气如何", "写个周报"):
            self.assertFalse(self.recaller.should_recall(text), text)


class TestRecall(unittest.TestCase):
    def test_recalls_older_message(self) -> None:
        session = make_session(
            [
                ("user", "我们决定用 Postgres 做主库"),
                ("assistant", "好的，已记录：主库 Postgres"),
                ("user", "今天天气不错"),
                ("assistant", "是的"),
            ]
        )
        recaller = MemoryRecaller(recent_window=2)
        snippets = recaller.recall(session, "上次我们说的主库用什么？")
        self.assertTrue(snippets)
        self.assertIn("Postgres", snippets[0].text)
        self.assertGreater(snippets[0].score, 0)

    def test_recent_window_is_excluded(self) -> None:
        session = make_session(
            [("user", "Postgres 主库"), ("assistant", "好"), ("user", "Postgres 又提一次"), ("assistant", "嗯")]
        )
        recaller = MemoryRecaller(recent_window=4)
        # 全部消息都在最近窗口内，无需召回（它们本来就在 context 里）
        self.assertEqual(recaller.recall(session, "Postgres 是什么"), [])

    def test_no_overlap_returns_empty(self) -> None:
        session = make_session([("user", "今天吃火锅"), ("assistant", "好"), ("user", "x"), ("assistant", "y")])
        recaller = MemoryRecaller(recent_window=2)
        self.assertEqual(recaller.recall(session, "上周的 Kubernetes 部署"), [])

    def test_top_k_limit(self) -> None:
        session = make_session(
            [(f"user", f"第{i}次讨论 Postgres 迁移") for i in range(10)]
            + [("user", "最近一句"), ("assistant", "好")]
        )
        recaller = MemoryRecaller(recent_window=1, max_items=3)
        self.assertEqual(len(recaller.recall(session, "Postgres 迁移")), 3)


class TestRender(unittest.TestCase):
    def test_render_has_warning_banner(self) -> None:
        recaller = MemoryRecaller()
        text = recaller.render([RecalledSnippet(index=3, role="user", text="用 Postgres", score=0.5)])
        self.assertIn("相关的历史片段", text)
        self.assertIn("不是当前指令", text)  # 防止模型把旧要求当新指令
        self.assertIn("历史#3", text)

    def test_recall_for_skips_when_no_trigger(self) -> None:
        session = make_session([("user", "Postgres"), ("assistant", "ok")] * 3)
        recaller = MemoryRecaller(recent_window=1)
        self.assertEqual(recaller.recall_for(session, "帮我算 1+1"), "")

    def test_char_budget(self) -> None:
        recaller = MemoryRecaller(max_chars=80)
        snippets = [RecalledSnippet(index=i, role="user", text="很长的内容" * 50, score=1.0) for i in range(5)]
        rendered = recaller.render(snippets)
        self.assertLessEqual(len(rendered), 80 + 20 + 200)  # 允许最后一行的截断余量


class TestLoopIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = Path(self.tmp.name)

    def build(self, llm, **overrides):
        config = Config(
            provider="deepseek",
            api_key="test-key",
            data_dir=self.data_dir,
            stream=False,
            context_budget_tokens=4000,
            keep_recent_turns=1,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        tracer = Tracer()
        store = SessionStore(config.data_dir)
        runtime = AgentRuntime(
            config=config,
            llm=llm,
            registry=builtin_registry(),
            store=store,
            builder=ContextBuilder(config, store=store),
            compressor=Compressor(config, llm=llm, store=store),
            tracer=tracer,
        )
        return runtime, store

    def test_recall_injected_into_system_prompt(self) -> None:
        llm = FakeLLM(
            [
                text_response("好的，主库用 Postgres。"),
                text_response("今天确实不错。"),
                text_response("上次说的是 Postgres。"),
            ]
        )
        runtime, store = self.build(llm)
        session = store.create()

        runtime.run(session, "我们决定用 Postgres 做主库")
        runtime.run(session, "今天天气不错")
        result = runtime.run(session, "上次我们说的主库用什么？")

        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.recalled_snippets, 1)
        system = llm.last_system_prompt()
        self.assertIn("相关的历史片段", system)
        self.assertIn("Postgres", system)

    def test_no_recall_block_without_trigger(self) -> None:
        llm = FakeLLM([text_response("ok"), text_response("ok")])
        runtime, store = self.build(llm)
        session = store.create()

        runtime.run(session, "记住：主库是 Postgres")
        result = runtime.run(session, "帮我算 2+2")

        self.assertEqual(result.recalled_snippets, 0)
        self.assertNotIn("相关的历史片段", llm.last_system_prompt())

    def test_can_be_disabled(self) -> None:
        llm = FakeLLM([text_response("ok"), text_response("ok")])
        runtime, store = self.build(llm, enable_memory_recall=False)
        self.assertIsNone(runtime.memory)
        session = store.create()
        runtime.run(session, "记住：主库是 Postgres")
        result = runtime.run(session, "上次说的主库是什么？")
        self.assertEqual(result.recalled_snippets, 0)


if __name__ == "__main__":
    unittest.main()
