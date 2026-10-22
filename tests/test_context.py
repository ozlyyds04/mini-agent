"""上下文层单测：system prompt、tool 配对、L0 截断、硬预算、L0-L3 压缩。

全部离线运行（tempfile 临时目录 + FakeLLM，不联网、不读 .env）。
压缩水位靠"先构造对话、再按测量值反推 budget"来保证，不对字符数做脆弱假设。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from agent.config import Config
from agent.context.budget import ContextBudget, estimate_tokens
from agent.context.builder import ContextBuilder
from agent.context.compressor import Compressor
from agent.llm.types import Message, ToolCall
from agent.session.store import SessionStore
from tests.fake_llm import FakeLLM, FlakyLLM, text_response


class ContextTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = Config(
            provider="deepseek", data_dir=self.root, context_budget_tokens=4000
        )
        self.store = SessionStore(self.root)
        self.builder = ContextBuilder(self.config, self.store)

    # -- 辅助：构造 20 轮"有内容"的对话 ---------------------------------
    def seed_turns(self, session, turns: int) -> None:
        self.store.append_message(
            session,
            Message.user(
                "第 0 轮：请记住目标 ALPHA-0（原始目标 ORIGINAL-GOAL），后续每轮都会补充。"
            ),
        )
        self.store.append_message(
            session, Message.assistant("已记录 ALPHA-0。这是第 0 轮的回复，包含填充文字。")
        )
        for index in range(1, turns):
            self.store.append_message(
                session,
                Message.user(f"第 {index} 轮：请记住目标 ALPHA-{index}，并继续下一步。"),
            )
            self.store.append_message(
                session,
                Message.assistant(
                    f"已记录 ALPHA-{index}。这是第 {index} 轮的回复，"
                    "包含一些用于制造上下文压力的填充文字。"
                ),
            )

    def prepare_pressure(self, turns: int = 20, target_ratio: float = 0.70):
        """构造对话并反推 context_budget_tokens，使占用比例落在目标水位。"""
        session = self.store.create("压缩场景")
        self.seed_turns(session, turns)
        raw_tokens = self.builder.measure(session)
        self.assertGreater(raw_tokens, 0)
        self.config.context_budget_tokens = max(1, int(raw_tokens / target_ratio))
        return session, raw_tokens


class SystemPromptTest(ContextTestBase):
    def test_build_system_prompt_contains_time_tools_and_todos(self) -> None:
        """CONTRACT 用例 4：第一条是 system，且含当前时间、工具清单、待办状态。"""
        session = self.store.create("上下文")
        session.add_todo("整理测试报告")
        self.store.save_todos(session)
        self.store.append_message(session, Message.user("帮我看看广州天气"))

        built = self.builder.build(
            session, tools_description="- weather(city, date?): 查询城市天气"
        )
        self.assertEqual(built[0]["role"], "system")
        system = built[0]["content"]
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        self.assertIn(today, system)
        self.assertIn("weather(city, date?)", system)
        self.assertIn("整理测试报告", system)
        self.assertIn("待办", system)
        self.assertIn("当前本地时间", system)
        self.assertEqual(built[1], {"role": "user", "content": "帮我看看广州天气"})
        # extra_system 也拼进 system
        built_extra = self.builder.build(session, extra_system="## 额外约束\n只回答中文")
        self.assertIn("只回答中文", built_extra[0]["content"])

    def test_thinking_is_not_backfilled_for_deepseek(self) -> None:
        """DeepSeek 不接受 reasoning_content 回填，context 里不能出现该字段。"""
        session = self.store.create("thinking")
        self.store.append_message(session, Message.user("问题"))
        self.store.append_message(
            session, Message.assistant("回答", thinking="这是推理过程")
        )
        built = self.builder.build(session)
        self.assertNotIn("reasoning_content", built[-1])
        # 但思考过程仍要落盘可查（session 里保留）
        self.assertEqual(session.messages[-1].thinking, "这是推理过程")
        # 显式要求时才回填（Message 层能力，配置层默认关闭）
        self.assertEqual(
            session.messages[-1].to_openai(backfill_thinking=True).get("reasoning_content"),
            "这是推理过程",
        )


class ToolPairingTest(ContextTestBase):
    def test_tool_pairs_are_preserved(self) -> None:
        """CONTRACT 用例 5（前半）：assistant(tool_calls) + tool 结果成对保留。"""
        session = self.store.create("配对")
        self.store.append_message(session, Message.user("算 1+1 和 2+2"))
        calls = [
            ToolCall(id="call_a", name="calculator", arguments={"expression": "1+1"}),
            ToolCall(id="call_b", name="calculator", arguments={"expression": "2+2"}),
        ]
        self.store.append_message(session, Message.assistant(tool_calls=calls))
        self.store.append_message(
            session, Message.tool_result("call_a", "1+1 = 2", "calculator")
        )
        self.store.append_message(
            session, Message.tool_result("call_b", "2+2 = 4", "calculator")
        )

        built = self.builder.build(session)
        self.assertEqual(
            [message["role"] for message in built[1:]],
            ["user", "assistant", "tool", "tool"],
        )
        self.assertEqual(built[2]["tool_calls"][0]["id"], "call_a")
        self.assertEqual(
            [message["tool_call_id"] for message in built[3:]], ["call_a", "call_b"]
        )

    def test_assistant_dropped_when_a_tool_result_is_missing(self) -> None:
        """CONTRACT 用例 5（后半）：缺一条 tool 结果 -> 整条 assistant 丢弃。"""
        session = self.store.create("缺结果")
        self.store.append_message(session, Message.user("算两题"))
        calls = [
            ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}),
            ToolCall(id="c2", name="calculator", arguments={"expression": "2+2"}),
        ]
        self.store.append_message(session, Message.assistant(tool_calls=calls))
        # 只回填了 c1，c2 的 tool 结果丢了
        self.store.append_message(session, Message.tool_result("c1", "1+1 = 2", "calculator"))

        built = self.builder.build(session)
        self.assertEqual([message["role"] for message in built[1:]], ["user"])
        self.assertNotIn("assistant", [message["role"] for message in built[1:]])

    def test_orphan_tool_result_is_dropped(self) -> None:
        session = self.store.create("孤立 tool")
        self.store.append_message(session, Message.user("问题"))
        self.store.append_message(session, Message.tool_result("ghost", "无主结果", "x"))
        built = self.builder.build(session)
        self.assertEqual([message["role"] for message in built[1:]], ["user"])


class ToolResultTruncationTest(ContextTestBase):
    def test_long_tool_result_is_truncated_with_marker(self) -> None:
        """CONTRACT 用例 6：超长 tool 结果被截断，且截断标记存在。"""
        config = Config(
            provider="deepseek",
            data_dir=self.root,
            context_budget_tokens=8000,
            tool_result_max_chars=200,
        )
        builder = ContextBuilder(config, self.store)
        session = self.store.create("截断")
        self.store.append_message(session, Message.user("读文件"))
        call = ToolCall(id="c1", name="read_docs", arguments={"path": "README.md"})
        self.store.append_message(session, Message.assistant(tool_calls=[call]))
        original = "A" * 2000 + "TAIL"
        self.store.append_message(session, Message.tool_result("c1", original, "read_docs"))

        built = builder.build(session)
        tool_message = built[-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn("已截断", tool_message["content"])
        self.assertLess(len(tool_message["content"]), len(original))
        self.assertTrue(tool_message["content"].endswith("TAIL"))
        # build 是只读的：原始消息未被修改
        self.assertEqual(session.messages[-1].content, original)


class HardBudgetTest(ContextTestBase):
    def test_hard_budget_always_keeps_last_user_message(self) -> None:
        """CONTRACT 用例 6（硬预算）：超预算从最旧丢，但最后一条 user 必须保留。"""
        config = Config(
            provider="deepseek",
            data_dir=self.root,
            context_budget_tokens=120,
            compress_l3_ratio=0.92,
        )
        builder = ContextBuilder(config, self.store)
        session = self.store.create("硬预算")
        self.store.append_message(session, Message.user("很早之前的问题：" + "填充" * 100))
        self.store.append_message(session, Message.assistant("很早之前的回答"))
        self.store.append_message(session, Message.user("这是最后一个问题"))

        built = builder.build(session)
        self.assertEqual(built[-1]["role"], "user")
        self.assertEqual(built[-1]["content"], "这是最后一个问题")
        self.assertNotIn("很早之前的回答", json.dumps(built, ensure_ascii=False))
        self.assertEqual(built[1]["content"], "这是最后一个问题")


class TokenEstimateTest(unittest.TestCase):
    def test_estimate_and_budget_levels(self) -> None:
        self.assertGreater(estimate_tokens("你好世界"), estimate_tokens("hi"))
        self.assertGreater(estimate_tokens("a" * 400), 50)
        self.assertEqual(estimate_tokens(""), 0)

        budget = ContextBudget(total=1000)
        self.assertEqual(budget.level(0.30), 0)
        self.assertEqual(budget.level(0.60), 1)
        self.assertEqual(budget.level(0.80), 2)
        self.assertEqual(budget.level(0.92), 3)
        self.assertAlmostEqual(budget.used(600), 0.6)
        self.assertEqual(budget.limit(0.5), 500)


class CompressionTest(ContextTestBase):
    def test_l0_level_does_not_call_llm(self) -> None:
        session = self.store.create("安全区")
        self.seed_turns(session, 8)
        raw = self.builder.measure(session)
        self.config.context_budget_tokens = max(1, int(raw / 0.30))
        llm = FakeLLM()  # 空脚本：一旦被调用就会 AssertionError
        compressor = Compressor(self.config, llm=llm, store=self.store)

        report = compressor.maybe_compress(session)
        self.assertEqual(report.level, 0)
        self.assertTrue(report.ok)
        self.assertEqual(report.before_tokens, report.after_tokens)
        self.assertEqual(llm.call_count, 0)
        self.assertEqual(session.summary, "")
        self.assertEqual(session.summary_upto, 0)

    def test_l1_compression_folds_history_and_summary_reaches_system(self) -> None:
        """CONTRACT 用例 7：触发 L1，摘要非空、游标推进、compressed 消息不再进 build。"""
        session, _raw = self.prepare_pressure(turns=20, target_ratio=0.70)
        summary_text = (
            "【用户目标】持续记录 ALPHA 系列目标。【已完成的动作与结论】已记录前十几轮目标。"
            "【关键事实与ID】压缩标记 CMP-9。【未完成/待确认】（无）【当前待办】继续下一步。"
        )
        llm = FakeLLM([text_response(summary_text)])
        compressor = Compressor(self.config, llm=llm, store=self.store)

        report = compressor.maybe_compress(session)
        self.assertTrue(report.ok, report.note)
        self.assertEqual(report.level, 1, report.note)
        self.assertEqual(llm.call_count, 1)
        self.assertIn("CMP-9", session.summary)
        self.assertGreater(session.summary_upto, 0)
        self.assertLess(report.after_tokens, report.before_tokens)
        self.assertGreater(report.folded_messages, 0)
        rotated = session.messages[: session.summary_upto]
        self.assertTrue(rotated)
        self.assertTrue(all(message.compressed for message in rotated))

        # pinned：第一条 user 消息不进摘要（摘要 prompt 里看不到它的原文）
        transcript = llm.calls[0]["messages"][1]["content"]
        self.assertNotIn("ORIGINAL-GOAL", transcript)
        self.assertIn("【用户目标】", llm.calls[0]["messages"][0]["content"])

        built = self.builder.build(session)
        system = built[0]["content"]
        # "压缩后仍能回答早期目标"：摘要 + 第一条 user 原文都进了 system
        self.assertIn("较早对话的摘要", system)
        self.assertIn("CMP-9", system)
        self.assertIn("ALPHA-0", system)
        self.assertIn("ORIGINAL-GOAL", system)
        # 被压缩的消息本身不再出现在请求消息体里
        body_contents = [message.get("content") or "" for message in built[1:]]
        self.assertFalse(any("ALPHA-0" in content for content in body_contents))
        self.assertFalse(any(message.get("role") == "assistant" and "第 0 轮" in (message.get("content") or "") for message in built[1:]))

        # 归档可逆：原文逐行写入 artifacts，路径记在 session.meta
        archive = Path(session.meta["last_archive"])
        self.assertTrue(archive.is_file())
        self.assertIn(session.id, str(archive))
        rows = [
            json.loads(line)
            for line in archive.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), report.folded_messages)
        self.assertTrue(any("ALPHA-0" in (row.get("content") or "") for row in rows))

        # 摘要游标已持久化：重新加载仍能 build 出摘要
        loaded = self.store.load(session.id)
        self.assertEqual(loaded.summary, session.summary)
        self.assertEqual(loaded.summary_upto, session.summary_upto)
        self.assertIn("CMP-9", self.builder.build(loaded)[0]["content"])

    def test_compression_failure_is_reported_not_raised(self) -> None:
        """CONTRACT 用例 8：摘要抛异常时 ok=False，绝不向外抛。"""
        session, _raw = self.prepare_pressure(turns=20, target_ratio=0.70)
        llm = FlakyLLM([RuntimeError("网络超时")])
        compressor = Compressor(self.config, llm=llm, store=self.store)

        report = compressor.maybe_compress(session)  # 不应抛异常
        self.assertFalse(report.ok)
        self.assertIn("摘要调用失败", report.note)
        self.assertEqual(session.summary, "")
        self.assertEqual(session.summary_upto, 0)
        self.assertFalse(any(message.compressed for message in session.messages))

        # builder 硬预算兜底：仍然能发出合法请求，最后一条 user 一定在
        built = ContextBuilder(self.config, self.store).build(session)
        self.assertEqual(built[0]["role"], "system")
        user_messages = [message for message in built[1:] if message["role"] == "user"]
        self.assertTrue(user_messages)
        self.assertIn("ALPHA-19", user_messages[-1]["content"])

    def test_fold_without_llm_reports_failure(self) -> None:
        session, _raw = self.prepare_pressure(turns=20, target_ratio=0.70)
        compressor = Compressor(self.config, llm=None, store=self.store)
        report = compressor.maybe_compress(session)
        self.assertFalse(report.ok)
        self.assertIn("LLM", report.note)
        self.assertEqual(session.summary_upto, 0)

    def test_l2_recursive_rewrite_uses_existing_summary(self) -> None:
        session = self.store.create("L2")
        self.seed_turns(session, 20)
        session.summary = "【用户目标】旧摘要：记录 ALPHA 系列目标。"
        raw = self.builder.measure(session)
        self.config.context_budget_tokens = max(1, int(raw / 0.85))
        llm = FakeLLM([text_response("【用户目标】新摘要 L2-OK，旧摘要已合并。")])
        compressor = Compressor(self.config, llm=llm, store=self.store)

        report = compressor.maybe_compress(session)
        self.assertTrue(report.ok, report.note)
        self.assertEqual(report.level, 2, report.note)
        prompt = llm.calls[0]["messages"][0]["content"]
        self.assertIn("递归压缩", prompt)
        self.assertIn("旧摘要", prompt)
        self.assertIn("L2-OK", session.summary)
        self.assertGreater(session.summary_upto, 0)

    def test_l3_hard_trim_without_llm(self) -> None:
        session, _raw = self.prepare_pressure(turns=20, target_ratio=1.10)
        llm = FakeLLM()  # L3 不允许调模型
        compressor = Compressor(self.config, llm=llm, store=self.store)

        report = compressor.maybe_compress(session)
        self.assertEqual(report.level, 3, report.note)
        self.assertTrue(report.ok, report.note)
        self.assertGreater(report.folded_messages, 0)
        self.assertGreater(session.meta["l3_trimmed"], 0)
        self.assertEqual(llm.call_count, 0)
        self.assertLess(report.after_tokens, report.before_tokens)

        built = ContextBuilder(self.config, self.store).build(session)
        user_messages = [message for message in built[1:] if message["role"] == "user"]
        self.assertTrue(user_messages)
        self.assertIn("ALPHA-19", user_messages[-1]["content"])
        archive = Path(session.meta["last_archive"])
        self.assertTrue(archive.is_file())

    def test_pinned_facts_survive_compression(self) -> None:
        session, _raw = self.prepare_pressure(turns=20, target_ratio=0.70)
        session.meta["pinned"] = ["订单号 ORDER-42 不得丢失", "部署环境是 staging"]
        llm = FakeLLM([text_response("【用户目标】继续 ALPHA 任务 CMP-PIN。")])
        compressor = Compressor(self.config, llm=llm, store=self.store)

        report = compressor.maybe_compress(session)
        self.assertTrue(report.ok, report.note)
        system = self.builder.build(session)[0]["content"]
        self.assertIn("ORDER-42", system)
        self.assertIn("staging", system)
        self.assertIn("ALPHA-0", system)


if __name__ == "__main__":
    unittest.main()
