"""主循环集成测试：用真实的 session / context / 工具，只有 LLM 是假的。

覆盖题目要求的全部行为：直接回复、调工具、多工具、错误自愈、追问、
最大轮次、取消、busy、异步工具、trace。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from agent.config import Config
from agent.context.builder import ContextBuilder
from agent.context.compressor import Compressor
from agent.errors import SessionBusy
from agent.llm.types import Message
from agent.runtime.loop import AgentRuntime, CancelToken
from agent.session.store import SessionStore
from agent.tools.base import ToolResult, tool
from agent.tools.registry import builtin_registry
from agent.trace.tracer import Tracer
from tests.fake_llm import FakeLLM, text_response, tool_response


# ---------------------------------------------------------------- 夹具
class RuntimeFactory:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)

    def build(self, llm, **overrides):
        config = Config(
            provider="deepseek",
            api_key="test-key",
            data_dir=self.data_dir,
            stream=False,
            context_budget_tokens=4000,
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        tracer = Tracer()
        store = SessionStore(config.data_dir)
        registry = builtin_registry()
        runtime = AgentRuntime(
            config=config,
            llm=llm,
            registry=registry,
            store=store,
            builder=ContextBuilder(config, store=store),
            compressor=Compressor(config, llm=llm, store=store),
            tracer=tracer,
        )
        return runtime, store, tracer

    def close(self) -> None:
        self.tmp.cleanup()


class LoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = RuntimeFactory()
        self.addCleanup(self.factory.close)

    def session(self, runtime, store):
        return store.create(title="t")

    def assert_tool_pairing(self, session) -> None:
        """每条 assistant 的 tool_call 都必须有对应的 tool 结果（否则厂商会 400）。"""
        results = {m.tool_call_id for m in session.messages if m.role == "tool"}
        for message in session.messages:
            if message.role == "assistant":
                for call in message.tool_calls:
                    self.assertIn(call.id, results, f"tool_call {call.id} 没有对应结果")


# ---------------------------------------------------------------- 用例
class TestBasicLoop(LoopTestCase):
    def test_direct_reply_without_tool(self) -> None:
        llm = FakeLLM([text_response("你好，我是 mini-agent。")])
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        result = runtime.run(session, "你是谁？")

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "你好，我是 mini-agent。")
        self.assertEqual(result.steps, 1)
        self.assertEqual(llm.call_count, 1)
        # 请求里确实带了工具 schema
        self.assertTrue(llm.calls[0]["tools"])
        self.assertEqual(len(llm.calls[0]["tools"]), 5)

    def test_single_tool_then_answer(self) -> None:
        llm = FakeLLM(
            [
                tool_response([("calculator", {"expression": "12*12"})]),
                text_response("12*12 = 144"),
            ]
        )
        runtime, store, tracer = self.factory.build(llm)
        session = store.create()

        result = runtime.run(session, "算一下 12*12")

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "12*12 = 144")
        self.assertEqual(result.steps, 2)
        self.assert_tool_pairing(session)
        # 第二轮的 context 里能看到工具结果
        second_request = llm.calls[1]["messages"]
        self.assertTrue(any(m["role"] == "tool" and "144" in (m.get("content") or "") for m in second_request))
        # trace 覆盖 LLM 与工具
        kinds = {span.kind for span in tracer.spans}
        self.assertIn("llm.request", kinds)
        self.assertIn("tool.call", kinds)

    def test_multiple_tools_in_one_turn(self) -> None:
        llm = FakeLLM(
            [
                tool_response(
                    [
                        ("weather", {"city": "广州"}),
                        ("todo", {"action": "add", "content": "带伞"}),
                    ]
                ),
                text_response("天气查到了，待办也记了。"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        result = runtime.run(session, "查广州天气并记待办")

        self.assertTrue(result.ok)
        self.assert_tool_pairing(session)
        self.assertEqual([item.content for item in session.todos], ["带伞"])

    def test_todo_isolation_between_sessions(self) -> None:
        llm = FakeLLM(
            [
                tool_response([("todo", {"action": "add", "content": "窗口1的事"})]),
                text_response("已记"),
                tool_response([("todo", {"action": "add", "content": "窗口2的事"})]),
                text_response("已记"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        window1 = store.create(title="w1")
        window2 = store.create(title="w2")

        runtime.run(window1, "记待办")
        runtime.run(window2, "记待办")

        self.assertEqual([i.content for i in window1.todos], ["窗口1的事"])
        self.assertEqual([i.content for i in window2.todos], ["窗口2的事"])


class TestErrorRecovery(LoopTestCase):
    def test_unknown_tool_then_self_heal(self) -> None:
        llm = FakeLLM(
            [
                tool_response([("nonexistent_tool", {})]),
                tool_response([("calculator", {"expression": "1+1"})]),
                text_response("结果是 2"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        result = runtime.run(session, "算 1+1")

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.steps, 3)
        # 失败被回喂成观测，而不是抛异常打断
        observations = [m.content for m in session.messages if m.role == "tool"]
        self.assertTrue(any("[tool_error]" in text for text in observations))

    def test_missing_argument_gets_fed_back(self) -> None:
        llm = FakeLLM(
            [
                tool_response([("calculator", {})]),                     # 缺 expression
                tool_response([("calculator", {"expression": "2+2"})]),  # 修正
                text_response("2+2 = 4"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        result = runtime.run(session, "算 2+2")
        self.assertTrue(result.ok, result.error)
        observations = [m.content for m in session.messages if m.role == "tool"]
        self.assertTrue(any("缺少必填参数" in text for text in observations))

    def test_max_steps_gives_fallback(self) -> None:
        llm = FakeLLM(
            [
                tool_response([("calculator", {"expression": "1+1"})]),
                tool_response([("calculator", {"expression": "2+2"})]),
            ]
        )
        runtime, store, _ = self.factory.build(llm, max_steps=2)
        session = store.create()

        result = runtime.run(session, "一直算")

        self.assertFalse(result.ok)
        self.assertIn("最大工具轮次", result.error)
        self.assertIn("2", result.text)  # 至少把已完成结果带出来
        self.assertEqual(result.steps, 2)
        # 兜底也必须保证配对，否则下一次请求会 400
        self.assert_tool_pairing(session)

    def test_max_turns_guard(self) -> None:
        llm = FakeLLM([text_response("不该被调用")])
        runtime, store, _ = self.factory.build(llm, max_turns=0)
        session = store.create()

        result = runtime.run(session, "hi")

        self.assertFalse(result.ok)
        self.assertIn("轮次", result.error)
        self.assertEqual(llm.call_count, 0)

    def test_cancel_returns_early(self) -> None:
        llm = FakeLLM([text_response("不会被调用")])
        runtime, store, _ = self.factory.build(llm)
        session = store.create()
        cancel = CancelToken()
        cancel.cancel()

        result = runtime.run(session, "hi", cancel=cancel)

        self.assertTrue(result.cancelled)
        self.assertEqual(llm.call_count, 0)

    def test_busy_session_is_rejected(self) -> None:
        """同一 session 并发两个 run：第二个必须被拒绝，而不是交错写坏状态。

        注意：store 内部用的是可重入锁（store 自身方法会嵌套加锁），所以"占用"
        必须来自另一个线程——这也正是 CLI 的真实形态（请求跑在 worker 线程）。
        """
        import threading

        llm = FakeLLM([text_response("ok")])
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        lock = store.lock(session.id)
        acquired = threading.Event()
        release = threading.Event()

        def holder() -> None:
            lock.acquire()
            acquired.set()
            release.wait(3)
            lock.release()

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(acquired.wait(2))
        try:
            result = runtime.run(session, "hi")
        finally:
            release.set()
            thread.join(timeout=3)

        self.assertFalse(result.ok)
        self.assertIn("busy", result.error)
        self.assertEqual(llm.call_count, 0)

    def test_llm_error_does_not_crash(self) -> None:
        from agent.errors import LLMError

        def router(messages, tools):
            raise LLMError("模拟模型不可用")

        runtime, store, _ = self.factory.build(FakeLLM(router=router))
        session = store.create()

        result = runtime.run(session, "hi")

        self.assertFalse(result.ok)
        self.assertIn("模型不可用", result.error)
        # 状态必须回到 idle，否则会话永久卡死
        self.assertEqual(session.state, "idle")


class TestFollowUp(LoopTestCase):
    def test_pure_conversation_follow_up(self) -> None:
        llm = FakeLLM(
            [
                text_response("我叫 Mini。"),
                text_response("你刚才问了我的名字。"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        runtime.run(session, "你叫什么？")
        result = runtime.run(session, "我刚才问了你什么？")

        self.assertTrue(result.ok)
        # 第二次请求必须包含第一轮的问答
        second = llm.calls[1]["messages"]
        contents = " ".join(str(m.get("content")) for m in second)
        self.assertIn("我叫 Mini", contents)
        self.assertIn("你叫什么", contents)

    def test_tool_follow_up_keeps_pairing(self) -> None:
        llm = FakeLLM(
            [
                tool_response([("todo", {"action": "add", "content": "写周报"})]),
                text_response("记好了"),
                tool_response([("todo", {"action": "list"})]),
                text_response("你有一条待办：写周报"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        runtime.run(session, "记个待办：写周报")
        result = runtime.run(session, "我的待办有啥？")

        self.assertTrue(result.ok)
        self.assert_tool_pairing(session)
        # 待办以结构化状态常驻 system，而不是靠模型回忆
        system_prompt = llm.last_system_prompt()
        self.assertIn("写周报", system_prompt)


class TestAsyncTool(LoopTestCase):
    def test_async_submit_then_inject(self) -> None:
        @tool(name="slow_task", description="异步任务", is_async=True, timeout_s=10)
        def slow_task(topic: str = "x") -> ToolResult:
            time.sleep(0.15)
            return ToolResult.success(f"{topic} 完成")

        from agent.runtime.async_tools import AsyncToolRunner

        llm = FakeLLM(
            [
                tool_response([("slow_task", {"topic": "周报"})]),
                text_response("已提交后台任务"),
                text_response("后台任务完成了"),
            ]
        )
        runtime, store, _ = self.factory.build(llm, enable_async_tools=True)
        runtime.registry.register(slow_task)
        runtime.async_runner = AsyncToolRunner()
        session = store.create()

        first = runtime.run(session, "后台生成周报")
        self.assertTrue(first.ok)
        # 立即返回，不阻塞
        self.assertLess(first.steps, 3)

        time.sleep(0.3)
        second = runtime.run(session, "好了吗？")
        self.assertTrue(second.ok)

        injected = [m for m in session.messages if m.content.startswith("[异步工具完成]")]
        self.assertEqual(len(injected), 1)
        self.assertIn("完成", injected[0].content)
        self.assert_tool_pairing(session)

    def test_async_tool_failure_does_not_break_run(self) -> None:
        @tool(name="bad_async", description="会失败的异步任务", is_async=True)
        def bad_async(x: str = "") -> ToolResult:
            raise RuntimeError("后台炸了")

        from agent.runtime.async_tools import AsyncToolRunner

        llm = FakeLLM(
            [
                tool_response([("bad_async", {"x": "1"})]),
                text_response("已提交"),
                text_response("后台失败了，我换个方式"),
            ]
        )
        runtime, store, _ = self.factory.build(llm)
        runtime.registry.register(bad_async)
        runtime.async_runner = AsyncToolRunner()
        session = store.create()

        runtime.run(session, "跑个后台任务")
        time.sleep(0.2)
        result = runtime.run(session, "怎么样")
        self.assertTrue(result.ok)

    def test_pending_result_survives_until_next_run(self) -> None:
        """结果在 run 结束后才到，也不能丢——下次 run 自动补投。"""
        from agent.runtime.async_tools import AsyncToolRunner

        @tool(name="late_task", description="很慢的异步任务", is_async=True)
        def late_task(x: str = "") -> ToolResult:
            time.sleep(0.2)
            return ToolResult.success("迟到但重要")

        llm = FakeLLM([tool_response([("late_task", {"x": "1"})]), text_response("已提交"), text_response("收到结果")])
        runtime, store, _ = self.factory.build(llm)
        runtime.registry.register(late_task)
        runtime.async_runner = AsyncToolRunner()
        session = store.create()

        runtime.run(session, "提交任务")
        time.sleep(0.4)          # 结果在我们的 run 之外到达
        runtime.run(session, "有结果了吗")   # 下一次 run 开头补投
        injected = [m for m in session.messages if m.content.startswith("[异步工具完成]")]
        self.assertEqual(len(injected), 1)


class TestContextWindow(LoopTestCase):
    def test_compression_runs_with_tight_budget(self) -> None:
        """预算很小时应触发压缩，且不能把请求搞坏。"""
        script = []
        for index in range(12):
            script.append(text_response(f"第{index}轮回答，这里放一些占位内容让上下文变长。" * 3))
        llm = FakeLLM(script, default=text_response("收尾"))
        runtime, store, _ = self.factory.build(llm, context_budget_tokens=300, keep_recent_turns=2)
        session = store.create()

        for index in range(12):
            result = runtime.run(session, f"第{index}轮问题，也写长一点。" * 2)
            self.assertTrue(result.ok, result.error)

        # 最后一次请求必须仍然合法：system 在最前、最后一条是 user
        last = llm.last_messages()
        self.assertEqual(last[0]["role"], "system")
        self.assertEqual(last[-1]["role"], "user")
        # 并且确实发生过压缩
        self.assertTrue(
            any("压缩" in m.content for m in session.messages if m.role == "user")
            or session.summary_upto > 0
            or len(session.messages) > len(last),
            "预期触发过压缩",
        )

    def test_system_prompt_contains_time_tools_and_todos(self) -> None:
        llm = FakeLLM([tool_response([("todo", {"action": "add", "content": "买咖啡"})]), text_response("ok"), text_response("ok")])
        runtime, store, _ = self.factory.build(llm)
        session = store.create()

        runtime.run(session, "记个待办")
        runtime.run(session, "再看看")

        system = llm.last_system_prompt()
        self.assertIn("calculator", system)     # 工具清单
        self.assertIn("买咖啡", system)          # 结构化状态
        self.assertRegex(system, r"20\d\d-\d\d-\d\d")  # 当前时间


if __name__ == "__main__":
    unittest.main()
