"""异步工具演示：提交立即返回，结果在 step 边界以"通知"注入。

对应架构题模块四：异步工具不能让用户一直等，但结果依然重要。

    python scripts/demo_async.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import Config                                    # noqa: E402
from agent.context.builder import ContextBuilder                   # noqa: E402
from agent.context.compressor import Compressor                    # noqa: E402
from agent.runtime.loop import AgentRuntime                        # noqa: E402
from agent.session.store import SessionStore                       # noqa: E402
from agent.tools.base import ToolResult, tool                      # noqa: E402
from agent.tools.registry import builtin_registry                  # noqa: E402
from agent.trace.tracer import Tracer                              # noqa: E402
from tests.fake_llm import FakeLLM, text_response, tool_response   # noqa: E402


# ---------------------------------------------------------------- 异步工具
@tool(
    name="long_report",
    description="异步生成一份周报草稿（耗时任务，提交后立即返回 task_id）。",
    is_async=True,
    timeout_s=30,
)
def long_report(topic: str) -> ToolResult:
    """生成周报草稿。

    Args:
        topic: 周报主题
    """
    time.sleep(1.2)  # 模拟真实耗时（检索、聚合、生成长文）
    return ToolResult.success(f"《{topic}》草稿已完成：本周完成 3 项、风险 1 项、下周计划 2 项。")


def main() -> int:
    config = Config.from_env(require_api_key=False)
    tmp = tempfile.mkdtemp(prefix="mini-agent-async-")
    config.data_dir = Path(tmp)
    config.stream = False
    config.enable_async_tools = True

    registry = builtin_registry()
    registry.register(long_report)

    tracer = Tracer(trace_dir=config.traces_dir)
    store = SessionStore(config.data_dir)
    llm = FakeLLM(
        [
            tool_response([("long_report", {"topic": "本周周报"})]),
            text_response("已经在后台生成周报了，你可以先做别的事，完成后我会告诉你。"),
            text_response("刚收到后台结果：本周周报草稿已完成，要我现在展开吗？"),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        llm=llm,
        registry=registry,
        store=store,
        builder=ContextBuilder(config, store=store),
        compressor=Compressor(config, llm=llm, store=store),
        tracer=tracer,
    )

    session = store.create(title="异步工具演示")
    print("=" * 68)
    print("异步工具演示：提交即返回，结果稍后注入")
    print("=" * 68)

    t0 = time.perf_counter()
    first = runtime.run(session, "帮我在后台生成本周周报")
    first_cost = (time.perf_counter() - t0) * 1000
    print(f"\n第一轮（{first_cost:.0f}ms 返回）:")
    print(f"  Agent> {first.text}")
    pending = runtime.async_runner.pending()
    print(f"  仍在后台执行的任务: {[t.to_dict() for t in pending]}")

    print("\n（等待后台任务完成…）")
    runtime.async_runner.wait_all(timeout=10)
    time.sleep(0.1)

    second = runtime.run(session, "周报好了吗？")
    print("\n第二轮（后台结果已被自动注入 context）:")
    print(f"  Agent> {second.text}")

    injected = [m for m in session.messages if m.content.startswith("[异步工具完成]")]
    print(f"\n注入的通知消息数: {len(injected)}")
    if injected:
        print(f"  内容片段: {injected[0].content[:90]}…")

    ok = bool(injected) and second.ok
    print("\n" + ("通过 ✅：结果没有丢，在 step 边界被注入" if ok else "失败 ❌"))
    runtime.close()
    tracer.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
