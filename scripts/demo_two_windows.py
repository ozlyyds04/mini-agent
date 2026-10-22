"""复现题面场景：用户 A 开两个窗口，各自独立，可随时切回继续聊。

题面原文：
    用户 A 开了窗口 1：让 Agent 查天气记待办
    用户 A 开了窗口 2：让 Agent 写周报记待办
    这两个窗口应该是独立的 session，用户 A 可以随时接着窗口 1/2 继续聊，彼此不会影响。

这个脚本把这句话跑成可验证的断言，并且**默认不需要 API key**——用可编程的
假 LLM 跑完整链路，证明"会话隔离"是运行时的结构保证，而不是模型碰巧表现好。

用法：
    python scripts/demo_two_windows.py            # 没有 key 时自动走离线模式
    python scripts/demo_two_windows.py --live     # 用真实 LLM（需 .env 配置）
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import Config                                    # noqa: E402
from agent.context.builder import ContextBuilder                   # noqa: E402
from agent.context.compressor import Compressor                    # noqa: E402
from agent.llm.client import LLMClient                             # noqa: E402
from agent.runtime.loop import AgentRuntime                        # noqa: E402
from agent.session.store import SessionStore                       # noqa: E402
from agent.tools.registry import builtin_registry                  # noqa: E402
from agent.trace.tracer import Tracer                              # noqa: E402
from tests.fake_llm import FakeLLM, text_response, tool_response   # noqa: E402


WINDOW1_PROMPT = "查一下广州今天的天气，然后记个待办：明天带伞"
WINDOW2_PROMPT = "帮我写周报，顺便记个待办：周五交周报"
WINDOW1_FOLLOW_UP = "我现在的待办有哪些？"
WINDOW2_FOLLOW_UP = "我这个窗口的待办有哪些？"


def build_runtime(config: Config, llm, store: SessionStore, tracer: Tracer) -> AgentRuntime:
    registry = builtin_registry()
    return AgentRuntime(
        config=config,
        llm=llm,
        registry=registry,
        store=store,
        builder=ContextBuilder(config, store=store),
        compressor=Compressor(config, llm=llm, store=store),
        tracer=tracer,
    )


def offline_llm() -> FakeLLM:
    """一段剧本：两个窗口各跑两轮工具，最后各自 list 自己的待办。"""
    script = [
        # 窗口 1：一次返回两个工具调用（查天气 + 记待办）
        tool_response(
            [
                ("weather", {"city": "广州"}),
                ("todo", {"action": "add", "content": "明天带伞"}),
            ],
            thinking="用户要两件事：查天气、记待办。我可以一次都做完。",
        ),
        text_response("广州今天多云，24–30℃。已记入待办：明天带伞。"),
        # 窗口 2
        tool_response([("todo", {"action": "add", "content": "周五交周报"})]),
        text_response("周报要点我列了三块：本周进展、风险、下周计划。已记入待办：周五交周报。"),
        # 切回窗口 1 追问
        tool_response([("todo", {"action": "list"})]),
        text_response("窗口 1 只有一条待办：明天带伞。"),
        # 窗口 2 追问
        tool_response([("todo", {"action": "list"})]),
        text_response("这个窗口只有一条待办：周五交周报。"),
    ]
    return FakeLLM(script)


def print_banner(mode: str) -> None:
    print("=" * 68)
    print(f"用户 A 两窗口会话隔离演示（{mode}）")
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="用真实 LLM（需要 LLM_API_KEY）")
    parser.add_argument("--keep", action="store_true", help="保留临时数据目录")
    args = parser.parse_args()

    config = Config.from_env(require_api_key=False)
    tmp = tempfile.mkdtemp(prefix="mini-agent-demo-")
    config.data_dir = Path(tmp)
    config.stream = False

    live = args.live and bool(config.api_key)
    if args.live and not config.api_key:
        print("没有检测到 LLM_API_KEY，自动降级为离线模式。")
    print_banner("真实 LLM" if live else "离线 FakeLLM，无需 key")
    print(f"模型：{config.model if live else 'fake-model'}   数据目录：{tmp}\n")

    tracer = Tracer(trace_dir=config.traces_dir)
    store = SessionStore(config.data_dir)
    llm = LLMClient(config, tracer=tracer) if live else offline_llm()
    runtime = build_runtime(config, llm, store, tracer)

    # 两个窗口 = 两个 session
    window1 = store.create(title="窗口1-天气与待办")
    window2 = store.create(title="窗口2-周报与待办")
    print(f"窗口1 session = {window1.id}")
    print(f"窗口2 session = {window2.id}\n")

    def ask(session, prompt: str) -> str:
        result = runtime.run(session, prompt)
        status = "ok" if result.ok else f"failed({result.error})"
        print(f"[{session.title}] 你> {prompt}")
        print(f"[{session.title}] Agent> {result.text or '（无输出）'}")
        print(f"    steps={result.steps} trace={result.trace_id} {status}")
        print(f"    待办 = {[item.content for item in session.todos]}\n")
        return result.text

    ask(window1, WINDOW1_PROMPT)
    ask(window2, WINDOW2_PROMPT)
    ask(window1, WINDOW1_FOLLOW_UP)
    ask(window2, WINDOW2_FOLLOW_UP)

    # ---------------------------------------------------------------- 断言
    print("-" * 68)
    print("结构断言（这些才是『会话隔离』的真正证据，不依赖模型表现）")
    checks: list[tuple[str, bool, str]] = []

    w1_todos = [item.content for item in window1.todos]
    w2_todos = [item.content for item in window2.todos]
    checks.append(("窗口1 的待办只包含自己的", w1_todos == ["明天带伞"], str(w1_todos)))
    checks.append(("窗口2 的待办只包含自己的", w2_todos == ["周五交周报"], str(w2_todos)))
    checks.append(
        ("两个 session 的 id 不同", window1.id != window2.id, f"{window1.id} / {window2.id}")
    )
    checks.append(
        ("两个 session 的消息历史互相独立",
         len(window1.messages) > 0 and window1.messages is not window2.messages,
         f"w1={len(window1.messages)} 条, w2={len(window2.messages)} 条")
    )

    # 落盘 + 重新加载，验证"随时接着聊"
    reloaded1 = store.load(window1.id)
    checks.append(
        ("窗口1 重新加载后待办仍在", [i.content for i in reloaded1.todos] == ["明天带伞"],
         str([i.content for i in reloaded1.todos]))
    )
    reloaded2 = store.load(window2.id)
    checks.append(
        ("窗口2 重新加载后待办仍在", [i.content for i in reloaded2.todos] == ["周五交周报"],
         str([i.content for i in reloaded2.todos]))
    )

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")

    runtime.close()
    tracer.close()

    all_ok = all(ok for _, ok, _ in checks)
    print("-" * 68)
    print("全部通过 ✅" if all_ok else "存在失败项 ❌")
    if not args.keep:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
