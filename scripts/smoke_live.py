"""真实 LLM 联调冒烟：验证协议、流式、工具调用、session 全链路。

需要 `.env` 中的 LLM_API_KEY。CI 不跑它（默认离线），本地联调用：

    python scripts/smoke_live.py
    python scripts/smoke_live.py --provider glm --model glm-4-flash
"""

from __future__ import annotations

import argparse
import sys
import time
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    config = Config.from_env()
    if args.provider:
        config.provider = args.provider.lower()
        config._profile = None
    if args.model:
        config.model = args.model
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    config.stream = True

    print("=" * 68)
    print(f"真实 LLM 冒烟：provider={config.provider} model={config.model} stream=on")
    print("=" * 68)

    tracer = Tracer(trace_dir=config.traces_dir)
    store = SessionStore(config.data_dir)
    llm = LLMClient(config, tracer=tracer)
    runtime = AgentRuntime(
        config=config,
        llm=llm,
        registry=builtin_registry(),
        store=store,
        builder=ContextBuilder(config, store=store),
        compressor=Compressor(config, llm=llm, store=store),
        tracer=tracer,
    )
    session = store.create(title="smoke-live")

    checks: list[tuple[str, bool, str]] = []

    def ask(prompt: str, stream: bool = True) -> tuple[str, float]:
        started = time.perf_counter()
        buffer: list[str] = []

        def on_delta(text: str) -> None:
            buffer.append(text)
            if stream:
                sys.stdout.write(text)
                sys.stdout.flush()

        result = runtime.run(session, prompt, on_delta=on_delta)
        elapsed = (time.perf_counter() - started) * 1000
        if not buffer and result.text and stream:
            sys.stdout.write(result.text)
        print()
        if not result.ok:
            print(f"  [失败] {result.error}")
        return result.text, elapsed

    # 1) 直接回复 + 流式
    text, ms = ask("用一句话介绍你自己。")
    checks.append(("流式输出可用", bool(text), f"{ms:.0f}ms"))
    checks.append(("首 token 时间已记录", ms > 0, f"{ms:.0f}ms"))

    # 2) 单工具：calculator
    text, ms = ask("帮我算一下 (23*7+11)/4 等于多少？只给结果。")
    checks.append(("工具调用：calculator", "43" in text, text[:80]))

    # 3) 带工具 + 会话状态：todo
    text, ms = ask("记一个待办：周五交周报。")
    checks.append(("工具调用：todo add", bool(session.todos), f"{[t.content for t in session.todos]}"))

    # 4) 带工具的追问：能读到自己刚记的待办
    text, ms = ask("我现在有哪些待办？")
    checks.append(("带工具追问（读到会话状态）", "周报" in text, text[:80]))

    # 5) 多工具一轮
    text, ms = ask("查一下广州今天的天气，再算 15% 的税后 200 是多少。")
    checks.append(("一轮多工具", len([m for m in session.messages if m.role == "tool"]) >= 2, text[:80]))

    # 6) 落盘可恢复
    reloaded = store.load(session.id)
    checks.append(("会话落盘可恢复", len(reloaded.messages) >= len(session.messages) - 1, f"{len(reloaded.messages)} 条"))

    summary = tracer.summary()
    print("-" * 68)
    print(f"trace: spans={summary['spans']} kinds={summary['kinds']}")
    print(f"LLM {summary['llm_ms']}ms / 工具 {summary['tool_ms']}ms / 错误 {len(summary['errors'])}")
    print("-" * 68)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")

    runtime.close()
    tracer.close()
    all_ok = all(ok for _, ok, _ in checks)
    print("-" * 68)
    print("联调通过 ✅" if all_ok else "存在失败项 ❌（检查 key / 网络 / 模型名）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
