"""交互式 CLI：多窗口（session）切换、流式输出、工具调用可视化。

题目里的场景在这里可以直接复现：

    /new 查天气     ← 窗口 1
    /new 写周报     ← 窗口 2
    /sessions       ← 两个窗口各自独立
    /switch s1      ← 切回窗口 1 接着聊

设计上 CLI 只是一个"事件订阅者"：主循环通过 EventBus 广播 tool.call /
tool.result / compression 等事件，CLI 负责展示。所以同一套运行时可以接
终端、Web 或测试，互不影响。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

from .config import Config
from .errors import ConfigError
from .llm.client import LLMClient
from .runtime.events import (
    ASYNC_DONE,
    ASYNC_SUBMIT,
    COMPRESSION,
    ERROR,
    Event,
    EventBus,
    TOOL_CALL,
    TOOL_RESULT,
)
from .runtime.loop import AgentRuntime, CancelToken, RunResult
from .session.store import SessionStore
from .context.builder import ContextBuilder
from .context.compressor import Compressor
from .tools.registry import builtin_registry
from .trace.tracer import Tracer


# ---------------------------------------------------------------- 终端样式
def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class Style:
    _on = _color_enabled()

    @classmethod
    def _wrap(cls, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls._on else text

    @classmethod
    def dim(cls, text: str) -> str:
        return cls._wrap("2", text)

    @classmethod
    def cyan(cls, text: str) -> str:
        return cls._wrap("36", text)

    @classmethod
    def green(cls, text: str) -> str:
        return cls._wrap("32", text)

    @classmethod
    def red(cls, text: str) -> str:
        return cls._wrap("31", text)

    @classmethod
    def yellow(cls, text: str) -> str:
        return cls._wrap("33", text)

    @classmethod
    def bold(cls, text: str) -> str:
        return cls._wrap("1", text)


HELP = """\
可用命令：
  /new [标题]        新建一个会话（窗口），并切换过去
  /switch <id|序号>  切换到已有会话
  /sessions          列出全部会话
  /todos             查看当前会话的待办
  /tasks             查看异步工具任务
  /trace             查看本次运行的 trace 汇总
  /title <文本>      重命名当前会话
  /help              显示本帮助
  /quit              退出（Ctrl-C 可中断正在执行的请求）
"""


def _ensure_utf8_stdio() -> None:
    """Windows 控制台默认可能是 cp936，emoji / 箭头会直接抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 极少数终端不支持
            pass


# ---------------------------------------------------------------- 主类
class ChatCLI:
    def __init__(
        self,
        config: Config,
        session_id: str | None = None,
        show_thinking: bool = True,
        show_trace: bool = False,
    ) -> None:
        self.config = config
        self.show_thinking = show_thinking
        self.show_trace = show_trace
        self._delta_printed = False
        self._current_cancel: CancelToken | None = None

        self.tracer = Tracer(trace_dir=config.traces_dir, echo=False)
        self.bus = EventBus()
        self.bus.subscribe(self._on_event)

        self.store = SessionStore(config.data_dir)
        self.registry = builtin_registry()
        self.llm = LLMClient(config, tracer=self.tracer)
        self.builder = ContextBuilder(config, store=self.store)
        self.compressor = Compressor(config, llm=self.llm, store=self.store)
        self.runtime = AgentRuntime(
            config=config,
            llm=self.llm,
            registry=self.registry,
            store=self.store,
            builder=self.builder,
            compressor=self.compressor,
            tracer=self.tracer,
            bus=self.bus,
        )

        if session_id:
            self.session = self.store.load(session_id)
        else:
            self.session = self.store.create(title="新会话")

    # ------------------------------------------------------------ 事件展示
    def _on_event(self, event: Event) -> None:
        kind = event.kind
        payload = event.payload
        if kind == TOOL_CALL:
            args = payload.get("arguments") or {}
            rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(Style.cyan(f"  ⚙ 调用工具 {payload.get('tool')}({rendered})"))
        elif kind == TOOL_RESULT:
            mark = Style.green("✓") if payload.get("ok") else Style.red("✗")
            preview = (payload.get("preview") or "").replace("\n", " ")[:120]
            print(Style.dim(f"  {mark} {payload.get('tool')}: {preview}"))
        elif kind == ASYNC_SUBMIT:
            print(Style.yellow(f"  ⏳ 异步任务已提交 {payload.get('tool')} task={payload.get('task_id')}"))
        elif kind == ASYNC_DONE:
            print(Style.yellow(f"  ⏰ 异步任务完成 task={payload.get('task_id')} status={payload.get('status')}"))
        elif kind == COMPRESSION:
            if payload.get("level", 0) > 0 or not payload.get("ok", True):
                print(
                    Style.dim(
                        f"  ⤵ context 压缩 L{payload.get('level')} "
                        f"{payload.get('before_tokens')}→{payload.get('after_tokens')} tokens"
                        + ("" if payload.get("ok", True) else f"（失败：{payload.get('note')}）")
                    )
                )
        elif kind == ERROR:
            print(Style.red(f"  ! {payload.get('error')}"))

    def _on_delta(self, text: str) -> None:
        self._delta_printed = True
        sys.stdout.write(text)
        sys.stdout.flush()

    # ------------------------------------------------------------ 提问
    def ask(self, text: str) -> RunResult:
        self._delta_printed = False
        cancel = CancelToken()
        self._current_cancel = cancel
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                box["result"] = self.runtime.run(
                    self.session, text, cancel=cancel, on_delta=self._on_delta
                )
            except BaseException as exc:  # noqa: BLE001 - 交给主线程展示
                box["error"] = exc

        thread = threading.Thread(target=worker, name="mini-agent-run", daemon=True)
        thread.start()
        try:
            while thread.is_alive():
                thread.join(timeout=0.15)
        except KeyboardInterrupt:
            cancel.cancel()
            print(Style.yellow("\n[已请求取消，将在当前步骤结束后生效]"))
            thread.join()

        self._current_cancel = None
        if "error" in box:
            raise box["error"]
        result: RunResult = box["result"]
        # 流式已经逐字打印过，避免重复输出
        if not self._delta_printed and result.text:
            print(result.text)
        if not self._delta_printed and not result.text:
            print(Style.dim("（无输出）"))
        print()
        return result

    # ------------------------------------------------------------ 命令
    def cmd_sessions(self) -> None:
        rows = self.store.list_sessions()
        if not rows:
            print("（还没有会话）")
            return
        for index, row in enumerate(rows, 1):
            marker = Style.green("*") if row["id"] == self.session.id else " "
            meta = Style.dim(
                f"turns={row.get('turns', 0)} updated={row.get('updated_at', '')}"
            )
            title = row.get("title") or "(无标题)"
            print(f"{marker} {index}. {Style.bold(row['id'])}  {title}  {meta}")

    def cmd_new(self, title: str) -> None:
        self.session = self.store.create(title=title or "新会话")
        print(Style.dim(f"已切换到新会话 {self.session.id}"))

    def cmd_switch(self, key: str) -> None:
        rows = self.store.list_sessions()
        target = None
        if key.isdigit():
            index = int(key) - 1
            if 0 <= index < len(rows):
                target = rows[index]["id"]
        else:
            target = key
        if not target:
            print(Style.red(f"找不到会话 {key}"))
            return
        self.session = self.store.load(target)
        print(Style.dim(f"已切换到会话 {self.session.id}（{self.session.title}）"))

    def handle_command(self, line: str) -> bool:
        """返回 False 表示退出。"""
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            return False
        if command in ("/help", "/h", "/?"):
            print(HELP)
        elif command == "/new":
            self.cmd_new(argument)
        elif command == "/switch":
            if not argument:
                print("用法：/switch <id|序号>")
            else:
                self.cmd_switch(argument)
        elif command == "/sessions":
            self.cmd_sessions()
        elif command == "/todos":
            print(self.session.render_todos())
        elif command == "/tasks":
            runner = self.runtime.async_runner
            if runner is None:
                print("异步工具未启用（MINI_AGENT_ASYNC_TOOLS=0）")
            else:
                tasks = runner.drained_tasks()
                if not tasks:
                    print("（没有已完成的异步任务）")
                for task in tasks:
                    print(f"- {task.task_id} {task.tool_name} {task.status} {task.error}")
        elif command == "/trace":
            summary = self.tracer.summary()
            print(f"run_id={summary['run_id']} spans={summary['spans']} kinds={summary['kinds']}")
            print(
                f"LLM 累计 {summary['llm_ms']}ms / 工具累计 {summary['tool_ms']}ms "
                f"/ 错误 {len(summary['errors'])}"
            )
            for item in summary["errors"][-5:]:
                print(Style.red(f"  - {item}"))
        elif command == "/title":
            if argument:
                self.session.title = argument
                self.store.append_note(self.session, "title", {"title": argument})
                print(Style.dim("已重命名"))
            else:
                print(f"当前标题：{self.session.title}")
        else:
            print(Style.yellow(f"未知命令 {command}，输入 /help 查看帮助"))
        return True

    # ------------------------------------------------------------ REPL
    def repl(self) -> None:
        print(Style.bold("mini-agent") + Style.dim(" — 从零实现的最小可用 Agent（/help 查看命令）"))
        print(Style.dim(f"模型：{self.config.model}  会话：{self.session.id}  /quit 退出"))
        print()
        while True:
            try:
                line = input(Style.green("你> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue
            result = self.ask(line)
            if self.show_trace:
                summary = self.tracer.summary()
                print(
                    Style.dim(
                        f"  [trace] steps={result.steps} spans={summary['spans']} "
                        f"tokens={result.usage.get('total_tokens', 0)}"
                    )
                )
        self.close()

    def close(self) -> None:
        self.runtime.close()
        self.tracer.close()


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="从零实现的最小可用 Agent（零第三方依赖）",
    )
    parser.add_argument("--once", metavar="TEXT", help="只执行一轮然后退出（适合脚本/CI）")
    parser.add_argument("--session", metavar="ID", help="恢复已有会话")
    parser.add_argument("--list-sessions", action="store_true", help="列出全部会话后退出")
    parser.add_argument("--new-session", metavar="TITLE", help="新建会话（可配合 --once）")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--hide-thinking", action="store_true", help="不展示模型思考过程")
    parser.add_argument("--trace", action="store_true", help="每轮结束后打印 trace 汇总")
    parser.add_argument("--model", help="覆盖 LLM_MODEL（默认 deepseek-chat）")
    parser.add_argument("--data-dir", help="会话/产物/trace 的存放目录")
    parser.add_argument("--version", action="version", version="mini-agent 1.0.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    args = build_parser().parse_args(argv)

    try:
        config = Config.from_env(require_api_key=not args.list_sessions)
    except ConfigError as exc:
        print(Style.red(str(exc)), file=sys.stderr)
        return 2

    if args.model:
        config.model = args.model
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.no_stream:
        config.stream = False

    if args.list_sessions:
        store = SessionStore(config.data_dir)
        for row in store.list_sessions():
            print(f"{row['id']}  {row.get('title') or ''}  turns={row.get('turns', 0)}  {row.get('updated_at', '')}")
        return 0

    if not config.api_key and not args.once:
        print(
            Style.yellow(
                "未检测到 LLM_API_KEY。复制 .env.example 为 .env 并填入 key；"
                "离线测试请运行：python -m unittest discover -s tests -t ."
            ),
            file=sys.stderr,
        )

    cli = ChatCLI(
        config,
        session_id=args.session,
        show_thinking=not args.hide_thinking,
        show_trace=args.trace,
    )

    if args.new_session:
        cli.cmd_new(args.new_session)

    if args.once:
        result = cli.ask(args.once)
        cli.close()
        return 0 if result.ok else 1

    cli.repl()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
