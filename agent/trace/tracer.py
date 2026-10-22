"""可观测层：trace / span / JSONL 落盘。

为什么自己写而不是接 OpenTelemetry：这是一个零依赖项目，且 Agent 的 trace
需要记录的是"业务语义"（这一步是 LLM 调用还是工具调用、花了多少 token、
context 水位多少、有没有触发压缩），而不是通用 RPC 链路。结构简单反而好用。

一次 run 一个 trace_id；每次 LLM 调用 / 工具调用 / context 组装都是一个 span。
落盘格式：``traces/<日期>.jsonl``，每行一条记录，可直接 grep。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def now_iso() -> str:
    """本地时间 + 时区，精确到毫秒。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass
class Span:
    """一个执行片段（LLM 调用 / 工具调用 / context 组装 / 压缩）。"""

    kind: str
    seq: int
    started_at: str
    fields: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str = ""
    status: str = "ok"


class Tracer:
    """按 run 分组的 trace 记录器；``trace_dir=None`` 时只留内存（测试用）。"""

    def __init__(
        self,
        trace_dir: Path | str | None = None,
        run_id: str | None = None,
        echo: bool = False,
    ) -> None:
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.echo = echo
        self._events: list[dict[str, Any]] = []
        self._spans: list[Span] = []
        self._seq = 0
        self._handle = None
        self._closed = False
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    # -- 基础 ---------------------------------------------------------
    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _write(self, record: dict[str, Any]) -> None:
        record.setdefault("run_id", self.run_id)
        record.setdefault("ts", now_iso())
        self._events.append(record)
        if self.trace_dir is None:
            return
        try:
            if self._handle is None:
                day = datetime.now().strftime("%Y-%m-%d")
                self._handle = (self.trace_dir / f"{day}.jsonl").open("a", encoding="utf-8")
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
        except OSError:  # 落盘失败不能影响主流程
            self._handle = None

    # -- 事件 / span ---------------------------------------------------
    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        """记录一条离散事件（run_start / tool_error / retry ...）。"""
        record = {"type": "event", "kind": kind, "seq": self._next_seq(), **fields}
        self._write(record)
        return record

    @contextmanager
    def span(self, kind: str, **fields: Any) -> Iterator[Span]:
        """记录一段耗时操作；异常会被记录后原样抛出。"""
        span = Span(kind=kind, seq=self._next_seq(), started_at=now_iso(), fields=dict(fields))
        self._spans.append(span)
        self._write({"type": "span_start", "kind": kind, "seq": span.seq, **fields})
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - 记录后重新抛出
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - started) * 1000
            self._write(
                {
                    "type": "span_end",
                    "kind": kind,
                    "seq": span.seq,
                    "status": span.status,
                    "duration_ms": round(span.duration_ms, 2),
                    "error": span.error,
                    **(span.fields if span.kind != "llm.request" else {}),
                }
            )

    # -- 派生 ---------------------------------------------------------
    def child(self) -> "Tracer":
        """同一 trace 文件下开一个新的 run（一次用户请求 = 一个 run）。"""
        return Tracer(trace_dir=self.trace_dir, echo=self.echo)

    def summary(self) -> dict[str, Any]:
        """给测试/CLI 用的聚合视图。"""
        kinds: dict[str, int] = {}
        errors: list[str] = []
        llm_ms = 0.0
        tool_ms = 0.0
        for span in self._spans:
            kinds[span.kind] = kinds.get(span.kind, 0) + 1
            if span.error:
                errors.append(f"{span.kind}: {span.error}")
            if span.kind == "llm.request":
                llm_ms += span.duration_ms
            elif span.kind == "tool.call":
                tool_ms += span.duration_ms
        return {
            "run_id": self.run_id,
            "spans": len(self._spans),
            "kinds": kinds,
            "errors": errors,
            "llm_ms": round(llm_ms, 2),
            "tool_ms": round(tool_ms, 2),
        }

    def close(self) -> None:
        if self._handle is not None and not self._closed:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        self._closed = True
