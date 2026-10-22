"""Trace 层测试：span 记录、错误标记、落盘可解析。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.trace.tracer import Tracer


class TestTracer(unittest.TestCase):
    def test_memory_only(self) -> None:
        tracer = Tracer()
        tracer.event("run.start", session_id="s1")
        self.assertEqual(len(tracer.events), 1)
        self.assertEqual(tracer.events[0]["kind"], "run.start")
        tracer.close()

    def test_span_duration_and_seq(self) -> None:
        tracer = Tracer()
        with tracer.span("llm.request", step=1) as span:
            span.fields["tool_calls"] = 2
        self.assertGreaterEqual(span.duration_ms, 0)
        self.assertEqual(span.status, "ok")
        self.assertEqual(span.seq, 1)
        starts = [e for e in tracer.events if e["type"] == "span_start"]
        ends = [e for e in tracer.events if e["type"] == "span_end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["status"], "ok")
        tracer.close()

    def test_span_records_error_and_reraises(self) -> None:
        tracer = Tracer()
        with self.assertRaises(ValueError):
            with tracer.span("tool.call", tool="calculator"):
                raise ValueError("boom")
        errors = [s for s in tracer.spans if s.status == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("boom", errors[0].error)
        tracer.close()

    def test_summary(self) -> None:
        tracer = Tracer()
        with tracer.span("llm.request"):
            pass
        with tracer.span("tool.call", tool="search"):
            pass
        tracer.event("tool.error", tool="search")
        summary = tracer.summary()
        self.assertEqual(summary["spans"], 2)
        self.assertEqual(summary["kinds"]["llm.request"], 1)
        self.assertEqual(summary["kinds"]["tool.call"], 1)
        tracer.close()

    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracer = Tracer(trace_dir=Path(tmp))
            with tracer.span("llm.request", step=1):
                pass
            tracer.event("run.end", steps=1)
            tracer.close()

            files = list(Path(tmp).glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            lines = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line]
            self.assertTrue(any(line["type"] == "span_start" for line in lines))
            self.assertTrue(any(line["type"] == "event" and line["kind"] == "run.end" for line in lines))
            # 每行都带 run_id 与时间戳，方便 grep 定位一次请求
            self.assertTrue(all("run_id" in line and "ts" in line for line in lines))

    def test_child_is_new_run_same_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracer = Tracer(trace_dir=Path(tmp))
            child = tracer.child()
            self.assertNotEqual(child.run_id, tracer.run_id)
            self.assertEqual(child.trace_dir, tracer.trace_dir)
            tracer.close()
            child.close()


if __name__ == "__main__":
    unittest.main()
