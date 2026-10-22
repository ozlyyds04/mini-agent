"""LLM 输出解析测试：覆盖真实世界里遇到过的各种畸形输出。"""

from __future__ import annotations

import unittest

from agent.llm.parser import (
    StreamAccumulator,
    extract_text_tool_calls,
    extract_thinking,
    normalize_content,
    parse_chat_completion,
    parse_sse_line,
)


class TestThinking(unittest.TestCase):
    def test_reasoning_content(self) -> None:
        message = {"role": "assistant", "content": "答案", "reasoning_content": "先想一下"}
        self.assertEqual(extract_thinking(message), "先想一下")

    def test_alternative_field_names(self) -> None:
        self.assertEqual(extract_thinking({"reasoning": "r1"}), "r1")
        self.assertEqual(extract_thinking({"thinking": "t1"}), "t1")
        self.assertEqual(extract_thinking({"reasoning_text": "r2"}), "r2")

    def test_block_style(self) -> None:
        message = {"thinking": [{"text": "第一步"}, {"text": "第二步"}]}
        self.assertEqual(extract_thinking(message), "第一步第二步")

    def test_missing(self) -> None:
        self.assertEqual(extract_thinking({"content": "hi"}), "")


class TestContent(unittest.TestCase):
    def test_string(self) -> None:
        self.assertEqual(normalize_content("hello"), "hello")

    def test_none(self) -> None:
        self.assertEqual(normalize_content(None), "")

    def test_block_array_skips_thinking(self) -> None:
        content = [
            {"type": "thinking", "text": "内部推理"},
            {"type": "text", "text": "可见答案"},
        ]
        self.assertEqual(normalize_content(content), "可见答案")


class TestParseCompletion(unittest.TestCase):
    def test_plain_text(self) -> None:
        raw = {
            "model": "deepseek-chat",
            "choices": [
                {"message": {"role": "assistant", "content": "你好"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        response = parse_chat_completion(raw)
        self.assertEqual(response.content, "你好")
        self.assertFalse(response.has_tool_calls)
        self.assertEqual(response.usage["total_tokens"], 5)
        self.assertEqual(response.finish_reason, "stop")

    def test_tool_calls(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression": "1+1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        response = parse_chat_completion(raw)
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0].name, "calculator")
        self.assertEqual(response.tool_calls[0].arguments, {"expression": "1+1"})

    def test_responses_with_error_payload(self) -> None:
        # 正常响应里带 reasoning_content（DeepSeek 推理模型）
        raw = {
            "choices": [
                {
                    "message": {
                        "content": "4",
                        "reasoning_content": "1+3=4",
                        "tool_calls": [],
                    }
                }
            ]
        }
        response = parse_chat_completion(raw)
        self.assertEqual(response.thinking, "1+3=4")
        self.assertEqual(response.content, "4")


class TestDirtyArguments(unittest.TestCase):
    def test_truncated_arguments_do_not_crash(self) -> None:
        from agent.llm.types import ToolCall

        call = ToolCall.from_openai(
            {"id": "c1", "function": {"name": "todo", "arguments": '{"action": "ad'}}
        )
        self.assertEqual(call.name, "todo")
        self.assertEqual(call.arguments, {})
        self.assertIn("action", call.raw_arguments)

    def test_fenced_arguments(self) -> None:
        from agent.llm.types import ToolCall

        call = ToolCall.from_openai(
            {
                "id": "c1",
                "function": {
                    "name": "todo",
                    "arguments": '```json\n{"action": "list"}\n```',
                },
            }
        )
        self.assertEqual(call.arguments, {"action": "list"})

    def test_single_quotes_and_trailing_comma(self) -> None:
        from agent.llm.types import ToolCall

        call = ToolCall.from_openai(
            {"id": "c1", "function": {"name": "todo", "arguments": "{'action': 'list',}"}}
        )
        self.assertEqual(call.arguments, {"action": "list"})


class TestTextFallback(unittest.TestCase):
    """弱 function calling 模型会把工具调用写进 content。"""

    def test_fenced_tool_call(self) -> None:
        content = '我来算一下：\n```json\n{"name": "calculator", "arguments": {"expression": "2*3"}}\n```'
        calls = extract_text_tool_calls(content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "calculator")
        self.assertEqual(calls[0].arguments, {"expression": "2*3"})

    def test_bare_json_with_prose(self) -> None:
        content = '好的，调用工具 {"tool": "weather", "args": {"city": "广州"}} 完成'
        calls = extract_text_tool_calls(content)
        self.assertEqual(calls[0].name, "weather")
        self.assertEqual(calls[0].arguments, {"city": "广州"})

    def test_arguments_as_string(self) -> None:
        content = '{"name": "todo", "arguments": "{\\"action\\": \\"list\\"}"}'
        calls = extract_text_tool_calls(content)
        self.assertEqual(calls[0].arguments, {"action": "list"})

    def test_no_tool_call(self) -> None:
        self.assertEqual(extract_text_tool_calls("今天天气不错"), [])

    def test_plain_json_without_name_is_ignored(self) -> None:
        self.assertEqual(extract_text_tool_calls('{"foo": 1}'), [])


class TestStreamAccumulator(unittest.TestCase):
    def _chunks(self):
        """模拟一段真实的流式输出：文本 + 分片的工具调用参数。"""
        return [
            {"model": "deepseek-chat", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "我"}}]},
            {"choices": [{"delta": {"content": "来算"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "calculator", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"expr'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ession": "1+'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '1"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]

    def test_assembles_fragmented_arguments(self) -> None:
        accumulator = StreamAccumulator()
        visible = []
        for chunk in self._chunks():
            text = accumulator.feed(chunk)
            if text:
                visible.append(text)
        response = accumulator.finalize()
        self.assertEqual("".join(visible), "我来算")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "calculator")
        # 关键：参数被拆成 3 段，必须拼完才 parse
        self.assertEqual(response.tool_calls[0].arguments, {"expression": "1+1"})
        self.assertEqual(response.tool_calls[0].id, "call_abc")
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_two_tool_calls_by_index(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "id": "b", "function": {"name": "weather", "arguments": '{"city":"厦门"}'}}
                            ]
                        }
                    }
                ]
            }
        )
        accumulator.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "a", "function": {"name": "todo", "arguments": '{"action":"list"}'}}
                            ]
                        }
                    }
                ]
            }
        )
        response = accumulator.finalize()
        self.assertEqual([c.name for c in response.tool_calls], ["todo", "weather"])

    def test_thinking_deltas(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed({"choices": [{"delta": {"reasoning_content": "先思考"}}]})
        accumulator.feed({"choices": [{"delta": {"content": "再回答"}}]})
        response = accumulator.finalize()
        self.assertEqual(response.thinking, "先思考")
        self.assertEqual(response.content, "再回答")

    def test_usage_captured(self) -> None:
        accumulator = StreamAccumulator()
        accumulator.feed({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})
        self.assertEqual(accumulator.finalize().usage["total_tokens"], 10)


class TestSSE(unittest.TestCase):
    def test_data_line(self) -> None:
        self.assertEqual(parse_sse_line('data: {"a":1}'), {"a": 1})

    def test_done(self) -> None:
        self.assertIsNone(parse_sse_line("data: [DONE]"))

    def test_comment_or_blank(self) -> None:
        self.assertIsNone(parse_sse_line(": keep-alive"))
        self.assertIsNone(parse_sse_line(""))

    def test_broken_json(self) -> None:
        self.assertIsNone(parse_sse_line("data: {not json"))


if __name__ == "__main__":
    unittest.main()
