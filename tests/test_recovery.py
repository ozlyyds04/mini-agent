"""异常处理测试：LLM 重试、不可重试错误、缺 key、网络抖动。

不联网：直接替换 client 的请求方法，验证重试策略本身。
"""

from __future__ import annotations

import unittest
from unittest import mock

from agent.config import Config
from agent.errors import LLMError, LLMHTTPError, LLMRetryExhausted
from agent.llm.client import LLMClient
from agent.llm.types import LLMResponse


def make_config(**overrides) -> Config:
    config = Config(provider="deepseek", api_key="test-key", model="deepseek-chat")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class ScriptedClient(LLMClient):
    """按脚本返回结果或抛异常，并记录调用次数。"""

    def __init__(self, outcomes, config: Config | None = None) -> None:
        super().__init__(config or make_config())
        self.outcomes = list(outcomes)
        self.attempts = 0

    def _request_once(self, messages, tools):  # type: ignore[override]
        self.attempts += 1
        outcome = self.outcomes.pop(0) if self.outcomes else LLMResponse(content="默认")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestRetry(unittest.TestCase):
    def test_retry_on_429_then_success(self) -> None:
        client = ScriptedClient([LLMHTTPError(429, "rate limited"), LLMResponse(content="ok")])
        with mock.patch("agent.llm.client.time.sleep") as sleep:
            response = client.chat([], stream=False)
        self.assertEqual(response.content, "ok")
        self.assertEqual(client.attempts, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_retry_on_500(self) -> None:
        client = ScriptedClient([LLMHTTPError(500, "boom"), LLMHTTPError(503, "busy"), LLMResponse(content="ok")])
        with mock.patch("agent.llm.client.time.sleep"):
            response = client.chat([], stream=False)
        self.assertEqual(response.content, "ok")
        self.assertEqual(client.attempts, 3)

    def test_no_retry_on_400(self) -> None:
        client = ScriptedClient([LLMHTTPError(400, "bad request")])
        with mock.patch("agent.llm.client.time.sleep") as sleep:
            with self.assertRaises(LLMHTTPError):
                client.chat([], stream=False)
        self.assertEqual(client.attempts, 1)
        self.assertEqual(sleep.call_count, 0)

    def test_exhausted_raises(self) -> None:
        client = ScriptedClient([LLMHTTPError(500, "x"), LLMHTTPError(500, "x"), LLMHTTPError(500, "x")])
        client.config.max_retries = 3
        with mock.patch("agent.llm.client.time.sleep"):
            with self.assertRaises(LLMHTTPError):
                client.chat([], stream=False)
        self.assertEqual(client.attempts, 3)

    def test_backoff_grows(self) -> None:
        client = ScriptedClient(
            [LLMHTTPError(500, "x"), LLMHTTPError(500, "x"), LLMResponse(content="ok")]
        )
        delays: list[float] = []
        with mock.patch("agent.llm.client.time.sleep", side_effect=delays.append):
            client.chat([], stream=False)
        self.assertEqual(len(delays), 2)
        # 退避时间递增（含 jitter，但量级必须上升）
        self.assertGreater(delays[1], delays[0] * 0.9)

    def test_missing_api_key_gives_actionable_error(self) -> None:
        client = LLMClient(make_config(api_key=""))
        with self.assertRaises(LLMError) as ctx:
            client.chat([], stream=False)
        self.assertIn("LLM_API_KEY", str(ctx.exception))
        self.assertIn(".env", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
