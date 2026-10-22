"""HTTP 客户端：urllib 手写 chat/completions，带重试、超时、流式。

为什么不直接用 openai SDK：题目要求"从零实现"，且本项目**零第三方依赖**，
clone 下来就能跑是最重要的工程指标。手写一遍也更清楚协议长什么样。
"""

from __future__ import annotations

import json
import logging
import random
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from ..config import Config
from ..errors import LLMError, LLMHTTPError, LLMParseError, LLMRetryExhausted
from ..trace.tracer import Tracer
from .adapters import Adapter, build_adapter
from .parser import StreamAccumulator, parse_chat_completion, iter_sse_chunks
from .types import LLMResponse

logger = logging.getLogger(__name__)

#: 流式回调：收到一段可见文本时调用（CLI 用它做打字机效果）
DeltaCallback = Callable[[str], None]


class LLMClient:
    """对 LLM 的唯一入口。"""

    def __init__(
        self,
        config: Config,
        adapter: Adapter | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or build_adapter(config)
        self.tracer = tracer

    # ---------------------------------------------------------------- 公开
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        """调用模型，自动重试可恢复错误。

        :param on_delta: 流式回调；给了就一定能收到增量文本（非流式时一次性回调）。
        """
        use_stream = self.config.stream if stream is None else stream
        # 没有 API key 时给出明确错误，而不是等 401 让人猜
        if not self.config.api_key:
            raise LLMError(
                "缺少 LLM_API_KEY：请在 .env 或环境变量中配置（离线测试请用 FakeLLM）。"
            )

        last_error: Exception | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(1, attempts + 1):
            try:
                if use_stream:
                    response = self._request_stream(messages, tools, on_delta)
                else:
                    response = self._request_once(messages, tools)
                    if on_delta and response.content:
                        on_delta(response.content)
                return response
            except LLMHTTPError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts:
                    raise
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                last_error = LLMError(f"network error: {exc}")
                if attempt == attempts:
                    raise last_error from exc
            except LLMParseError as exc:
                # 解析失败通常是代理返回了非 JSON，重试一次有意义
                last_error = exc
                if attempt == attempts:
                    raise
            backoff = min(8.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.3)
            logger.warning("LLM 调用失败（第 %d/%d 次），%.2fs 后重试：%s", attempt, attempts, backoff, last_error)
            if self.tracer is not None:
                self.tracer.event(
                    "llm.retry",
                    attempt=attempt,
                    backoff_s=round(backoff, 3),
                    error=str(last_error)[:300],
                )
            time.sleep(backoff)

        raise LLMRetryExhausted(f"LLM 调用重试 {attempts} 次后仍失败：{last_error}")

    # ---------------------------------------------------------------- 内部
    @property
    def _endpoint(self) -> str:
        base = (self.config.base_url or self.adapter.profile.base_url).rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _build_request(self, payload: dict[str, Any]) -> urllib.request.Request:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            self._endpoint,
            data=body,
            headers=self.adapter.headers(),
            method="POST",
        )

    def _timeout(self) -> float:
        return float(self.config.request_timeout_s)

    def _request_once(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        payload = self.adapter.payload(messages, tools, stream=False)
        payload["stream"] = False
        payload.pop("stream_options", None)
        request = self._build_request(payload)

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout()) as response:
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            # 模型名不被该账号/网关支持是最常见的 400，给出可直接执行的下一步
            if exc.code in (400, 404) and "model" in detail.lower():
                detail += (
                    f"\n提示：当前 LLM_MODEL={self.config.model!r} 可能不被该 key 支持。"
                    "运行 `python scripts/list_models.py` 查看可用模型，再改 .env 的 LLM_MODEL。"
                )
            raise LLMHTTPError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"network error: {exc.reason}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMParseError(f"响应不是合法 JSON：{body[:200]!r}") from exc

        if isinstance(raw, dict) and raw.get("error"):
            message = raw["error"].get("message") if isinstance(raw["error"], dict) else raw["error"]
            raise LLMError(f"LLM 返回错误：{message}")

        result = self.adapter.parse(raw)
        result.latency_ms = latency_ms
        self._trace_response(result, stream=False)
        return result

    def _request_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_delta: DeltaCallback | None,
    ) -> LLMResponse:
        payload = self.adapter.payload(messages, tools, stream=True)
        request = self._build_request(payload)
        accumulator = StreamAccumulator()
        started = time.perf_counter()

        try:
            with urllib.request.urlopen(request, timeout=self._timeout()) as response:
                for line in response:
                    for chunk in iter_sse_chunks([line]):
                        delta = accumulator.feed(chunk)
                        if delta and on_delta:
                            on_delta(delta)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            # 模型名不被该账号/网关支持是最常见的 400，给出可直接执行的下一步
            if exc.code in (400, 404) and "model" in detail.lower():
                detail += (
                    f"\n提示：当前 LLM_MODEL={self.config.model!r} 可能不被该 key 支持。"
                    "运行 `python scripts/list_models.py` 查看可用模型，再改 .env 的 LLM_MODEL。"
                )
            raise LLMHTTPError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"network error: {exc.reason}") from exc

        result = accumulator.finalize()
        result.latency_ms = (time.perf_counter() - started) * 1000
        self._trace_response(result, stream=True)
        return result

    def _trace_response(self, result: LLMResponse, stream: bool) -> None:
        if self.tracer is None:
            return
        self.tracer.event(
            "llm.response",
            model=result.model or self.config.model,
            stream=stream,
            latency_ms=round(result.latency_ms, 2),
            prompt_tokens=result.usage.get("prompt_tokens", 0),
            completion_tokens=result.usage.get("completion_tokens", 0),
            tool_calls=len(result.tool_calls),
            thinking_chars=len(result.thinking),
            finish_reason=result.finish_reason,
        )

    # ---------------------------------------------------------------- 关闭
    def close(self) -> None:  # urllib 无需显式关闭；保留接口方便替换成 http.client
        return None
