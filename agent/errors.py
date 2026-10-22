"""统一的异常类型。

为什么要单独一层：Agent 的失败模式比普通服务多（模型超时、工具参数错、
工具抛异常、轮次超限），如果全部用 Exception 捕获，调用方无法区分
"可以自愈"和"必须中断"两类问题。这里按可恢复性分层。
"""

from __future__ import annotations


class MiniAgentError(Exception):
    """所有本库异常的基类。"""


class ConfigError(MiniAgentError):
    """配置缺失或非法（如没有 API key）。"""


# ---------------------------------------------------------------- LLM 相关
class LLMError(MiniAgentError):
    """LLM 调用失败的基类。"""


class LLMHTTPError(LLMError):
    """HTTP 层错误，带状态码。"""

    def __init__(self, status: int, body: str = "", message: str = "") -> None:
        super().__init__(message or f"LLM HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body

    @property
    def retryable(self) -> bool:
        """429 / 5xx 可以退避重试；4xx（除 429）重试没有意义。"""
        return self.status == 429 or self.status >= 500


class LLMRetryExhausted(LLMError):
    """重试次数用尽。"""


class LLMParseError(LLMError):
    """模型输出无法解析（非预期 JSON 等）。"""


# ---------------------------------------------------------------- 工具相关
class ToolError(MiniAgentError):
    """工具执行失败。通常会被 executor 捕获并转成观测结果回喂模型。"""


class ToolNotFound(ToolError):
    def __init__(self, name: str, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        super().__init__(
            f"unknown tool: {name!r}; available: {', '.join(self.available) or '(none)'}"
        )


class ToolValidationError(ToolError):
    """参数不符合工具 schema（缺必填、类型错、枚举越界）。"""


# ---------------------------------------------------------------- 运行时相关
class MaxStepsReached(MiniAgentError):
    """单次 run 内工具循环超过 max_steps。"""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        super().__init__(f"agent exceeded max_steps={steps}")


class RunCancelled(MiniAgentError):
    """协作式取消（用户打断 / 关闭会话）。"""


class SessionBusy(MiniAgentError):
    """同一 session 已有 run 在执行。"""
