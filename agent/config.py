"""集中配置：全部来自环境变量（可选读取仓库根目录的 .env）。

设计取舍
--------
1. 不引入 pydantic / python-dotenv —— 零第三方依赖是这个项目的卖点之一。
2. 各厂商在"同一个 OpenAI-compatible 协议"下的真实差异，收敛进
   ``ProviderProfile``，上层只做能力探测，不写 ``if provider == "glm"`` 散落各处。
3. 所有阈值（压缩水位、max_steps、tool 结果截断）都可配，方便面试时现场演示。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- Provider
@dataclass(frozen=True)
class ProviderProfile:
    """一个 provider 的能力画像（由官方文档/实测确认，见 docs/system-design.md「Provider 差异」）。"""

    name: str
    base_url: str
    default_model: str
    #: 是否把上一轮的 thinking / reasoning_content 原样回填给下一轮请求。
    #: 注意各家的要求是**相反**的：GLM 交错思考要求回传，DeepSeek 推理模型不接受回传。
    backfill_thinking: bool = False
    #: 是否支持一次返回多个 tool_call（并行工具调用）。
    supports_parallel_tool_calls: bool = True
    #: tool_choice 是否支持 required / 指定函数（GLM 只支持 auto）。
    supports_tool_choice: bool = True
    #: 是否支持 strict JSON Schema 约束（多数国内厂商不支持）。
    supports_strict_schema: bool = False


PROVIDERS: dict[str, ProviderProfile] = {
    "deepseek": ProviderProfile(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        backfill_thinking=False,
    ),
    "glm": ProviderProfile(
        name="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        # 交错思考（thinking + tool_calls 混合）时必须把 reasoning_content 原样回传，
        # 且 clear_thinking=false 才保留历史思考。
        backfill_thinking=True,
        supports_tool_choice=False,
    ),
    "openai": ProviderProfile(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        supports_strict_schema=True,
    ),
    "qwen": ProviderProfile(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
    ),
    "ark": ProviderProfile(
        name="ark",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-seed-1-6-250615",
    ),
}


def _load_env_file(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE / # 注释 / 允许引号。已存在的环境变量优先。"""
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    """运行期配置。用 :func:`Config.from_env` 构造，测试里直接构造即可。"""

    provider: str = "deepseek"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    stream: bool = True

    # 网络
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 60.0
    max_retries: int = 3

    # 循环
    max_steps: int = 8          # 单次 run 内最多几轮工具往返
    max_turns: int = 200        # 单个 session 累计轮次上限（软提醒 + 硬上限）
    keep_recent_turns: int = 6  # context 里保留的完整轮数
    #: 是否并行执行同一轮内的多个工具。默认关闭：工具可能共享 session 状态
    #: （如 todo），串行执行结果确定、可复现；并行留给无状态工具集。
    parallel_tool_calls: bool = False

    # context 预算与压缩水位
    context_budget_tokens: int = 8000
    compress_l1_ratio: float = 0.60
    compress_l2_ratio: float = 0.80
    compress_l3_ratio: float = 0.92
    tool_result_max_chars: int = 800

    # 存储
    data_dir: Path = field(default_factory=lambda: REPO_ROOT)

    # 加分项
    enable_async_tools: bool = True
    async_tool_delay_s: float = 0.05   # mock 异步工具的模拟耗时
    #: 是否启用"按需记忆召回"（只在出现指代/时间词时才翻历史）
    enable_memory_recall: bool = True

    # 派生字段（from_env 填充）
    _profile: ProviderProfile | None = None

    # -- 目录 ---------------------------------------------------------
    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    # -- provider -----------------------------------------------------
    @property
    def profile(self) -> ProviderProfile:
        if self._profile is not None:
            return self._profile
        profile = PROVIDERS.get(self.provider)
        if profile is None:
            raise ConfigError(
                f"unknown LLM_PROVIDER={self.provider!r}; "
                f"choose from {', '.join(sorted(PROVIDERS))}"
            )
        return profile

    # -- 构造 ---------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: Path | None = None, require_api_key: bool = True) -> "Config":
        _load_env_file(env_file if env_file is not None else REPO_ROOT / ".env")

        provider = _env_str("LLM_PROVIDER", "deepseek").lower()
        if provider not in PROVIDERS:
            raise ConfigError(
                f"unknown LLM_PROVIDER={provider!r}; choose from {', '.join(sorted(PROVIDERS))}"
            )
        profile = PROVIDERS[provider]

        cfg = cls(
            provider=provider,
            base_url=_env_str("LLM_BASE_URL", profile.base_url).rstrip("/"),
            api_key=_env_str("LLM_API_KEY"),
            model=_env_str("LLM_MODEL", profile.default_model),
            temperature=_env_float("LLM_TEMPERATURE", 0.3),
            max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
            stream=_env_bool("LLM_STREAM", True),
            connect_timeout_s=_env_float("LLM_CONNECT_TIMEOUT", 10.0),
            request_timeout_s=_env_float("LLM_TIMEOUT", 60.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 3),
            max_steps=_env_int("MINI_AGENT_MAX_STEPS", 8),
            max_turns=_env_int("MINI_AGENT_MAX_TURNS", 200),
            keep_recent_turns=_env_int("MINI_AGENT_KEEP_TURNS", 6),
            parallel_tool_calls=_env_bool("MINI_AGENT_PARALLEL_TOOLS", False),
            context_budget_tokens=_env_int("MINI_AGENT_CONTEXT_BUDGET", 8000),
            tool_result_max_chars=_env_int("MINI_AGENT_TOOL_RESULT_CHARS", 800),
            enable_async_tools=_env_bool("MINI_AGENT_ASYNC_TOOLS", True),
            enable_memory_recall=_env_bool("MINI_AGENT_MEMORY_RECALL", True),
            data_dir=Path(_env_str("MINI_AGENT_DATA_DIR", str(REPO_ROOT))),
        )
        cfg._profile = profile

        if require_api_key and not cfg.api_key:
            raise ConfigError(
                "缺少 LLM_API_KEY。复制 .env.example 为 .env 并填入 key；"
                "离线测试不需要 key（python -m unittest discover -s tests -t .）。"
            )
        return cfg

    def apply_overrides(self, **kwargs: object) -> "Config":
        """测试/CLI 用的轻量覆盖（只接受已知字段）。"""
        known = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in known:
                raise ConfigError(f"unknown config field: {key}")
            setattr(self, key, value)
        return self
