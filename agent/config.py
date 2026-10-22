"""集中配置：全部来自环境变量（可选读取仓库根目录的 .env）。

设计取舍
--------
1. 不引入 pydantic / python-dotenv —— 零第三方依赖是这个项目的卖点之一。
2. **只对接 DeepSeek**。能力差异（思考过程是否回填、tool_choice、strict schema）
   不散落在业务代码里，而是收敛成一个 :data:`DEEPSEEK` 画像常量：
   上层只问"能不能"，不写 ``if 厂商``。
3. 所有阈值（压缩水位、max_steps、tool 结果截断）都可配，方便现场演示。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 当前唯一支持的厂商。保留常量是为了让"能力画像"这个概念有落点，
#: 而不是把厂商差异硬编码进 builder / client。
PROVIDER_NAME = "deepseek"


# ---------------------------------------------------------------- 模型画像
@dataclass(frozen=True)
class ModelProfile:
    """DeepSeek 的能力画像（依据官方文档 + 实测）。"""

    name: str
    base_url: str
    default_model: str
    #: 是否把上一轮的 reasoning_content 回填给下一轮请求。
    #: DeepSeek 的推理模型**不接受**回填（会报错），所以固定为 False。
    backfill_thinking: bool = False


DEEPSEEK = ModelProfile(
    name=PROVIDER_NAME,
    base_url="https://api.deepseek.com/v1",
    default_model="deepseek-chat",
    backfill_thinking=False,
)


def _load_env_file(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE / # 注释 / 允许引号。已存在的环境变量优先。"""
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip().strip('"').strip("'")
        # 支持 ${OTHER_VAR} 引用与已存在的环境变量优先
        if value.startswith("${") and value.endswith("}"):
            value = os.environ.get(value[2:-1], "")
        if key not in os.environ:
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

    #: 仅用于展示与追踪；DeepSeek 之外的值会被拒绝（见 from_env）
    provider: str = PROVIDER_NAME
    base_url: str = DEEPSEEK.base_url
    api_key: str = ""
    model: str = DEEPSEEK.default_model
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

    # -- 模型能力 -----------------------------------------------------
    @property
    def profile(self) -> ModelProfile:
        return DEEPSEEK

    # -- 构造 ---------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: Path | None = None, require_api_key: bool = True) -> "Config":
        _load_env_file(env_file if env_file is not None else REPO_ROOT / ".env")

        # 只接受 deepseek（留出 LLM_PROVIDER 是为了让旧配置报出清晰错误，
        # 而不是静默连到错误的地址）
        provider = _env_str("LLM_PROVIDER", PROVIDER_NAME).lower() or PROVIDER_NAME
        if provider != PROVIDER_NAME:
            raise ConfigError(
                f"本项目只支持 DeepSeek，但检测到 LLM_PROVIDER={provider!r}。"
                f"请删除该变量或设为 {PROVIDER_NAME!r}。"
            )

        # 兼容两种 key 变量名：LLM_API_KEY / DEEPSEEK_API_KEY
        api_key = _env_str("LLM_API_KEY") or _env_str("DEEPSEEK_API_KEY")
        base_url = (
            _env_str("LLM_BASE_URL") or _env_str("DEEPSEEK_BASE_URL") or DEEPSEEK.base_url
        ).rstrip("/")

        cfg = cls(
            provider=PROVIDER_NAME,
            base_url=base_url,
            api_key=api_key,
            model=_env_str("LLM_MODEL", DEEPSEEK.default_model),
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

        if require_api_key and not cfg.api_key:
            raise ConfigError(
                "缺少 API key。复制 .env.example 为 .env，填入 LLM_API_KEY（或 DEEPSEEK_API_KEY）；"
                "离线测试不需要 key（python -m unittest discover -s tests -t .）。"
            )
        return cfg

    def apply_overrides(self, **kwargs: object) -> "Config":
        """测试 / CLI 用的轻量覆盖（只接受已知字段）。"""
        known = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in known:
                raise ConfigError(f"unknown config field: {key}")
            setattr(self, key, value)
        return self
