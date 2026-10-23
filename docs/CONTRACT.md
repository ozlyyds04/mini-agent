# 内部接口契约（Lead 与并行开发者共同遵守）

> 目标：`agent/` 内各模块可以并行开发、互不阻塞。**任何人不得修改本文件之外的"别人负责的文件"**；
> 需要改接口时，先告诉 Lead，由 Lead 统一改并通知。
>
> 运行环境：Python 3.11+，**零第三方依赖**（只用标准库）。
> 测试命令：`python -m unittest discover -s tests -t . -v`（在仓库根目录执行）。

## 0. 已经写好的稳定层

| 文件 | 内容 |
|---|---|
| `agent/errors.py` | 异常体系：`ConfigError / LLMError / LLMHTTPError(retryable) / LLMRetryExhausted / LLMParseError / ToolError / ToolNotFound / ToolValidationError / MaxStepsReached / RunCancelled / SessionBusy` |
| `agent/config.py` | `Config`、`ProviderProfile`、`PROVIDERS`、`Config.from_env()` |
| `agent/llm/types.py` | `ToolCall`、`Message`、`LLMResponse` |
| `agent/tools/base.py` | `@tool` 装饰器、`Tool`、`ToolResult`、`ToolContext`、`build_parameters_schema()`、`parse_docstring()` |
| `agent/tools/registry.py` | `ToolRegistry`、`builtin_registry()` |
| `tests/fake_llm.py` | `FakeLLM`、`text_response()`、`tool_response()`、`FlakyLLM` |

### 关键数据结构（照抄签名即可，不要另造）

```python
# agent/llm/types.py
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = {}
    raw_arguments: str = ""

@dataclass
class Message:
    role: str                 # "system" | "user" | "assistant" | "tool"
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str = ""    # role=="tool" 时必填
    name: str = ""
    compressed: bool = False  # 已被压缩摘要覆盖，不再送给 LLM
    # 方法：to_openai(backfill_thinking=False) -> dict
    #       to_dict() / from_dict()
    #       Message.system() / .user() / .assistant() / .tool_result()

@dataclass
class LLMResponse:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = []
    usage: dict = {}
    model: str = ""
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw: dict | None = None
    # 属性：has_tool_calls / text
```

```python
# agent/tools/base.py
@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    error: str = ""
    meta: dict = {}
    # ToolResult.success(content, **meta) / ToolResult.failure(error, **meta)
    # .to_observation() -> 回喂模型的文本（失败时带修正建议）

@dataclass
class ToolContext:
    session: Any = None       # Session 实例
    session_id: str = ""
    data_dir: Path | None = None
    store: Any = None         # SessionStore
    runtime: Any = None       # 异步工具回投事件用

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # 自动生成的 JSON Schema
    func: Callable
    python_types: dict
    timeout_s: float = 10.0
    is_async: bool = False
    needs_context: bool = False
    # .schema() -> OpenAI tools 元素；.describe() -> 一行式清单
    # .normalize_arguments(args: dict) -> dict（校验+类型强制，失败抛 ToolValidationError）
    # .__call__(ctx, **kwargs) -> ToolResult
```

## 1. 任务 A：内置工具实现（负责目录 `agent/tools/`，除 base.py / registry.py）

**必须实现 5 个文件**，每个文件暴露一个模块级常量 `TOOL: Tool`：

```python
# agent/tools/calculator.py 结尾必须形如
TOOL = tool(
    name="calculator",
    description="计算数学表达式。支持 + - * / // % ** 与括号，以及 sqrt/sin/cos/abs 等函数。",
)(calculator)
```

### calculator.py
- 函数签名：`def calculator(expression: str) -> str`
- **安全求值**：用 `ast.parse(expression, mode="eval")` + 白名单递归求值，**禁止 `eval`/`exec`**。
  允许：`ast.Expression/BinOp/UnaryOp/Constant/Num`、运算符 `Add/Sub/Mult/Div/FloorDiv/Mod/Pow`、
  `ast.Call`（仅 `sqrt/abs/round/min/max/sin/cos/tan/log/log10/pow`，参数 1–2 个）。
- 异常（语法错、除零、未授权节点/函数、结果过大）→ 返回 `ToolResult.failure("...")`，**不要抛出去**。
  这是展示"工具失败也是观测"的关键点。
- 返回 `ToolResult.success(f"{expression} = {value}")`（结果保留合适精度，整数不显示 `.0`）。

### search.py
- 签名：`def search(query: str, top_k: int = 3) -> str`
- **mock 实现但接口是真的**：模块内维护 `CORPUS: list[dict]`（约 8–12 条中文文档，
  主题建议：Agent 循环/工具调用/session 隔离/context 压缩/memory 召回/异步工具/TTFT/OpenHands），
  每条含 `title`、`body`、`tags`。
- 打分：query 与 title/tags/body 的关键词命中数 + 字符重叠加权；中文按 2-gram 切分即可（不要引入分词库）。
- 返回前 `top_k` 条，格式：`[1] 标题\n摘要(截断 200 字)\n`。
- 顶部注释写清楚"替换 `_backend()` 即可接真实检索 API"，并留一个 `_backend()` 钩子（默认返回 None）。
- 无结果时返回 `ToolResult.success("没有找到相关内容。")`（不是失败）。

### todo.py（**题目强制要求，也是 session 隔离的证据**）
- 签名：`def todo(ctx, action: Literal["add","list","done","clear"], content: str = "", id: int = 0) -> str`
  —— 第一个参数名必须是 `ctx`，装饰器会自动识别并注入 `ToolContext`。
- 语义：
  - `add`：`ctx.session.add_todo(content)`，缺 `content` → `ToolResult.failure(...)`
  - `list`：列出全部待办，形如 `#1 [ ] 内容` / `#2 [x] 内容`；无待办返回 "当前没有待办。"
  - `done`：`ctx.session.done_todo(id)`，找不到 id → failure
  - `clear`：清空（返回清除数量）
- **只操作 `ctx.session.todos`，禁止任何模块级可变全局状态**。

### read_docs.py
- 签名：`def read_docs(path: str, max_chars: int = 2000) -> str`
- 从仓库根（`agent/tools/../../`）的 `docs/`、`artifacts/`、`README.md` 里读文件；
  **必须做路径穿越防护**：`resolve()` 后必须仍在允许的根目录内，否则 failure。
- 支持按标题/关键字检索：若 `path` 不以 `.md/.txt` 结尾，则当作关键词，在 `docs/*.md` 中检索并返回最相关片段
  （简单按段落打分即可）。文件不存在 → failure 并提示可用文件列表。
- 大文件截断到 `max_chars` 并在末尾标注 `...[已截断，共 N 字符]`。

### weather.py
- 签名：`def weather(city: str, date: str = "today") -> str`
- mock 一张城市天气表（至少 8 个城市：广州、厦门、北京、上海、深圳、杭州、成都、武汉），
  含 `condition/high/low/humidity`；`date` 支持 `today/tomorrow/YYYY-MM-DD`，未知城市 → failure
  并列出支持的城市。输出一行自然语言。

### 要求
- 每个工具函数都要有**中文 docstring**，第一段会被当成工具描述发给 LLM；参数用 `Args:` 段落写说明
  （`parse_docstring()` 会解析它填进 schema 的 `description`）。
- 类型注解必须写全，schema 由注解自动生成（这是评分点，别手写 schema）。
- 每个文件 60–120 行，不要超过 150 行。
- 工具执行时间 < 50ms；不得联网、不得写仓库外文件。

### 任务 A 的测试：`tests/test_tools.py`
至少覆盖：
1. `builtin_registry()` 能加载 5 个工具，`len == 5`，`schemas()` 是合法 OpenAI 格式且含 `required`。
2. `TOOL.schema()` 的 description 非空、参数类型正确（`top_k` 是 integer、`action` 有 enum）。
3. `normalize_arguments`：缺必填报 `ToolValidationError`、未知参数报错、`"3"` 能强制成 `3`。
4. calculator：正常四则运算、`sqrt(16)`、除零 failure、`__import__('os').system('echo')` 这类注入必须 failure 而不是执行。
5. todo：**两个不同 session 的 ToolContext 互不影响**（核心隔离测试）。
6. read_docs：路径穿越 `../../etc/passwd` 必须被拒绝。
7. weather：未知城市 failure；search：无结果不算失败。

## 2. 任务 B：Session 与 Context（负责 `agent/session/`、`agent/context/`）

### `agent/session/models.py`
```python
def now_iso() -> str          # 本地时间带时区，如 "2026-10-22T11:33:37+08:00"

@dataclass
class TodoItem:
    id: int
    content: str
    done: bool = False
    created_at: str = ""

@dataclass
class Session:
    id: str
    created_at: str = ""
    updated_at: str = ""
    title: str = ""
    messages: list[Message] = []
    todos: list[TodoItem] = []
    summary: str = ""          # 滚动摘要（压缩产物）
    summary_upto: int = 0      # messages[:summary_upto] 已被 summary 覆盖
    turn_count: int = 0
    state: str = "idle"        # idle | busy | waiting_tool | paused
    meta: dict = {}
    lock: threading.RLock = ...   # 不序列化

    # 必须提供
    def append(self, message: Message) -> Message      # 追加 + 更新 updated_at；role=="user" 时 turn_count += 1
    def touch(self) -> None
    def next_todo_id(self) -> int
    def add_todo(self, content: str) -> TodoItem
    def done_todo(self, todo_id: int) -> TodoItem | None
    def clear_todos(self, only_done: bool = False) -> int
    def pending_todos(self) -> list[TodoItem]
    def render_todos(self) -> str        # "#1 [ ] 内容"; 空时 "（无）"
    def render_state_block(self) -> str  # 给 system prompt 的结构化状态：轮次 + 待办
    def visible_messages(self) -> list[Message]   # messages[summary_upto:]
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, data: dict) -> "Session"
```

### `agent/session/store.py`
文件布局：`<data_dir>/sessions/<session_id>.jsonl`，第一行 meta，其后每行一个事件。

```python
class SessionStore:
    def __init__(self, root: Path)      # root 是仓库根/数据根；内部用 root/"sessions"
    def create(self, title: str = "", session_id: str | None = None) -> Session
        # 默认 id 形如 "s20261022-a1b2c3"（日期+随机后缀），并立刻写 meta 行
    def exists(self, session_id: str) -> bool
    def load(self, session_id: str) -> Session
        # 逐行 replay：meta / message / todos / summary / note
        # 坏行必须跳过，并把数量记进 session.meta["corrupt_lines"]
    def list_sessions(self) -> list[dict]
        # 只读每文件首行 meta，返回 [{id, title, created_at, updated_at, turns, state}]
        # 按 updated_at 倒序
    def append_message(self, session: Session, message: Message) -> None
    def save_todos(self, session: Session) -> None      # 写一份完整快照行
    def save_summary(self, session: Session) -> None    # 写摘要行（含 summary_upto）
    def append_note(self, session: Session, kind: str, payload: dict) -> None
    def delete(self, session_id: str) -> None
    def lock(self, session_id: str) -> threading.RLock  # 同 id 返回同一把锁（进程内）
```

JSONL 行格式（必须严格一致，方便人肉排查）：
```json
{"type":"meta","session":{"id":"...","title":"...","created_at":"...","updated_at":"...","state":"idle","turn_count":0}}
{"type":"message","ts":"...","message":{"role":"user","content":"..."}}
{"type":"todos","ts":"...","todos":[{"id":1,"content":"...","done":false,"created_at":"..."}]}
{"type":"summary","ts":"...","summary":"...","summary_upto":12}
{"type":"note","ts":"...","kind":"tool_trace","payload":{...}}
```

### `agent/context/budget.py`
```python
def estimate_tokens(text: str) -> int
    # 经验公式：CJK 字符 ≈1 token/字；ASCII ≈1 token/4 字符；标点按 1/2 计。
    # 目的不是精确，是"同一坐标系下比较大小"，文档里要写明这是估算。
def estimate_messages_tokens(messages: list[dict] | list[Message]) -> int

@dataclass
class ContextBudget:
    total: int
    def used(self, tokens: int) -> float   # 返回占用比例
    def level(self, ratio: float) -> int   # 0/1/2/3 对应四档水位（用 Config 的 compress_*_ratio）
```

### `agent/context/builder.py`
```python
class ContextBuilder:
    def __init__(self, config: Config, store: SessionStore | None = None)
    def build(self, session: Session, tools_description: str = "", extra_system: str = "") -> list[dict]
```
组装顺序（**必须**）：
1. system 消息，内容包含：角色设定、**当前本地时间**、工具清单（`tools_description`）、
   `session.render_state_block()`、若有摘要则附 `## 较早对话的摘要\n{session.summary}`。
2. 近期消息：`session.visible_messages()` 里最靠后的若干轮（受 `config.keep_recent_turns` 与 token 预算约束）。
3. **保持 tool_call / tool_result 配对**：assistant 消息带 `tool_calls` 时，其所有 `tool_call.id`
   都必须能在随后的 `role=="tool"` 消息里找到；否则丢弃这条 assistant 消息（宁缺勿错配，
   否则部分厂商直接 400）。
4. **工具结果截断**：`role=="tool"` 且 `len(content) > config.tool_result_max_chars` 时，
   保留前 60% + 后 20% 并插入 `\n...[已截断 N 字符]...\n`（这是 L0 压缩）。
5. 丢弃 `compressed=True` 的消息。
6. 最后做一次**硬预算**：估算 token 超过 `config.context_budget_tokens * config.compress_l3_ratio`
   时，从最旧的非 system 消息开始丢，但**永远保留最后一条 user 消息**。
7. 返回 OpenAI 格式 dict 列表，使用 `Message.to_openai(backfill_thinking=config.profile.backfill_thinking)`。

### `agent/context/compressor.py`
```python
@dataclass
class CompressionReport:
    level: int = 0
    before_tokens: int = 0
    after_tokens: int = 0
    folded_messages: int = 0
    ok: bool = True
    note: str = ""

class Compressor:
    def __init__(self, config: Config, llm=None, store: SessionStore | None = None)
    def maybe_compress(self, session: Session) -> CompressionReport
```
分层策略（**必须先便宜后贵**）：
- 用 `ContextBuilder`（或直接估算）得到当前 token 占 `config.context_budget_tokens` 的比例。
- `ratio < compress_l1_ratio(0.60)` → 只做 L0（tool 结果截断，可以在 builder 里做，这里返回 level=0 且不调 LLM）。
- `0.60 ≤ ratio < 0.80` → **L1**：把 `messages[summary_upto : len - keep_recent]` 折叠成摘要，
  与已有 `session.summary` 合并，更新 `summary_upto`；被折叠的消息标记 `compressed=True`；
  **原文归档**到 `<data_dir>/artifacts/<session_id>/archived-<n>.jsonl`；调 `store.save_summary()`。
- `0.80 ≤ ratio < 0.92` → **L2**：递归合并（把已有摘要 + 新折叠区间一起重写），产物更短（≤600 字）。
- `≥ 0.92` → **L3**：不再调 LLM，直接硬裁剪（保留最近的 `keep_recent_turns` 与 pinned 事实）。
- **摘要调用失败/超时/返回空 → 不抛异常**，`report.ok=False`、`note=原因`，并由 builder 的硬预算兜底。
- 摘要 prompt 用固定模板，要求输出这几段（缺段也要能工作）：
  `【用户目标】【已完成的动作与结论】【关键事实与ID】【未完成/待确认】【当前待办】`，
  并硬性要求**不超过 400 字**。
- pinned 事实：`session.meta.get("pinned", [])`（字符串列表）永远不进摘要、也不被裁剪；
  另外**第一条 user 消息**视为 pinned。

### 任务 B 的测试：`tests/test_session.py` + `tests/test_context.py`
至少覆盖：
1. 两个 session 的 todo / messages / summary 完全隔离。
2. `create → append → save_todos → load` 后状态一致；坏行被跳过且 `corrupt_lines>0`。
3. `list_sessions()` 按 updated_at 倒序。
4. `build()` 输出第一条是 system，且包含当前时间、工具清单、待办状态。
5. **带工具追问**：assistant(tool_calls) + tool 结果成对保留；故意删掉一条 tool 结果时，
   对应 assistant 消息被丢弃（不产生 400 风险）。
6. 超长 tool 结果被截断，且截断标记存在。
7. 压缩：构造 20 轮对话使 `maybe_compress` 触发 L1，`summary` 非空、`summary_upto>0`、
   `compressed=True` 的消息不再出现在 `build()` 结果里；**压缩后仍能回答早期目标**（用 FakeLLM 验证摘要被放进 system）。
8. 压缩失败回退：`FakeLLM` 抛异常时 `report.ok is False` 且不抛出。

## 3. 环境与风格约定

- 一律 `from __future__ import annotations`；类型注解写全。
- **不要**引入任何第三方包；**不要**用 `eval`。
- 文件编码 UTF-8；注释与 docstring 用中文，代码标识符用英文。
- 日志不要 `print`，需要可观测性时用 `logging.getLogger(__name__)`。
- 时间统一用 `now_iso()` 产生的带时区 ISO 字符串。
- 测试用 `unittest`（不要 pytest），临时目录用 `tempfile.TemporaryDirectory()`。
- 测试里构造配置：`Config(provider="deepseek", data_dir=Path(tmp), context_budget_tokens=500)`，
  不要读 `.env`、不要联网。
