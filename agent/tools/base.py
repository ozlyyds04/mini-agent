"""工具层基础：``@tool`` 装饰器、JSON Schema 自动生成、统一结果结构。

设计要点
--------
1. **Schema 从函数签名 + docstring 自动生成**，不手写。手写 schema 与实现漂移
   是 Agent 项目最常见的 bug 来源（模型按旧 schema 传参，工具静默出错）。
2. **失败也是结果**：工具抛异常不打断主循环，而是变成 ``ToolResult(ok=False)``
   回喂给模型，让它自己修正参数重试（见 runtime/executor.py）。
3. **会话级状态通过 ToolContext 注入**（如 todo），工具模块里禁止全局可变状态，
   这样多个 session 天然隔离。
"""

from __future__ import annotations

import inspect
import re
import types as _types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ..errors import ToolValidationError

# JSON Schema 基础类型映射
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}

#: 不进入 schema 的注入参数
_INJECTED_PARAMS = {"ctx", "self", "cls"}

_SECTION_HEADS = {
    "args:",
    "arguments:",
    "parameters:",
    "params:",
    "returns:",
    "return:",
    "raises:",
    "yields:",
    "examples:",
    "example:",
    "note:",
    "notes:",
}


# ---------------------------------------------------------------- ToolContext
@dataclass
class ToolContext:
    """工具执行时的会话上下文（由 executor 注入）。

    工具通过它拿到"当前是哪个会话"，因此 todo 之类的工具不需要任何全局状态。
    """

    session: Any = None          # session.models.Session
    session_id: str = ""
    data_dir: Path | None = None
    store: Any = None            # session.store.SessionStore（可选）
    runtime: Any = None          # 供异步工具回投事件（可选）

    def artifact_dir(self) -> Path:
        base = Path(self.data_dir) if self.data_dir else Path.cwd()
        path = base / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------- ToolResult
@dataclass
class ToolResult:
    """工具执行结果。成功与失败走同一条通道，避免异常逃逸打断循环。"""

    ok: bool
    content: str = ""
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, content: str, **meta: Any) -> "ToolResult":
        return cls(ok=True, content=content, meta=dict(meta))

    @classmethod
    def failure(cls, error: str, **meta: Any) -> "ToolResult":
        return cls(ok=False, content="", error=error, meta=dict(meta))

    def to_observation(self) -> str:
        """转成回喂给模型的文本。

        失败时**必须**给出可操作的修正建议，否则模型只会原样重试。
        """
        if self.ok:
            return self.content
        return (
            f"[tool_error] {self.error}\n"
            "请检查参数是否符合工具 schema，修正后重新调用；不要重复同样的调用。"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "content": self.content, "error": self.error, "meta": self.meta}


# ---------------------------------------------------------------- Schema 生成
def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    """把 Python 类型注解翻译成 JSON Schema 片段（够用即可，不追求完备）。"""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        values = list(args)
        is_str = all(isinstance(v, str) for v in values)
        return {"type": "string" if is_str else "integer", "enum": values}

    # Optional[X] / X | None（typing.Union 与 PEP 604 的 types.UnionType 都要处理）
    if origin is typing.Union or origin is _types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _schema_for_annotation(non_none[0])
        return {"anyOf": [_schema_for_annotation(a) for a in non_none]}

    if origin in (list, typing.List):
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _schema_for_annotation(args[0])
        return schema

    if origin in (dict, typing.Dict):
        return {"type": "object"}

    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}

    # 未识别的注解退化成 string（保证 schema 总是可用的）
    return {"type": "string"}


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """从 docstring 里拆出 (工具描述, 参数描述)。

    支持 Google 风格的 ``Args:`` / ``Parameters:`` 段落，例如::

        管理当前会话的待办清单。

        Args:
            action: 操作类型，add 表示新增
            content: 待办内容
    """
    doc = inspect.cleandoc(doc or "")
    if not doc.strip():
        return "", {}

    lines = doc.splitlines()

    # 1) 第一段作为工具描述（遇到空行或段落标题就停）
    desc_parts: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in _SECTION_HEADS:
            break
        if stripped:
            desc_parts.append(stripped)
        elif desc_parts:
            break
    description = " ".join(desc_parts)

    # 2) 解析 Args 段落
    param_docs: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered in ("args:", "arguments:", "parameters:", "params:"):
            in_args = True
            current = None
            continue
        if in_args and lowered in _SECTION_HEADS - {
            "args:",
            "arguments:",
            "parameters:",
            "params:",
        }:
            in_args = False
            current = None
            continue
        if not in_args:
            continue
        match = re.match(r"^([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
        if match:
            current = match.group(1)
            param_docs[current] = match.group(2).strip()
        elif current and stripped:
            # 参数的续行
            param_docs[current] = (param_docs[current] + " " + stripped).strip()

    return description, param_docs


def build_parameters_schema(fn: Callable[..., Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """从函数签名生成 JSON Schema，同时返回 python 注解表（用于类型强制）。"""
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # pragma: no cover - 极端情况下注解不可解析
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    python_types: dict[str, Any] = {}

    for name, param in signature.parameters.items():
        if name in _INJECTED_PARAMS:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, str)
        schema = _schema_for_annotation(annotation)
        properties[name] = schema
        python_types[name] = annotation
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = param.default

    # 参数描述来自 docstring
    _, param_docs = parse_docstring(getattr(fn, "__doc__", None))
    for name, doc in param_docs.items():
        if name in properties and doc:
            properties[name]["description"] = doc

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, python_types


# ---------------------------------------------------------------- 参数校验
def _coerce(value: Any, annotation: Any) -> Any:
    """把模型常见的"类型不对"参数尽量掰回来（"3" -> 3），失败则抛校验错误。"""
    try:
        if annotation is int and isinstance(value, str):
            return int(value)
        if annotation is float and isinstance(value, str):
            return float(value)
        if annotation is bool and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        if annotation is str and not isinstance(value, str):
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"参数 {value!r} 无法转换为 {annotation}: {exc}") from exc
    return value


@dataclass
class Tool:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    python_types: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 10.0
    #: 异步工具：调用立即返回 task_id，结果稍后以事件回投（见 runtime/async_tools.py）
    is_async: bool = False
    #: 是否需要注入 ToolContext（第一个参数名为 ctx 时自动识别）
    needs_context: bool = False

    # -- 供 LLM 使用 ---------------------------------------------------
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def describe(self) -> str:
        """给 system prompt 用的一行式工具清单。"""
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        params = ", ".join(
            f"{name}{'' if name in required else '?'}" for name in props
        )
        return f"- {self.name}({params}): {self.description}"

    # -- 参数处理 ------------------------------------------------------
    def normalize_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """校验必填、剔除未知参数、做轻度类型强制。"""
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        unknown = [k for k in arguments if k not in props]
        normalized: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in props:
                normalized[key] = _coerce(value, self.python_types.get(key))

        missing = [k for k in required if k not in normalized or normalized[k] is None]
        if missing:
            raise ToolValidationError(
                f"缺少必填参数: {', '.join(missing)}；"
                f"期望 schema: {_compact_schema(self.parameters)}"
            )
        if unknown:
            extra = ", ".join(unknown)
            raise ToolValidationError(
                f"存在未知参数: {extra}；可用参数: {', '.join(props)}"
            )

        # 枚举校验
        for key, value in normalized.items():
            enum = props[key].get("enum")
            if enum is not None and value not in enum:
                raise ToolValidationError(
                    f"参数 {key}={value!r} 不在枚举 {enum} 中"
                )
        return normalized

    def __call__(self, ctx: ToolContext | None, /, **kwargs: Any) -> ToolResult:
        if self.needs_context:
            return self.func(ctx, **kwargs)
        return self.func(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "is_async": self.is_async,
            "timeout_s": self.timeout_s,
        }


def _compact_schema(schema: dict[str, Any]) -> str:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    parts = []
    for name, spec in props.items():
        marker = "" if name in required else "?"
        parts.append(f"{name}{marker}:{spec.get('type', 'any')}")
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------- 装饰器
def tool(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_s: float = 10.0,
    is_async: bool = False,
) -> Any:
    """把普通函数声明成工具。

    用法::

        @tool(name="calculator", description="四则运算")
        def calculator(expression: str) -> str:
            ...

    工具函数应返回 ``str``（或可被 ``str()`` 的值）；需要会话状态时把第一个参数
    命名为 ``ctx``，executor 会自动注入 :class:`ToolContext`。
    """

    def decorate(fn: Callable[..., Any]) -> Tool:
        signature = inspect.signature(fn)
        needs_context = "ctx" in signature.parameters
        parameters, python_types = build_parameters_schema(fn)
        doc = inspect.getdoc(fn) or ""
        doc_description, _ = parse_docstring(doc)
        tool_name = name or fn.__name__
        tool_description = description or doc_description or tool_name
        return Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=fn,
            python_types=python_types,
            timeout_s=timeout_s,
            is_async=is_async,
            needs_context=needs_context,
        )

    if _fn is not None:
        return decorate(_fn)
    return decorate
