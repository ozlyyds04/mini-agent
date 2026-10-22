"""工具注册表：名称 / 描述 / 参数 Schema 的唯一来源。

主循环不认识任何具体工具名，只认 registry —— 这让"新增一个工具"变成
"新增一个文件 + 注册一行"，而不是去改 if/elif。
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .base import Tool


class ToolRegistry:
    """线程内共享的工具表（不可变语义优先：注册后不鼓励动态增删）。"""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.register(item)

    # -- 注册 ---------------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise TypeError(f"expect Tool instance, got {type(tool)!r}")
        if not tool.name:
            raise ValueError("tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def register_many(self, tools: Iterable[Tool]) -> "ToolRegistry":
        for item in tools:
            self.register(item)
        return self

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # -- 查询 ---------------------------------------------------------
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    # -- 给 LLM 的视图 -------------------------------------------------
    def schemas(self) -> list[dict]:
        """OpenAI-compatible 的 tools 数组。"""
        return [tool.schema() for tool in self._tools.values()]

    def describe(self) -> str:
        """system prompt 里的人类可读工具清单。"""
        return "\n".join(tool.describe() for tool in self._tools.values())


def builtin_registry() -> ToolRegistry:
    """装载全部内置工具（5 个：calculator / search / todo / read_docs / weather）。"""
    from . import calculator, read_docs, search, todo, weather

    registry = ToolRegistry()
    for module in (calculator, search, todo, read_docs, weather):
        registry.register(module.TOOL)
    return registry
