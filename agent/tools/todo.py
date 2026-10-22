"""待办工具：状态全部存在 ``ctx.session.todos`` 里，模块级没有任何可变全局状态。

这是"会话隔离"的直接证据：工具函数第一个参数叫 ``ctx``，装饰器识别后由执行器注入
当前会话的 ``ToolContext``；两个 session 各自调用同一份工具代码，互不可见。
"""

from __future__ import annotations

from typing import Literal

from .base import ToolContext, ToolResult, tool


def _render_todos(session: object) -> str:
    """优先用 Session 自己的 render_todos()，兼容只有 todos 列表的最小实现。"""
    render = getattr(session, "render_todos", None)
    if callable(render):
        text = str(render())
        return "当前没有待办。" if text.strip() in ("", "（无）") else text
    todos = getattr(session, "todos", [])
    if not todos:
        return "当前没有待办。"
    return "\n".join(
        f"#{item.id} [{'x' if getattr(item, 'done', False) else ' '}] {item.content}"
        for item in todos
    )


def todo(
    ctx: ToolContext,
    action: Literal["add", "list", "done", "clear"],
    content: str = "",
    id: int = 0,
) -> ToolResult:
    """管理当前会话的待办清单（add 新增 / list 查看 / done 完成 / clear 清空）。

    待办只保存在当前 session 内，不同会话互不影响。

    Args:
        action: 操作类型：add 新增、list 列出全部、done 完成指定 id、clear 清空。
        content: 待办内容，action=add 时必填。
        id: 待办编号，action=done 时必填（来自 list 输出中的 #编号）。
    """
    session = getattr(ctx, "session", None)
    if session is None:
        return ToolResult.failure("当前没有会话上下文，无法操作待办（请先创建 session）。")

    if action == "add":
        text = (content or "").strip()
        if not text:
            return ToolResult.failure("add 操作需要提供 content，例如 content='买牛奶'。")
        item = session.add_todo(text)
        item_id = getattr(item, "id", "?")
        return ToolResult.success(f"已添加待办 #{item_id}：{text}\n{_render_todos(session)}")

    if action == "list":
        return ToolResult.success(_render_todos(session))

    if action == "done":
        if not isinstance(id, int) or isinstance(id, bool) or id <= 0:
            return ToolResult.failure("done 操作需要提供正整数 id，例如 id=1。")
        item = session.done_todo(id)
        if item is None:
            return ToolResult.failure(f"没有找到编号 #{id} 的待办；先用 action=list 查看现有编号。")
        return ToolResult.success(f"已完成待办 #{id}：{item.content}\n{_render_todos(session)}")

    if action == "clear":
        removed = session.clear_todos()
        return ToolResult.success(f"已清除 {removed} 条待办。")

    return ToolResult.failure(
        f"未知 action={action!r}；只支持 add / list / done / clear。"
    )


TOOL = tool(
    name="todo",
    description="管理当前会话的待办清单：add 新增、list 列出、done 完成、clear 清空。",
)(todo)
