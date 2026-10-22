"""计算器工具：用 ``ast`` 白名单求值，绝不使用 ``eval`` / ``exec``。

安全模型：表达式先 ``ast.parse(..., mode="eval")``，然后只允许"数字常量、四则运算、
白名单函数调用"三类节点递归求值；属性访问、下标、导入、lambda 等一律判失败。
失败同样经由 ``ToolResult.failure`` 回喂给模型，属于"工具失败也是观测"的示例。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

from .base import ToolResult, tool

#: 允许的二元运算符（**不含**位运算 / 比较 / 布尔运算）
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

#: 允许的一元运算符
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: 允许调用的函数名（只认裸名字，属性/下标调用直接拒绝）
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pow": pow,
}

_MAX_EXPRESSION_CHARS = 200
_MAX_ABS_VALUE = 1e100
_MAX_EXPONENT = 64


def _format_number(value: int | float) -> str:
    """整数不显示 ``.0``；浮点保留到 12 位有效数字，避免 0.30000000000000004。"""
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return f"{value:.12g}"


def _check_number(value: Any) -> int | float:
    """只接受有限实数，并拦住符号计算/溢出产生的非数字结果。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("只支持实数运算，结果不是数字")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("结果不是有限数字")
    if abs(value) > _MAX_ABS_VALUE:
        raise ValueError("结果过大（超过 1e100）")
    return value


def _eval(node: ast.AST) -> int | float:
    """白名单递归求值；任何未授权节点都抛 ValueError。"""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量 {node.value!r}，只允许数字")
        return _check_number(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符 {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"指数过大（绝对值上限 {_MAX_EXPONENT}）")
        return _check_number(op(left, right))
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ValueError(f"不支持的一元运算符 {type(node.op).__name__}")
        return _check_number(unary(_eval(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只支持直接调用白名单函数，不支持属性或下标调用")
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"未授权的函数 {node.func.id!r}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        if not 1 <= len(node.args) <= 2:
            raise ValueError("函数参数个数必须是 1 到 2 个")
        return _check_number(fn(*(_eval(arg) for arg in node.args)))
    raise ValueError(f"不支持的表达式节点 {type(node).__name__}")


def calculator(expression: str) -> ToolResult:
    """计算数学表达式，安全求值，不会执行任何代码。

    支持 + - * / // % ** 与括号，以及 sqrt/abs/round/min/max/sin/cos/tan/log/log10/pow
    等函数；语法错误、除零、未授权函数、结果过大都会返回失败而不是抛异常。

    Args:
        expression: 要计算的数学表达式，例如 ``(1 + 2) * 3`` 或 ``sqrt(16) + 2**3``。
    """
    expr = (expression or "").strip()
    if not expr:
        return ToolResult.failure("表达式为空，请传入形如 `1 + 2 * 3` 的数学表达式。")
    if len(expr) > _MAX_EXPRESSION_CHARS:
        return ToolResult.failure(f"表达式过长（上限 {_MAX_EXPRESSION_CHARS} 字符）。")
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except SyntaxError as exc:
        return ToolResult.failure(f"表达式语法错误：{exc.msg}；请检查括号与运算符。")
    except ZeroDivisionError:
        return ToolResult.failure("除数不能为 0，请修改表达式后重试。")
    except (ValueError, TypeError, OverflowError) as exc:
        return ToolResult.failure(f"无法计算表达式 {expr!r}：{exc}")
    except Exception as exc:  # 兜底：工具入口永远不向外抛异常
        return ToolResult.failure(f"计算失败：{exc}")
    return ToolResult.success(f"{expr} = {_format_number(value)}")


TOOL = tool(
    name="calculator",
    description="计算数学表达式。支持 + - * / // % ** 与括号，以及 sqrt/sin/cos/abs 等函数。",
)(calculator)
