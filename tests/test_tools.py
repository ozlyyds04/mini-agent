"""工具层单元测试：覆盖 CONTRACT.md 第 1 节列出的 7 组用例。

会话用本文件里的最小假 Session（不 import agent.session，避免与队友的并行开发耦合），
测试全部离线、零第三方依赖。
"""

from __future__ import annotations

import unittest
from typing import Any

from agent.errors import ToolValidationError
from agent.tools import calculator, read_docs, search, todo, weather
from agent.tools.base import Tool, ToolContext
from agent.tools.registry import builtin_registry

REPO_ROOT = read_docs._REPO_ROOT


# ---------------------------------------------------------------- 假 Session
class FakeTodo:
    """最小 TodoItem 替身：id / content / done。"""

    def __init__(self, todo_id: int, content: str, done: bool = False) -> None:
        self.id = todo_id
        self.content = content
        self.done = done


class FakeSession:
    """最小 Session 替身：只实现 todo 契约涉及的方法与属性。"""

    def __init__(self, session_id: str = "s1") -> None:
        self.id = session_id
        self.todos: list[FakeTodo] = []
        self._next_id = 1

    def add_todo(self, content: str) -> FakeTodo:
        item = FakeTodo(self._next_id, content)
        self._next_id += 1
        self.todos.append(item)
        return item

    def done_todo(self, todo_id: int) -> FakeTodo | None:
        for item in self.todos:
            if item.id == todo_id:
                item.done = True
                return item
        return None

    def clear_todos(self, only_done: bool = False) -> int:
        before = len(self.todos)
        if only_done:
            self.todos = [item for item in self.todos if not item.done]
        else:
            self.todos.clear()
        return before - len(self.todos)

    def render_todos(self) -> str:
        if not self.todos:
            return "（无）"
        return "\n".join(f"#{item.id} [{'x' if item.done else ' '}] {item.content}" for item in self.todos)


def invoke(tool_obj: Tool, ctx: ToolContext | None = None, **kwargs: Any):
    """模拟 executor：先按 schema 校验/强制类型，再调用工具。"""
    return tool_obj(ctx, **tool_obj.normalize_arguments(kwargs))


def make_ctx(session_id: str, title: str = "") -> tuple[FakeSession, ToolContext]:
    session = FakeSession(session_id)
    _ = title
    return session, ToolContext(session=session, session_id=session_id)


REGISTRY = builtin_registry()
CALCULATOR = REGISTRY.get("calculator")
SEARCH = REGISTRY.get("search")
TODO = REGISTRY.get("todo")
READ_DOCS = REGISTRY.get("read_docs")
WEATHER = REGISTRY.get("weather")


# ---------------------------------------------------------------- 1. 注册表
class TestRegistry(unittest.TestCase):
    def test_builtin_registry_has_five_tools(self) -> None:
        self.assertEqual(len(REGISTRY), 5)
        self.assertEqual(
            set(REGISTRY.names()),
            {"calculator", "search", "todo", "read_docs", "weather"},
        )

    def test_every_module_exposes_module_level_tool(self) -> None:
        pairs = [
            (calculator, "calculator"),
            (search, "search"),
            (todo, "todo"),
            (read_docs, "read_docs"),
            (weather, "weather"),
        ]
        for module, expected_name in pairs:
            self.assertIsInstance(module.TOOL, Tool, module.__name__)
            self.assertEqual(module.TOOL.name, expected_name)

    def test_schemas_are_valid_openai_format(self) -> None:
        schemas = REGISTRY.schemas()
        self.assertEqual(len(schemas), 5)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            function = schema["function"]
            self.assertEqual(set(function), {"name", "description", "parameters"})
            self.assertTrue(function["description"].strip())
            params = function["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertTrue(params.get("required"), f"{function['name']} 缺少 required")
            self.assertIsInstance(params["properties"], dict)


# ---------------------------------------------------------------- 2. schema 细节
class TestSchemaDetails(unittest.TestCase):
    def test_search_top_k_is_integer_with_default(self) -> None:
        props = SEARCH.parameters["properties"]
        self.assertEqual(props["top_k"]["type"], "integer")
        self.assertEqual(props["top_k"]["default"], 3)
        self.assertIn("required", SEARCH.parameters)
        self.assertEqual(SEARCH.parameters["required"], ["query"])

    def test_todo_action_has_enum_and_ctx_is_not_exposed(self) -> None:
        props = TODO.parameters["properties"]
        self.assertEqual(props["action"]["type"], "string")
        self.assertEqual(props["action"]["enum"], ["add", "list", "done", "clear"])
        self.assertNotIn("ctx", props)
        self.assertTrue(TODO.needs_context)

    def test_every_parameter_has_generated_description(self) -> None:
        for tool_obj in REGISTRY:
            for name, spec in tool_obj.parameters["properties"].items():
                self.assertTrue(
                    spec.get("description"),
                    f"{tool_obj.name}.{name} 的 schema 缺少 description（docstring Args 未解析）",
                )
            self.assertTrue(tool_obj.description.strip())


# ---------------------------------------------------------------- 3. 参数校验
class TestNormalizeArguments(unittest.TestCase):
    def test_missing_required_raises(self) -> None:
        with self.assertRaises(ToolValidationError):
            CALCULATOR.normalize_arguments({})
        with self.assertRaises(ToolValidationError):
            WEATHER.normalize_arguments({})

    def test_unknown_argument_raises(self) -> None:
        with self.assertRaises(ToolValidationError):
            WEATHER.normalize_arguments({"city": "广州", "bogus": 1})

    def test_string_number_is_coerced_to_int(self) -> None:
        args = SEARCH.normalize_arguments({"query": "agent", "top_k": "3"})
        self.assertEqual(args["top_k"], 3)
        args = TODO.normalize_arguments({"action": "done", "id": "2"})
        self.assertEqual(args["id"], 2)
        self.assertIsInstance(args["id"], int)

    def test_invalid_enum_raises(self) -> None:
        with self.assertRaises(ToolValidationError):
            TODO.normalize_arguments({"action": "rename"})


# ---------------------------------------------------------------- 4. calculator
class TestCalculator(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        result = invoke(CALCULATOR, expression="(1 + 2) * 3")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.content, "(1 + 2) * 3 = 9")

    def test_whitelisted_functions(self) -> None:
        self.assertEqual(invoke(CALCULATOR, expression="sqrt(16)").content, "sqrt(16) = 4")
        self.assertEqual(invoke(CALCULATOR, expression="max(3, 7)").content, "max(3, 7) = 7")
        self.assertIn("= 3.14", invoke(CALCULATOR, expression="round(3.14159, 2)").content)

    def test_division_by_zero_is_failure(self) -> None:
        for expression in ("1 / 0", "1 // 0", "1 % 0"):
            result = invoke(CALCULATOR, expression=expression)
            self.assertFalse(result.ok, expression)
            self.assertIn("除数", result.error)
            self.assertEqual(result.content, "")

    def test_import_injection_is_blocked_not_executed(self) -> None:
        result = invoke(CALCULATOR, expression="__import__('os').system('echo hacked')")
        self.assertFalse(result.ok)
        # 属性调用在解析阶段就撞上白名单守卫，报错信息提示只允许白名单函数调用
        self.assertIn("不支持", result.error)
        self.assertEqual(result.content, "")

    def test_attribute_access_is_blocked(self) -> None:
        result = invoke(CALCULATOR, expression="().__class__.__mro__")
        self.assertFalse(result.ok)
        self.assertIn("不支持", result.error)

    def test_oversized_result_is_failure(self) -> None:
        result = invoke(CALCULATOR, expression="10 ** 500")
        self.assertFalse(result.ok)
        self.assertIn("指数过大", result.error)

    def test_syntax_error_is_failure(self) -> None:
        result = invoke(CALCULATOR, expression="1 +")
        self.assertFalse(result.ok)
        self.assertIn("语法错误", result.error)
        self.assertFalse(invoke(CALCULATOR, expression="   ").ok)


# ---------------------------------------------------------------- 5. todo 隔离
class TestTodoSessionIsolation(unittest.TestCase):
    def test_two_sessions_are_isolated(self) -> None:
        session_a, ctx_a = make_ctx("session-a")
        session_b, ctx_b = make_ctx("session-b")

        added = invoke(TODO, ctx_a, action="add", content="买牛奶")
        self.assertTrue(added.ok, added.error)
        self.assertEqual(len(session_a.todos), 1)
        self.assertEqual(session_b.todos, [])

        self.assertEqual(invoke(TODO, ctx_b, action="list").content, "当前没有待办。")
        self.assertIn("#1 [ ] 买牛奶", invoke(TODO, ctx_a, action="list").content)

        # B 没有 #1，done 必须失败；A 的 done 只改 A
        self.assertFalse(invoke(TODO, ctx_b, action="done", id=1).ok)
        self.assertTrue(invoke(TODO, ctx_a, action="done", id=1).ok)
        self.assertTrue(session_a.todos[0].done)
        self.assertEqual(session_b.todos, [])
        self.assertEqual(session_b._next_id, 1)

    def test_add_requires_content(self) -> None:
        _, ctx = make_ctx("s-add")
        result = invoke(TODO, ctx, action="add", content="   ")
        self.assertFalse(result.ok)
        self.assertIn("content", result.error)

    def test_done_unknown_id_fails(self) -> None:
        _, ctx = make_ctx("s-done")
        invoke(TODO, ctx, action="add", content="写周报")
        result = invoke(TODO, ctx, action="done", id=99)
        self.assertFalse(result.ok)
        self.assertIn("99", result.error)

    def test_clear_only_clears_own_session(self) -> None:
        session_a, ctx_a = make_ctx("s-clear-a")
        session_b, ctx_b = make_ctx("s-clear-b")
        invoke(TODO, ctx_a, action="add", content="A1")
        invoke(TODO, ctx_a, action="add", content="A2")
        invoke(TODO, ctx_b, action="add", content="B1")

        cleared = invoke(TODO, ctx_a, action="clear")
        self.assertTrue(cleared.ok)
        self.assertIn("2", cleared.content)
        self.assertEqual(session_a.todos, [])
        self.assertEqual([item.content for item in session_b.todos], ["B1"])

    def test_missing_session_is_failure(self) -> None:
        result = invoke(TODO, None, action="list")
        self.assertFalse(result.ok)
        self.assertIn("会话", result.error)


# ---------------------------------------------------------------- 6. read_docs
class TestReadDocs(unittest.TestCase):
    def test_path_traversal_is_rejected(self) -> None:
        attacks = [
            "../../etc/passwd",
            "docs/../../etc/passwd",
            "../agent/config.py",
            "..\\..\\Windows\\win.ini",
            str(REPO_ROOT.parent.parent / "outside.txt"),
        ]
        for path in attacks:
            result = invoke(READ_DOCS, path=path)
            self.assertFalse(result.ok, f"路径穿越未被拒绝: {path}")
            self.assertIn("越界", result.error)

    def test_read_existing_file_and_truncate(self) -> None:
        result = invoke(READ_DOCS, path="docs/CONTRACT.md", max_chars=120)
        self.assertTrue(result.ok, result.error)
        self.assertIn("docs/CONTRACT.md", result.content)
        self.assertIn("已截断", result.content)
        self.assertLessEqual(len(result.content), 120 + 40)
        self.assertFalse(invoke(READ_DOCS, path="docs/CONTRACT.md", max_chars=50_000).content.endswith("字符]"))

    def test_missing_file_lists_available(self) -> None:
        result = invoke(READ_DOCS, path="docs/不存在的文档.md")
        self.assertFalse(result.ok)
        self.assertIn("文件不存在", result.error)
        self.assertIn("CONTRACT.md", result.error)

    def test_keyword_search_returns_snippet(self) -> None:
        result = invoke(READ_DOCS, path="session 隔离")
        self.assertTrue(result.ok, result.error)
        self.assertIn("隔离", result.content)
        self.assertIn("docs/", result.content)


# ---------------------------------------------------------------- 7. search / weather
class TestSearchAndWeather(unittest.TestCase):
    def test_search_hit_returns_ranked_blocks(self) -> None:
        result = invoke(SEARCH, query="session 隔离", top_k=2)
        self.assertTrue(result.ok, result.error)
        self.assertIn("[1]", result.content)
        self.assertIn("隔离", result.content)
        self.assertNotIn("[3]", result.content)

    def test_search_no_result_is_success(self) -> None:
        result = invoke(SEARCH, query="量子纠缠zzz")
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "没有找到相关内容。")

    def test_search_missing_query_fails(self) -> None:
        self.assertFalse(invoke(SEARCH, query="   ").ok)

    def test_weather_known_city_today_and_tomorrow(self) -> None:
        today = invoke(WEATHER, city="广州")
        self.assertTrue(today.ok, today.error)
        self.assertIn("广州", today.content)
        self.assertIn("℃", today.content)
        tomorrow = invoke(WEATHER, city="厦门", date="tomorrow")
        self.assertTrue(tomorrow.ok, tomorrow.error)
        self.assertIn("明天", tomorrow.content)

    def test_weather_explicit_date(self) -> None:
        result = invoke(WEATHER, city="北京", date="2026-10-22")
        self.assertTrue(result.ok, result.error)
        self.assertIn("2026-10-22", result.content)

    def test_weather_unknown_city_fails_and_lists_supported(self) -> None:
        result = invoke(WEATHER, city="火星")
        self.assertFalse(result.ok)
        self.assertIn("广州", result.error)
        self.assertFalse(invoke(WEATHER, city="  ").ok)

    def test_weather_bad_date_fails(self) -> None:
        result = invoke(WEATHER, city="上海", date="2026-13-99")
        self.assertFalse(result.ok)
        self.assertIn("YYYY-MM-DD", result.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
