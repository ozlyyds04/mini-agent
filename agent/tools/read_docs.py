"""文档读取工具：只允许读仓库内 ``docs/``、``artifacts/`` 与 ``README.md``。

路径穿越是工具层的典型越权漏洞：先 ``resolve()`` 再做白名单父子校验，
``../../etc/passwd`` 这类输入直接拒绝；读不到文件时返回可用文件列表，方便模型自我纠正。
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import ToolResult, tool

#: 仓库根目录：agent/tools/../..
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCS_DIR = _REPO_ROOT / "docs"
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
_README = _REPO_ROOT / "README.md"
_TEXT_SUFFIXES = {".md", ".txt"}
_MAX_FILE_BYTES = 200_000
_CLEAN_RE = re.compile(r"[\W_]+", re.UNICODE)


def _safe_path(raw: str) -> Path:
    """解析成绝对路径；不在白名单内就抛 ValueError。"""
    candidate = Path(raw)
    target = (candidate if candidate.is_absolute() else _REPO_ROOT / candidate).resolve()
    if target != _README.resolve():
        roots = (_DOCS_DIR.resolve(), _ARTIFACTS_DIR.resolve())
        if not any(target == root or root in target.parents for root in roots):
            raise ValueError(
                f"路径越界：{raw!r} 只允许读取 docs/、artifacts/ 与 README.md 之内的文件。"
            )
    return target


def _available_files(limit: int = 12) -> list[str]:
    """列出现有可读文本文件（相对仓库根），用于拼"文件不存在"的提示。"""
    names: list[str] = []
    for base in (_DOCS_DIR, _ARTIFACTS_DIR):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
                names.append(path.relative_to(_REPO_ROOT).as_posix())
    if _README.is_file():
        names.append("README.md")
    return names[:limit]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...[已截断，共 {len(text)} 字符]"


def _grams(text: str) -> set[str]:
    clean = _CLEAN_RE.sub("", (text or "").lower())
    return {clean[i : i + 2] for i in range(len(clean) - 1)} if len(clean) > 1 else set(clean)


def _looks_like_path(raw: str) -> bool:
    """带分隔符/盘符/``..`` 的输入一律当路径处理，先进白名单校验再决定读文件。"""
    return "/" in raw or "\\" in raw or ":" in raw or raw.startswith(".") or raw.lower().endswith((".md", ".txt"))


def _search_paragraphs(keyword: str, max_chars: int) -> ToolResult:
    """在 docs/*.md 的段落里按 2-gram 重叠打分，返回最相关的一段。"""
    query_grams = _grams(keyword)
    best: tuple[float, Path, str] | None = None
    if _DOCS_DIR.is_dir():
        for path in sorted(_DOCS_DIR.rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for paragraph in re.split(r"\n\s*\n", text):
                para = paragraph.strip()
                score = float(len(query_grams & _grams(para)))
                if para and (best is None or score > best[0]):
                    best = (score, path, para)
    if best is None or best[0] <= 0.0:
        return ToolResult.success(f"没有找到与 {keyword!r} 相关的文档片段。")
    score, path, para = best
    relative = path.relative_to(_REPO_ROOT).as_posix()
    body = _truncate(para, max_chars)
    return ToolResult.success(f"在 {relative} 中检索 {keyword!r} 的最佳片段：\n\n{body}", source=relative, score=score)


def read_docs(path: str, max_chars: int = 2000) -> ToolResult:
    """读取仓库内的文档文件；path 不是文件路径时当作关键词在 docs 中检索。

    允许范围仅限 docs/、artifacts/ 与 README.md，路径穿越一律拒绝。

    Args:
        path: 相对仓库根的文件路径（如 ``docs/CONTRACT.md``），或检索关键词（如 ``session 隔离``）。
        max_chars: 返回内容最大字符数，超出会截断并标注总长度，默认 2000。
    """
    target_raw = (path or "").strip()
    if not target_raw:
        return ToolResult.failure("path 不能为空：请输入 docs/ 下的文件路径或检索关键词。")
    try:
        limit = max(1, min(int(max_chars), 20_000))
    except (TypeError, ValueError):
        return ToolResult.failure("max_chars 必须是整数，例如 2000。")

    if not _looks_like_path(target_raw):
        return _search_paragraphs(target_raw, limit)

    try:
        target = _safe_path(target_raw)
    except ValueError as exc:
        return ToolResult.failure(str(exc))

    if not target.is_file():
        available = "、".join(_available_files()) or "（当前没有可读文档）"
        return ToolResult.failure(f"文件不存在：{target_raw}。可用文件：{available}")

    try:
        size = target.stat().st_size
        if size > _MAX_FILE_BYTES:
            return ToolResult.failure(f"文件过大（{size} 字节，上限 {_MAX_FILE_BYTES}）。")
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult.failure(f"读取文件失败：{exc}")

    relative = target.relative_to(_REPO_ROOT).as_posix()
    return ToolResult.success(f"# {relative}\n\n{_truncate(text, limit)}", source=relative, chars=len(text))


TOOL = tool(
    name="read_docs",
    description="读取仓库 docs/、artifacts/ 与 README.md 中的文档；以关键词传入时在 docs 中检索片段。",
)(read_docs)
