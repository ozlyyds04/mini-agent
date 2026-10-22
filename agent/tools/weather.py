"""天气查询工具：内置 mock 数据，不联网（演示"外部 API 替身"的接法）。

接真实天气 API 时只需替换 :func:`_lookup` 的实现（保持签名不变），
工具签名、错误语义与输出格式都不用动。
"""

from __future__ import annotations

from datetime import date as _date
from datetime import datetime as _datetime
from datetime import timedelta
from typing import Any

from .base import ToolResult, tool

#: mock 城市天气表；字段：condition / high / low / humidity
CITY_WEATHER: dict[str, dict[str, Any]] = {
    "广州": {"condition": "多云", "high": 30, "low": 23, "humidity": 72},
    "厦门": {"condition": "晴", "high": 28, "low": 22, "humidity": 68},
    "北京": {"condition": "晴", "high": 21, "low": 9, "humidity": 40},
    "上海": {"condition": "小雨", "high": 24, "low": 17, "humidity": 78},
    "深圳": {"condition": "雷阵雨", "high": 31, "low": 25, "humidity": 80},
    "杭州": {"condition": "阴", "high": 23, "low": 16, "humidity": 70},
    "成都": {"condition": "多云", "high": 22, "low": 15, "humidity": 75},
    "武汉": {"condition": "晴转多云", "high": 26, "low": 14, "humidity": 55},
}

_SUPPORTED = "、".join(CITY_WEATHER)


def _resolve_day(raw: str) -> tuple[_date, str]:
    """把 today / tomorrow / YYYY-MM-DD 解析成 (日期, 展示标签)。"""
    value = (raw or "today").strip().lower()
    if value in ("", "today", "今天"):
        return _date.today(), "今天"
    if value in ("tomorrow", "明天"):
        return _date.today() + timedelta(days=1), "明天"
    try:
        day = _datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"无法识别日期 {raw!r}，支持 today / tomorrow / YYYY-MM-DD。"
        ) from exc
    return day, day.isoformat()


def _lookup(city: str, day: _date) -> dict[str, Any] | None:
    """mock 查询：按与今天的天数差做确定性的小幅波动（真实实现换成 HTTP 请求）。"""
    base = CITY_WEATHER.get(city)
    if base is None:
        return None
    delta = (day - _date.today()).days
    high = base["high"] + delta % 3
    low = base["low"] + delta % 2
    humidity = min(95, max(20, base["humidity"] + delta))
    return {
        "condition": base["condition"],
        "high": high,
        "low": low,
        "humidity": humidity,
    }


def weather(city: str, date: str = "today") -> ToolResult:
    """查询指定城市某天的天气（内置 mock 数据，不联网）。

    未知城市或无法识别的日期会返回失败，并提示支持的城市/格式。

    Args:
        city: 城市名，支持：广州、厦门、北京、上海、深圳、杭州、成都、武汉。
        date: 日期，支持 today、tomorrow 或 YYYY-MM-DD，默认 today。
    """
    name = (city or "").strip()
    if not name:
        return ToolResult.failure(f"city 不能为空，目前支持：{_SUPPORTED}。")
    if name not in CITY_WEATHER:
        return ToolResult.failure(f"暂不支持城市 {name!r}，目前支持：{_SUPPORTED}。")
    try:
        day, label = _resolve_day(date)
    except ValueError as exc:
        return ToolResult.failure(str(exc))

    info = _lookup(name, day)
    if info is None:  # pragma: no cover - 上面已校验过城市
        return ToolResult.failure(f"暂不支持城市 {name!r}，目前支持：{_SUPPORTED}。")
    return ToolResult.success(
        f"{name}{label}（{day.isoformat()}）{info['condition']}，"
        f"{info['low']}~{info['high']}℃，湿度 {info['humidity']}%。",
        city=name,
        date=day.isoformat(),
    )


TOOL = tool(
    name="weather",
    description="查询城市天气（mock 数据）：支持广州、厦门、北京、上海、深圳、杭州、成都、武汉。",
)(weather)
