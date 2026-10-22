"""列出当前 API key 可用的模型。

不同账号/网关开放的模型 ID 可能不同（例如 DeepSeek 官方文档写的是
``deepseek-chat`` / ``deepseek-reasoner``，而某些账号或中转网关只开放
``deepseek-flash`` / ``deepseek-v4-pro``）。填错模型名会得到 400，
所以先跑这个脚本确认，再写进 .env 的 ``LLM_MODEL``。

    python scripts/list_models.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import Config          # noqa: E402
from agent.errors import ConfigError     # noqa: E402


def main() -> int:
    try:
        config = Config.from_env(require_api_key=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    url = config.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {config.api_key}", "Accept": "application/json"},
    )
    print(f"GET {url}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    models = payload.get("data") or []
    if not models:
        print("（没有返回模型列表）")
        return 1

    print("\n可用模型：")
    for item in models:
        model_id = item.get("id") if isinstance(item, dict) else str(item)
        recommended = "  ← 当前 .env 使用" if model_id == config.model else ""
        print(f"  - {model_id}{recommended}")

    known = {item.get("id") for item in models if isinstance(item, dict)}
    print()
    if config.model not in known:
        print(f"⚠ 当前 LLM_MODEL={config.model!r} 不在可用列表里，改成上面任意一个。")
        return 1
    print(f"✓ 当前 LLM_MODEL={config.model!r} 可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
