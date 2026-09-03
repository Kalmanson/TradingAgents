"""Console entry point for the optional local web workbench."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            'Web 工作台未安装。请先运行：pip install "tradingagents[web]"'
        ) from exc

    app_path = Path(__file__).with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(streamlit_cli.main())
