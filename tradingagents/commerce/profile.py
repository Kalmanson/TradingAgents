"""Build a reproducible, bounded product request from two public inputs."""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from tradingagents.application.runner import ANALYST_ORDER, AnalysisRequest
from tradingagents.commerce.config import LANGUAGES
from tradingagents.dataflows.symbol_utils import normalize_symbol


def validate_inputs(ticker: str, language: str) -> str:
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?", ticker):
        raise ValueError("Enter a US stock ticker, such as RXRX or BRK-B.")
    ticker = ticker.replace(".", "-")
    if normalize_symbol(ticker) != ticker:
        raise ValueError("This ticker resolves to an unsupported instrument.")
    if language not in LANGUAGES:
        raise ValueError("Choose one of the available report languages.")
    return ticker


def validate_us_equity(ticker: str) -> None:
    # Deliberately fail closed before charging: the graph's identity resolver
    # is fail-open and therefore cannot serve as a purchase eligibility check.
    import yfinance as yf

    from tradingagents.commerce.creem import ProviderUnavailable

    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        raise ProviderUnavailable("Market data is temporarily unavailable.") from None
    if not info:
        raise ProviderUnavailable("We could not verify this stock. Please try again later.")
    if info.get("quoteType") != "EQUITY" or info.get("exchange") not in {
        "NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS",
    }:
        raise ValueError("The first release supports stocks listed on US exchanges.")


def build_profile(ticker: str, language: str, base_config: dict) -> dict:
    config = deepcopy(base_config)
    backend = config.get("backend_url")
    if backend:
        url = urlsplit(backend)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Model endpoint credentials must use environment variables, not URLs")
    request = AnalysisRequest(
        ticker=ticker,
        analysis_date=datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
        analysts=ANALYST_ORDER,
        research_depth=1,
        llm_provider=config["llm_provider"],
        quick_think_llm=config["quick_think_llm"],
        deep_think_llm=config["deep_think_llm"],
        backend_url=backend,
        output_language=LANGUAGES[language][0],
        checkpoint_enabled=False,
        google_thinking_level=config.get("google_thinking_level"),
        openai_reasoning_effort=config.get("openai_reasoning_effort"),
        anthropic_effort=config.get("anthropic_effort"),
    )
    request.validate()
    # Allowlist only non-secret configuration needed to reproduce a purchase.
    keys = ("data_vendors", "tool_vendors", "temperature", "llm_max_retries",
            "max_tokens", "max_recur_limit", "news_article_limit",
            "global_news_article_limit", "global_news_lookback_days", "global_news_queries")
    runtime_config = {key: config[key] for key in keys if key in config}
    for key, default in (("llm_max_retries", 2), ("max_tokens", 8192)):
        if config.get(key) is None:
            runtime_config[key] = default
    return {"version": 1, "request": request.to_dict(), "config": runtime_config}
