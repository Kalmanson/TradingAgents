"""Build a reproducible, bounded product request from two public inputs."""

from __future__ import annotations

import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from tradingagents.application.runner import ANALYST_ORDER, AnalysisRequest
from tradingagents.commerce.config import LANGUAGES
from tradingagents.commerce.observability import log_event
from tradingagents.dataflows.market_data import get_instrument_info
from tradingagents.graph.analyst_execution import applicable_analysts

logger = logging.getLogger(__name__)


def validate_inputs(ticker: str, language: str) -> str:
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?", ticker):
        raise ValueError("Enter a valid US-listed ticker, such as AAPL or BRK-B.")
    ticker = ticker.replace(".", "-")
    if language not in LANGUAGES:
        raise ValueError("Choose one of the available report languages.")
    return ticker


def get_etf_allowlist(config: dict | None = None) -> tuple[str, ...]:
    """Parse the storefront's reviewed ETF list; this is not a type classifier."""
    from tradingagents.default_config import DEFAULT_CONFIG

    raw = (config or {}).get("etf_allowlist", DEFAULT_CONFIG["etf_allowlist"])
    if not isinstance(raw, str):
        raise ValueError("etf_allowlist must be comma-separated ticker symbols")
    if not raw.strip() or raw.strip().lower() == "none":
        return ()
    symbols = [part.strip().upper().replace(".", "-") for part in raw.split(",")]
    if any(not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,2})?", symbol) for symbol in symbols):
        raise ValueError("etf_allowlist contains an invalid ticker")
    return tuple(dict.fromkeys(symbols))


def validate_us_equity(ticker: str, *, config: dict | None = None) -> str:
    # Deliberately fail closed before charging: the graph's identity resolver
    # is fail-open and therefore cannot serve as a purchase eligibility check.
    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.errors import NoMarketDataError
    from tradingagents.dataflows.interface import get_vendor

    config = get_config() if config is None else config
    started = time.monotonic()
    context = {"provider": get_vendor("instrument_data", "get_instrument_info", config=config),
               "operation": "get_instrument_info", "ticker": ticker}
    log_event(logger, "market_data_request", **context)
    try:
        # Do not let an earlier research request's cached or fail-open identity
        # authorize a purchase. Validate with this storefront's configuration.
        info = get_instrument_info(ticker, config=config, refresh=True)
    except NoMarketDataError:
        log_event(logger, "market_data_response", level=logging.WARNING, **context,
                  elapsed_ms=round((time.monotonic() - started) * 1000), result="unverifiable")
        raise ProviderUnavailable("We could not verify this instrument. Please try again later.") from None
    except Exception as exc:
        # yfinance/curl 异常原文可能包含 Cookie、crumb 或代理凭据，只记录结构化错误信息。
        response = getattr(exc, "response", None)
        log_event(logger, "market_data_request_failed", level=logging.WARNING, **context,
                  elapsed_ms=round((time.monotonic() - started) * 1000), error_type=type(exc).__name__,
                  error_code=getattr(exc, "code", None), http_status=getattr(response, "status_code", None))
        raise ProviderUnavailable("Market data is temporarily unavailable.") from None
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if not isinstance(info, dict) or not info:
        log_event(logger, "market_data_response", level=logging.WARNING, **context, elapsed_ms=elapsed_ms,
                  result="unverifiable", body_format=type(info).__name__)
        raise ProviderUnavailable("We could not verify this instrument. Please try again later.")
    if not info.get("quote_type") or not info.get("exchange"):
        raise ProviderUnavailable("We could not verify this instrument. Please try again later.")
    is_etf = info.get("quote_type") == "ETF"
    eligible = ((info.get("quote_type") == "EQUITY"
                 or (is_etf and config.get("etf_reports_enabled") is True
                     and ticker in get_etf_allowlist(config)))
                and info.get("country") in {"US", "USA"}
                and info.get("exchange") in {
                    "NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS",
                    "XNAS", "XNYS", "XASE", "ARCX", "BATS", "XNGS", "XNCM", "XNMS",
                })
    context["provider"] = info["provider"]
    log_event(logger, "market_data_response", level=logging.INFO if eligible else logging.WARNING,
              **context, elapsed_ms=elapsed_ms, eligible=eligible,
              response={field: info.get(field) for field in ("symbol", "quote_type", "exchange", "country")})
    if not eligible:
        if is_etf and config.get("etf_reports_enabled") is not True:
            raise ValueError("ETF reports are currently disabled. Choose a US stock.")
        raise ValueError("Reports support US-listed stocks and ETFs on the supported ETF list.")
    return "etf" if is_etf else "stock"


def build_profile(ticker: str, language: str, base_config: dict, *, asset_type: str = "stock") -> dict:
    config = deepcopy(base_config)
    backend = config.get("backend_url")
    if backend:
        url = urlsplit(backend)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Model endpoint credentials must use environment variables, not URLs")
    request = AnalysisRequest(
        ticker=ticker,
        asset_type=asset_type,
        analysis_date=datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
        analysts=applicable_analysts(ANALYST_ORDER, asset_type),
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
            "global_news_article_limit", "global_news_lookback_days", "global_news_queries",
            "industry_sources", "industry_max_related_companies")
    runtime_config = {key: config[key] for key in keys if key in config}
    from tradingagents.default_config import DEFAULT_CONFIG

    for key in ("industry_sources", "industry_max_related_companies"):
        runtime_config.setdefault(key, deepcopy(DEFAULT_CONFIG[key]))
    for key, default in (("llm_max_retries", 2), ("max_tokens", 8192)):
        if config.get(key) is None:
            runtime_config[key] = default
    return {"version": 1, "request": request.to_dict(), "config": runtime_config}
