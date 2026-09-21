"""Yahoo-specific structured adapters and rate-limit retry policy."""

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from .errors import NoMarketDataError, VendorError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Preserve the existing Yahoo rate-limit backoff policy."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("Yahoo Finance rate limited, retrying in %.0fs (attempt %d/%d)",
                           delay, attempt + 1, max_retries)
            time.sleep(delay)


def get_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    canonical = normalize_symbol(symbol)
    end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ticker = yf.Ticker(canonical)
    data = yf_retry(lambda: ticker.history(start=start_date, end=end, auto_adjust=True))
    if data.empty:
        raise NoMarketDataError(symbol, canonical, "Yahoo Finance returned no daily bars")
    return data.reset_index()


def get_instrument_info(symbol: str) -> dict:
    canonical = normalize_symbol(symbol)
    info = yf_retry(lambda: yf.Ticker(canonical).get_info())
    if not isinstance(info, dict):
        raise VendorError("Yahoo returned invalid instrument metadata")
    if not info:
        raise NoMarketDataError(symbol, canonical, "Yahoo returned no instrument metadata")
    exchange = info.get("exchange")
    # Listing country, not the issuer's headquarters (US-listed ADRs qualify).
    country = "US" if exchange in {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BTS"} else None
    name = info.get("longName")
    if not isinstance(name, str) or not name.strip():
        name = info.get("shortName")
    return {"symbol": canonical, "provider": "yfinance", "company_name": name,
            "quote_type": info.get("quoteType"), "exchange": exchange, "country": country,
            "currency": info.get("currency"), "sector": info.get("sector"), "industry": info.get("industry")}


def get_etf_fundamentals(ticker: str, curr_date: str) -> str:
    """Yahoo fund data for local ETF analysis, with explicit snapshot limits."""
    canonical = normalize_symbol(ticker)
    if datetime.strptime(curr_date, "%Y-%m-%d").date() < datetime.now(ZoneInfo("America/New_York")).date():
        raise NoMarketDataError(ticker, canonical, "Yahoo ETF data is a current snapshot, not historical point-in-time")
    try:
        instrument = yf.Ticker(canonical)
        info = yf_retry(instrument.get_info)
        if not isinstance(info, dict) or info.get("quoteType") != "ETF":
            raise NoMarketDataError(ticker, canonical, "Yahoo did not identify an ETF")
        funds = instrument.funds_data
        # Accessors share yfinance's fund-data fetch/cache.
        description = yf_retry(lambda: funds.description)
        overview = yf_retry(lambda: funds.fund_overview)
        operations = yf_retry(lambda: funds.fund_operations)
        holdings = yf_retry(lambda: funds.top_holdings)
        sectors = yf_retry(lambda: funds.sector_weightings)
        assets = yf_retry(lambda: funds.asset_classes)
    except NoMarketDataError:
        raise
    except Exception:
        raise VendorError("Yahoo ETF data is temporarily unavailable") from None
    if not isinstance(holdings, pd.DataFrame) or holdings.empty:
        raise NoMarketDataError(ticker, canonical, "Yahoo returned no ETF holdings")
    lines = [f"Name: {info.get('longName') or info.get('shortName') or 'N/A'}",
             f"Strategy / benchmark description: {description or 'N/A'}",
             f"Currency: {info.get('currency') or 'N/A'}"]
    lines.extend(f"{key}: {value if value is not None else 'N/A'}" for key, value in (overview or {}).items())
    return (f"# ETF Fundamentals for {canonical}\n# Provider: yfinance\n"
            f"# Retrieved at: {datetime.now(timezone.utc).isoformat()}\n"
            "# Current snapshot only; profile/holdings dates unavailable, not historical point-in-time.\n"
            "# Retrieval time does not establish freshness of holdings.\n\n" + "\n".join(lines)
            + "\n\n## Fund operations (expense ratio and turnover are fractions; assets in provider units)\n"
            + (operations.to_csv() if isinstance(operations, pd.DataFrame) and not operations.empty else "N/A")
            + "\n## Reported top holdings (Holding Percent is a fraction)\n" + holdings.head(10).to_csv()
            + "\n## Sector weights (fractions)\n" + pd.Series(sectors or {}, dtype=object).to_csv()
            + "\n## Asset allocation (fractions)\n" + pd.Series(assets or {}, dtype=object).to_csv()
            + "\nTracking error and premium / discount to NAV: N/A; matching index returns and synchronized NAV are required.")
