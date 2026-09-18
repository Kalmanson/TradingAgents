"""Yahoo-specific structured adapters and rate-limit retry policy."""

import logging
import time

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
