"""Structured market data shared by tools, verification, reflection and commerce.

Adapters perform vendor I/O; this module owns routing, validation and caches.
Only tool boundaries format prose. Internal callers always receive data/errors.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd

from . import alpha_vantage_stock, fmp, marketstack, yahoo
from .config import get_config
from .errors import NoMarketDataError
from .stockstats_utils import _assert_ohlcv_not_stale, _clean_dataframe, _needs_same_day_refresh
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


class InstrumentInfo(TypedDict, total=False):
    symbol: str
    provider: str
    company_name: str | None
    quote_type: str | None
    exchange: str | None
    country: str | None
    currency: str | None
    sector: str | None
    industry: str | None


OHLCV_ADAPTERS = {"alpha_vantage": alpha_vantage_stock.get_ohlcv, "yfinance": yahoo.get_ohlcv,
                  "marketstack": marketstack.get_ohlcv, "fmp": fmp.get_ohlcv}
INSTRUMENT_ADAPTERS = {"yfinance": yahoo.get_instrument_info, "marketstack": marketstack.get_instrument_info,
                       "fmp": fmp.get_instrument_info}
SYMBOL_NORMALIZERS = {"yfinance": yahoo.normalize_symbol, "marketstack": marketstack.normalize_symbol,
                      "fmp": fmp.normalize_symbol}
_instrument_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def _vendors(method, adapters, config, vendor):
    # Lazy import avoids the tools -> provider functions -> tools cycle.
    from .interface import get_category_for_method, get_vendor

    configured = vendor or get_vendor(get_category_for_method(method), method, config=config)
    explicit = [v.strip() for v in configured.split(",") if v.strip() and v.strip() != "default"]
    selected = [provider for provider in explicit if provider in adapters] if explicit else list(adapters)
    if not selected:
        raise ValueError(f"Configured vendor(s) {explicit} not available for '{method}'")
    return selected


def get_ohlcv(symbol: str, start_date: str, end_date: str, *, config=None, vendor=None) -> pd.DataFrame:
    """Inclusive daily range; adjusted OHLC, raw volume, naive trading dates.

    The Date column is ascending and unique. attrs records the actual provider,
    provider symbol and price basis, including when an explicit fallback is used.
    """
    config = get_config() if config is None else config
    start = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    end = min(pd.Timestamp(datetime.strptime(end_date, "%Y-%m-%d")), pd.Timestamp.today().normalize())
    if start > end:
        raise NoMarketDataError(symbol, detail="Requested daily range is empty or in the future")
    end_date = end.strftime("%Y-%m-%d")
    first_error = last_no_data = None
    for provider in _vendors("get_stock_data", OHLCV_ADAPTERS, config, vendor):
        try:
            normalizer = SYMBOL_NORMALIZERS.get(provider)
            canonical = normalizer(symbol) if normalizer else symbol.strip().upper()
            safe = safe_ticker_component(canonical)
            prefix = f"ohlcv-v1-{provider}-{safe}-adjusted-rawvolume-"
            root = Path(config["data_cache_dir"])
            frame = None
            # A five-year indicator snapshot can also serve a short price or
            # reflection window without another vendor request.
            for path in sorted(root.glob(f"{prefix}*.csv"), reverse=True):
                dates = path.stem[len(prefix):].split("_")
                if len(dates) != 2 or dates[0] > start_date or dates[1] < end_date:
                    continue
                if _needs_same_day_refresh(path, end, pd.Timestamp.today()):
                    continue
                try:
                    cached = pd.read_csv(path)
                    if not cached.empty and {"Date", "Close"}.issubset(cached.columns):
                        frame = cached
                        break
                except (OSError, ValueError, pd.errors.ParserError):
                    continue
            downloaded = frame is None
            if downloaded:
                frame = OHLCV_ADAPTERS[provider](canonical, start_date, end_date)
            if frame.empty:
                raise NoMarketDataError(symbol, canonical, "No daily bars returned")
            frame = _clean_dataframe(frame.copy())
            frame = frame.loc[(frame["Date"] >= start) & (frame["Date"] <= end)]
            frame = frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
            if frame.empty or "Close" not in frame:
                raise NoMarketDataError(symbol, canonical, "No daily bars in requested range")
            if pd.isna(frame["Close"].iloc[-1]):
                raise NoMarketDataError(symbol, canonical, "latest in-range OHLCV bar has no closing price")
            _assert_ohlcv_not_stale(frame, end_date, symbol, canonical)
            if downloaded:
                root.mkdir(parents=True, exist_ok=True)
                path = root / f"{prefix}{start_date}_{end_date}.csv"
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", dir=root, suffix=".tmp", delete=False) as handle:
                        temporary = handle.name
                        frame.to_csv(handle, index=False)
                    os.replace(temporary, path)
                finally:
                    if temporary and os.path.exists(temporary):
                        os.unlink(temporary)
            frame.attrs.update(provider=provider, symbol=canonical, price_basis="adjusted_ohlc_raw_volume")
            logger.info("market_data_result provider=%s symbol=%s cache_hit=%s rows=%d latest=%s",
                        provider, canonical, not downloaded, len(frame), frame["Date"].iloc[-1].date())
            return frame
        except NoMarketDataError as exc:
            last_no_data = exc
        except Exception as exc:
            first_error = first_error or exc
            logger.warning("OHLCV provider=%s failed error_type=%s", provider, type(exc).__name__)
    if last_no_data is not None:
        raise last_no_data
    if first_error is not None:
        raise first_error
    raise ValueError("No OHLCV vendor configured")


def get_instrument_info(symbol: str, *, config=None, vendor=None, refresh=False) -> InstrumentInfo:
    config = get_config() if config is None else config
    first_error = None
    for provider in _vendors("get_instrument_info", INSTRUMENT_ADAPTERS, config, vendor):
        try:
            normalizer = SYMBOL_NORMALIZERS.get(provider)
            canonical = normalizer(symbol) if normalizer else symbol.strip().upper()
            key = provider, canonical
            cached = _instrument_cache.get(key)
            if not refresh and cached and time.monotonic() - cached[0] < 86400:
                return deepcopy(cached[1])
            info = INSTRUMENT_ADAPTERS[provider](canonical)
            if len(_instrument_cache) >= 256:
                _instrument_cache.pop(next(iter(_instrument_cache)))
            _instrument_cache[key] = time.monotonic(), deepcopy(info)
            return info
        except Exception as exc:
            first_error = first_error or exc
            logger.warning("Instrument provider=%s failed error_type=%s", provider, type(exc).__name__)
    if first_error is not None:
        raise first_error
    raise ValueError("No instrument vendor configured")


def get_stock_data(symbol: str, start_date: str, end_date: str, *, vendor=None) -> str:
    frame = get_ohlcv(symbol, start_date, end_date, vendor=vendor)
    header = (f"# Stock data for {frame.attrs['symbol']} from {start_date} to {end_date}\n"
              f"# Provider: {frame.attrs['provider']}\n"
              f"# Price basis: {frame.attrs['price_basis']}\n"
              f"# Latest trading date: {frame['Date'].iloc[-1].date()} (daily, not real-time)\n"
              f"# Total records: {len(frame)}\n\n")
    return header + frame.round({k: 2 for k in ("Open", "High", "Low", "Close") if k in frame}).to_csv(index=False)
