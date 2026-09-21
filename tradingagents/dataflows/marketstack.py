"""Marketstack V2 adapter: US daily bars and deterministic instrument metadata."""

from __future__ import annotations

import logging
import math
import os
import re
import time
from contextlib import suppress

import pandas as pd
import requests

from .errors import NoMarketDataError, VendorError, VendorNotConfiguredError, VendorRateLimitError

logger = logging.getLogger(__name__)
API_URL = "https://api.marketstack.com/v2"
US_EXCHANGES = frozenset({"XNAS", "XNYS", "XASE", "ARCX", "BATS", "XNGS", "XNCM", "XNMS"})


def normalize_symbol(symbol: str) -> str:
    """Only translate US share-class punctuation, never broker asset aliases."""
    canonical = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?", canonical):
        raise NoMarketDataError(symbol, detail="Marketstack adapter supports US stocks and ETFs only")
    return canonical.replace("-", ".")


def _request(endpoint: str, params: dict) -> dict:
    key = os.environ.get("MARKETSTACK_API_KEY", "").strip()
    if not key:
        raise VendorNotConfiguredError("MARKETSTACK_API_KEY is not configured")
    # Neither raw exceptions nor response messages are safe: both can echo the
    # access_key query parameter. Only log bounded, locally generated fields.
    for attempt in range(3):
        started = time.monotonic()
        status = None
        retry_after = 0
        try:
            response = requests.get(
                f"{API_URL}/{endpoint}", params={**params, "access_key": key}, timeout=(5, 30)
            )
            status = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            error = body.get("error") or {}
            if not isinstance(error, dict):
                error = {}
            kind = error.get("type") or error.get("code")
            if not isinstance(kind, (str, int, type(None))):
                raise VendorError("Marketstack returned an invalid error response")
            # Usage exhaustion (monthly or daily) is not a short-lived throttle.
            if kind in {104, "104", "usage_limit_reached", "monthly_limit_reached",
                        "daily_usage_limit_reached"}:
                raise VendorRateLimitError("Marketstack request quota exhausted")
            if status in {401, 403} or kind in {
                101, "101", 102, "102", 105, "105", "invalid_access_key",
                "missing_access_key", "inactive_user", "function_access_restricted",
                "https_access_restricted",
            }:
                raise VendorNotConfiguredError("Marketstack authentication or plan permission denied")
            if status == 429 or kind in {429, "429", "rate_limit_reached", "too_many_requests"}:
                failure = VendorRateLimitError("Marketstack request rate limited")
                with suppress(ValueError, TypeError):
                    retry_after = min(30, max(0, float(response.headers.get("Retry-After", 0))))
            elif status >= 500:
                failure = VendorError("Marketstack temporarily unavailable")
            elif status == 404:
                raise NoMarketDataError(params.get("symbols") or endpoint.rsplit("/", 1)[-1],
                                        detail="Marketstack instrument not found")
            elif status >= 400 or error or body.get("success") is False:
                raise VendorError("Marketstack rejected the request")
            elif not body:
                raise VendorError("Marketstack returned an invalid response")
            else:
                return body
        except (requests.Timeout, requests.ConnectionError):
            failure = VendorError("Marketstack request timed out or connection failed")
        except requests.RequestException:
            raise VendorError("Marketstack transport failure") from None
        finally:
            logger.info("market_data_request provider=marketstack endpoint=%s attempt=%d status=%s elapsed_ms=%d",
                        endpoint, attempt + 1, status, round((time.monotonic() - started) * 1000))
        if attempt == 2:
            raise failure from None
        time.sleep(max(2 ** attempt, retry_after))
    raise AssertionError("Unreachable retry state")


def get_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    canonical = normalize_symbol(symbol)
    records = []
    offset = 0
    while True:
        body = _request("eod", {"symbols": canonical, "date_from": start_date,
                               "date_to": end_date, "sort": "ASC", "limit": 1000, "offset": offset})
        page = body.get("data")
        pagination = body.get("pagination")
        if not isinstance(page, list) or not isinstance(pagination, dict):
            raise VendorError("Marketstack EOD response has no valid data/pagination")
        total = pagination.get("total")
        if not isinstance(total, int) or total < 0 or pagination.get("offset") != offset:
            raise VendorError("Marketstack EOD pagination is inconsistent")
        if not page:
            if offset < total:
                raise VendorError("Marketstack EOD response ended before all pages arrived")
            break
        for row in page:
            if not isinstance(row, dict) or str(row.get("symbol") or "").upper().replace("-", ".") != canonical:
                raise VendorError("Marketstack returned a different or invalid instrument")
            if row.get("exchange") not in US_EXCHANGES:
                raise NoMarketDataError(symbol, canonical, "Marketstack daily bar is not a supported US listing")
            records.append(row)
        offset += len(page)
        if offset >= total:
            break
    if not records:
        raise NoMarketDataError(symbol, canonical, "Marketstack returned no daily bars")
    frame = pd.DataFrame(records)
    columns = {"date": "Date", "adj_open": "Open", "adj_high": "High",
               "adj_low": "Low", "adj_close": "Close", "volume": "Volume"}
    if not set(columns).issubset(frame.columns):
        raise NoMarketDataError(symbol, canonical, "Marketstack adjusted OHLCV fields are missing")
    # Filter the requested trading dates before validating prices: an extra
    # future bar must neither leak into a backtest nor invalidate its history.
    dates = pd.to_datetime(frame["date"].astype(str).str[:10], errors="coerce")
    frame = frame.loc[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]
    out = frame[list(columns)].rename(columns=columns)
    prices = out[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    if (out.empty or not prices.map(lambda value: math.isfinite(value)).all().all()
            or (prices[["Open", "High", "Low", "Close"]] <= 0).any().any()
            or (prices["Volume"] < 0).any()):
        raise NoMarketDataError(symbol, canonical, "Marketstack adjusted OHLCV is missing or incomplete")
    out[prices.columns] = prices
    return out


def get_instrument_info(symbol: str) -> dict:
    canonical = normalize_symbol(symbol)
    body = _request(f"tickers/{canonical}", {})
    info = body.get("data", body)
    if not isinstance(info, dict) or not info:
        raise NoMarketDataError(symbol, canonical, "Marketstack returned no instrument metadata")
    if info.get("symbol", "").upper().replace("-", ".") != canonical:
        raise VendorError("Marketstack metadata does not match the requested symbol")
    exchange = info.get("stock_exchange") or {}
    if not isinstance(exchange, dict):
        raise VendorError("Marketstack exchange metadata is invalid")
    return {
        "symbol": canonical, "provider": "marketstack", "company_name": info.get("name"),
        "quote_type": str(info.get("item_type") or "").upper(),
        "exchange": exchange.get("mic"), "country": exchange.get("country_code"),
        # V2 TickerResponse 没有货币字段；price_currency 只出现在 EOD 数据行中。
        # currency 无下游消费，保持 None 而不是读取 API 不会返回的字段。
        "currency": None, "sector": info.get("sector"),
        "industry": info.get("industry"),
    }
