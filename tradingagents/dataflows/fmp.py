"""FMP stable API adapters using fmpsdk; daily prices and report data.

The SDK is initialized only on a selected FMP request. Its retries are disabled
so quota exhaustion and transient throttling have distinct, bounded behavior.
Never expose SDK exception messages: they may contain credentials or payloads.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from functools import partial
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import get_config
from .date_window import in_window
from .errors import NoMarketDataError, VendorError, VendorNotConfiguredError, VendorRateLimitError

logger = logging.getLogger(__name__)
US_EXCHANGES = {"NASDAQ": "XNAS", "NYSE": "XNYS", "AMEX": "XASE",
                "NYSE AMERICAN": "XASE", "NYSE ARCA": "ARCX", "ARCA": "ARCX",
                "BATS": "BATS", "XNAS": "XNAS", "XNYS": "XNYS", "XASE": "XASE"}


def normalize_symbol(symbol: str) -> str:
    canonical = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z]{1,5}(?:[.-][A-Z]{1,2})?", canonical):
        raise NoMarketDataError(symbol, detail="FMP adapter supports US stocks and ETFs only")
    return canonical.replace(".", "-")


def _failure(status: int | None, detail: str) -> tuple[VendorError, bool]:
    """Classify upstream errors without putting upstream text into an exception."""
    detail = detail.lower()
    if (any(word in detail for word in ("quota", "bandwidth", "daily limit", "monthly limit",
                                        "daily request", "monthly request", "usage limit"))
            or ("limit reach" in detail and "upgrade" in detail)):
        return VendorRateLimitError("FMP request quota exhausted"), False
    if status in {401, 402, 403} or any(word in detail for word in (
        "invalid api", "invalid key", "unauthorized", "not authorized", "subscription",
        "upgrade", "premium", "restricted", "permission", "missing api",
    )):
        return VendorNotConfiguredError("FMP authentication or plan permission denied"), False
    if status == 429 or any(word in detail for word in ("rate limit", "too many requests", "limit reach")):
        return VendorRateLimitError("FMP request rate limited"), True
    if status is None or status >= 500:
        return VendorError("FMP temporarily unavailable or request timed out"), True
    return VendorError("FMP rejected the request or returned an invalid response"), False


def _request(group: str, method: str, **params) -> list[dict]:
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise VendorNotConfiguredError("FMP_API_KEY is not configured")
    try:
        import fmpsdk
    except ImportError:
        raise VendorNotConfiguredError("Install the project's pinned fmpsdk dependency") from None

    with requests.Session() as session:
        client = fmpsdk.Client(api_key=key, max_retries=0, connect_timeout=5, read_timeout=30,
                               session=session)
        operation = getattr(getattr(client, group), method)
        for attempt in range(3):
            started, status, row_count = time.monotonic(), None, 0
            try:
                body = operation(**params)
                status = 200
                if isinstance(body, dict):
                    # Stable endpoints used here all return arrays. An object
                    # can be an error even when FMP responds with HTTP 200.
                    failure, retry = _failure(status, str(body))
                elif not isinstance(body, list) or any(not isinstance(row, dict) for row in body):
                    raise VendorError("FMP returned an invalid record list")
                else:
                    for row in body:
                        if any(k.lower() in {"error", "error message", "error_message"} for k in row):
                            failure, retry = _failure(status, str(row))
                            break
                    else:
                        row_count = len(body)
                        return body
            except fmpsdk.FMPError as exc:
                status = exc.status_code
                if status == 404:
                    raise NoMarketDataError(params.get("symbol", "FMP"),
                                            detail="FMP resource not found") from None
                failure, retry = _failure(status, exc.response_text)
            except (requests.Timeout, requests.ConnectionError):
                failure, retry = _failure(None, "")
            except (requests.RequestException, ValueError):
                raise VendorError("FMP transport or response parsing failure") from None
            finally:
                logger.info("market_data_request provider=fmp endpoint=%s.%s attempt=%d status=%s rows=%d elapsed_ms=%d",
                            group, method, attempt + 1, status, row_count,
                            round((time.monotonic() - started) * 1000))
            if not retry or attempt == 2:
                raise failure from None
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable retry state")


def _symbol_rows(rows: list[dict], symbol: str) -> list[dict]:
    for row in rows:
        actual = row.get("symbol")
        if not isinstance(actual, str) or actual.upper().replace(".", "-") != symbol:
            raise VendorError("FMP returned a different or unidentified instrument")
    return rows


def get_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    canonical = normalize_symbol(symbol)
    start = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
    end = min(pd.Timestamp(datetime.strptime(end_date, "%Y-%m-%d")), pd.Timestamp.today().normalize())
    parts = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + pd.DateOffset(years=5))
        args = {"symbol": canonical, "from_": str(cursor.date()), "to": str(stop.date())}
        frames = []
        for method, columns in (
            ("historical_price_eod_dividend_adjusted",
             {"date": "Date", "adjOpen": "Open", "adjHigh": "High", "adjLow": "Low", "adjClose": "Close"}),
            ("historical_price_eod_non_split_adjusted", {"date": "Date", "volume": "Volume"}),
        ):
            rows = _symbol_rows(_request("chart", method, **args), canonical)
            if not rows:
                frames.append(pd.DataFrame(columns=list(columns.values())))
                continue
            frame = pd.DataFrame(rows)
            if "date" not in frame:
                raise NoMarketDataError(symbol, canonical, "FMP daily bars have no dates")
            dates = pd.to_datetime(frame["date"].astype(str).str[:10], errors="coerce")
            if dates.isna().any():
                raise VendorError("FMP daily bars contain invalid dates")
            frame = frame.assign(date=dates).loc[(dates >= cursor) & (dates <= stop)]
            if frame.empty:
                frames.append(pd.DataFrame(columns=list(columns.values())))
                continue
            if not set(columns).issubset(frame):
                raise NoMarketDataError(symbol, canonical, "FMP adjusted OHLC or raw volume is missing")
            frames.append(frame[list(columns)].rename(columns=columns).drop_duplicates("Date", keep="last"))
        if not all(frame.empty for frame in frames):
            # An outer join rejects missing dates on either side, including a
            # missing newest adjusted bar. Never silently truncate to a match.
            for frame in frames:
                frame["Date"] = pd.to_datetime(frame["Date"])
            parts.append(frames[0].merge(frames[1], on="Date", how="outer", validate="one_to_one"))
        cursor = stop + pd.Timedelta(days=1)
    if not parts:
        raise NoMarketDataError(symbol, canonical, "FMP returned no daily bars")
    out = pd.concat(parts, ignore_index=True).sort_values("Date").drop_duplicates("Date", keep="last")
    values = out[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    if (not values.map(math.isfinite).all().all()
            or (values[["Open", "High", "Low", "Close"]] <= 0).any().any()
            or (values["Volume"] < 0).any()):
        raise NoMarketDataError(symbol, canonical, "FMP adjusted OHLCV is missing or incomplete")
    out[values.columns] = values
    return out.reset_index(drop=True)


def get_instrument_info(symbol: str) -> dict:
    canonical = normalize_symbol(symbol)
    rows = _symbol_rows(_request("company", "profile", symbol=canonical), canonical)
    if len(rows) != 1:
        raise NoMarketDataError(symbol, canonical, "FMP returned no unique instrument metadata")
    info = rows[0]
    exchange = US_EXCHANGES.get(str(info.get("exchange") or "").upper())
    kind = None
    if info.get("isEtf") is True:
        kind = "ETF"
    elif info.get("isFund") is True:
        kind = "MUTUALFUND"
    elif (info.get("isEtf") is False and info.get("isFund") is False
          and info.get("isActivelyTrading") is True and exchange):
        kind = "EQUITY"
    return {"symbol": canonical, "provider": "fmp", "company_name": info.get("companyName"),
            "quote_type": kind, "exchange": exchange, "country": "US" if exchange else None,
            "currency": info.get("currency"), "sector": info.get("sector"), "industry": info.get("industry")}


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    canonical = normalize_symbol(ticker)
    today = datetime.now(ZoneInfo("America/New_York")).date()
    if curr_date and datetime.strptime(curr_date, "%Y-%m-%d").date() < today:
        # Profile/quote/TTM endpoints are current snapshots, not point-in-time.
        raise NoMarketDataError(ticker, canonical, "FMP overview is a current snapshot; use dated financial statements for historical analysis")
    data = {}
    for label, group, method in (("profile", "company", "profile"), ("quote", "quote", "quote"),
                                  ("ratios", "statements", "ratios_ttm"),
                                  ("metrics", "statements", "key_metrics_ttm")):
        rows = _symbol_rows(_request(group, method, symbol=canonical), canonical)
        if len(rows) > 1:
            raise VendorError("FMP returned ambiguous fundamentals")
        data[label] = rows[0] if rows else {}
    if not data["profile"].get("companyName"):
        raise NoMarketDataError(ticker, canonical, "FMP returned no company fundamentals")
    fields = (
        ("Name", "profile", "companyName"), ("Sector", "profile", "sector"),
        ("Industry", "profile", "industry"), ("Currency", "profile", "currency"),
        ("Market Cap", "quote", "marketCap"), ("Beta", "profile", "beta"),
        ("52 Week High", "quote", "yearHigh"), ("52 Week Low", "quote", "yearLow"),
        ("50 Day Average", "quote", "priceAvg50"), ("200 Day Average", "quote", "priceAvg200"),
        ("PE Ratio (TTM)", "ratios", "priceToEarningsRatioTTM"),
        ("PEG Ratio (TTM)", "ratios", "priceToEarningsGrowthRatioTTM"),
        ("Price to Book", "ratios", "priceToBookRatioTTM"),
        ("Dividend Yield (fraction)", "ratios", "dividendYieldTTM"),
        ("Profit Margin (fraction)", "ratios", "netProfitMarginTTM"),
        ("Operating Margin (fraction)", "ratios", "operatingProfitMarginTTM"),
        ("Return on Equity (fraction)", "metrics", "returnOnEquityTTM"),
        ("Return on Assets (fraction)", "metrics", "returnOnAssetsTTM"),
        ("Debt to Equity (ratio)", "ratios", "debtToEquityRatioTTM"),
        ("Current Ratio", "ratios", "currentRatioTTM"),
        ("Book Value per Share", "ratios", "bookValuePerShareTTM"),
        ("Free Cash Flow per Share (TTM)", "ratios", "freeCashFlowPerShareTTM"),
    )
    lines = [f"{label}: {data[source].get(key) if data[source].get(key) is not None else 'N/A'}"
             for label, source, key in fields]
    lines.extend(f"{label}: N/A" for label in ("Forward PE", "Forward EPS", "EPS (TTM)",
                 "Revenue (TTM)", "Gross Profit", "EBITDA", "Net Income", "Free Cash Flow"))
    return (f"# Company Fundamentals for {canonical}\n# Provider: fmp\n"
            f"# Retrieved at: {datetime.now(timezone.utc).isoformat()}\n"
            f"# Current snapshot, not historical point-in-time; quote timestamp: {data['quote'].get('timestamp', 'N/A')}\n\n"
            + "\n".join(lines))


def _statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None,
               *, method: str, title: str) -> str:
    canonical = normalize_symbol(ticker)
    if freq.lower() not in {"annual", "quarterly"}:
        raise ValueError("Financial statement frequency must be annual or quarterly")
    cutoff = min(datetime.strptime(curr_date, "%Y-%m-%d").date() if curr_date else datetime.now().date(),
                 datetime.now().date())
    rows = _symbol_rows(_request("statements", method, symbol=canonical,
                                period="quarter" if freq.lower() == "quarterly" else "annual",
                                limit=1000), canonical)
    kept = []
    for row in rows:
        period = pd.to_datetime(row.get("date"), errors="coerce", utc=True)
        disclosed = [pd.to_datetime(row.get(key), errors="coerce", utc=True)
                     for key in ("acceptedDate", "filingDate")]
        disclosed = [day for day in disclosed if day is not None and not pd.isna(day)]
        # A fiscal period end alone cannot prove the report was public.
        if period is None or pd.isna(period) or not disclosed:
            continue
        if period.date() <= cutoff and max(disclosed).date() <= cutoff:
            kept.append(row)
    if not kept:
        raise NoMarketDataError(ticker, canonical, "FMP has no financial statements with verified disclosure dates in range")
    frame = pd.DataFrame(kept).sort_values(["date"], ascending=False).drop_duplicates("date", keep="first")
    return (f"# {title} data for {canonical} ({freq})\n# Provider: fmp\n"
            f"# Disclosure cutoff: {cutoff}; latest fiscal period: {frame['date'].iloc[0]}\n"
            "# Filtered by disclosure date; provider restatements are not point-in-time vintages.\n\n"
            + frame.to_csv(index=False))


get_balance_sheet = partial(_statement, method="balance_sheet_statement", title="Balance Sheet")
get_cashflow = partial(_statement, method="cash_flow_statement", title="Cash Flow")
get_income_statement = partial(_statement, method="income_statement", title="Income Statement")


def _news(start_date: str, end_date: str, limit: int, symbol: str | None = None) -> str:
    start, end = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
    end = min(end, datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0))
    if limit < 1 or start > end:
        return "# Provider: fmp\nNo news in the requested range."
    canonical = normalize_symbol(symbol) if symbol else None
    method = "news_stock" if canonical else "news_general_latest"
    articles, seen, pages = [], set(), set()
    page_size = min(100, max(20, limit))
    for page in range(100):
        args = {"from_": str(start.date()), "to": str(end.date()), "page": page, "limit": page_size}
        if canonical:
            args["symbols"] = canonical
        rows = _request("news", method, **args)
        if not rows:
            break
        signature = tuple((str(row.get("url")), str(row.get("title")), str(row.get("publishedDate"))) for row in rows)
        if signature in pages:
            raise VendorError("FMP news pagination repeated a page")
        pages.add(signature)
        for row in rows:
            if canonical and str(row.get("symbol") or "").upper().replace(".", "-") != canonical:
                continue
            stamp = pd.to_datetime(row.get("publishedDate"), errors="coerce", utc=True)
            if stamp is None or pd.isna(stamp) or not in_window(stamp.to_pydatetime(), start, end):
                continue
            identity = row.get("url") or row.get("title")
            if not identity or identity in seen or not row.get("title"):
                continue
            seen.add(identity)
            articles.append((stamp, row))
        if len(articles) >= limit or len(rows) < page_size:
            break
    else:
        raise VendorError("FMP news pagination exceeded the safety limit")
    header = f"## {canonical or 'Global'} News, from {start.date()} to {end.date()}\n# Provider: fmp\n\n"
    if not articles:
        return header + "No news found in the requested range."
    return header + "\n\n".join(
        f"### {row['title']} (source: {row.get('publisher') or row.get('site') or 'Unknown'})\n"
        f"Published: {stamp.isoformat()}\n{row.get('text') or ''}\nLink: {row.get('url') or ''}"
        for stamp, row in sorted(articles, key=lambda item: item[0], reverse=True)[:limit])


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    return _news(start_date, end_date, get_config()["news_article_limit"], symbol=ticker)


def get_global_news(curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
    config = get_config()
    days = config["global_news_lookback_days"] if look_back_days is None else look_back_days
    limit = config["global_news_article_limit"] if limit is None else limit
    start = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=days)
    return _news(str(start.date()), curr_date, limit)


def get_insider_transactions(ticker: str) -> str:
    canonical = normalize_symbol(ticker)
    rows = _symbol_rows(_request("insider_trades", "insider_trading_search", symbol=canonical,
                                page=0, limit=100), canonical)
    header = (f"# Insider Transactions data for {canonical}\n# Provider: fmp\n"
              f"# Retrieved at: {datetime.now(timezone.utc).isoformat()}\n"
              "# Latest up to 100 records; not a historical as-of query.\n\n")
    if not rows:
        return header + f"No insider transactions reported for symbol '{canonical}'"
    frame = pd.DataFrame(rows)
    for column in ("transactionDate", "filingDate"):
        if column not in frame:
            frame[column] = "N/A"
    return header + frame.to_csv(index=False)
