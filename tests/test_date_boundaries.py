"""yfinance treats ``end`` as exclusive; we must request one extra day so the
requested end_date (and the current day) is actually included.

Regressions for #986 (current-day OHLCV excluded) and #987 (requested end_date
row omitted).
"""
import pandas as pd
import pytest

import tradingagents.dataflows.stockstats_utils as su
import tradingagents.dataflows.y_finance as yfin
from tradingagents.dataflows import market_data
from tradingagents.dataflows.config import set_config


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "BRK.B", "BRK-B", "SPY"])
def test_fmp_adjusted_prices_raw_volume_dates_and_cache(fmp_http, tmp_path, symbol):
    canonical = symbol.replace(".", "-")

    def respond(endpoint, params):
        assert params == {"symbol": canonical, "from": "2024-06-07", "to": "2024-06-09"}
        # A split/dividend changes price; raw trading volume must not be taken
        # from the adjusted endpoint. Also exclude a malformed future price.
        rows = [{"symbol": canonical, "date": "2024-06-07", "adjOpen": 100,
                 "adjHigh": 105, "adjLow": 99, "adjClose": 102,
                 "volume": 10 if endpoint.endswith("/dividend-adjusted") else 100}]
        return 200, [*rows, *rows, {"symbol": canonical, "date": "2024-06-10"}]

    fmp_http.respond = respond
    config = {"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "fmp"}}
    data = market_data.get_ohlcv(symbol, "2024-06-07", "2024-06-09", config=config)
    assert data["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-06-07"]
    assert data["Close"].tolist() == [102] and data["Volume"].tolist() == [100]
    assert data.attrs["provider"] == "fmp" and data.attrs["symbol"] == canonical
    again = market_data.get_ohlcv(symbol, "2024-06-07", "2024-06-09", config=config)
    assert data.equals(again) and len(fmp_http.calls) == 2


@pytest.mark.parametrize("bad", ["missing_adjusted", "missing_volume", "different_dates", "different_symbol", "empty"])
def test_fmp_incomplete_bars_cannot_become_usable_prices(fmp_http, bad):
    from tradingagents.dataflows import fmp
    from tradingagents.dataflows.errors import VendorError

    def respond(endpoint, params):
        if bad == "empty":
            return 200, []
        adjusted = endpoint.endswith("/dividend-adjusted")
        row = {"symbol": "AAPL", "date": "2026-06-10", "adjOpen": 10,
               "adjHigh": 12, "adjLow": 9, "adjClose": 11, "volume": 100}
        if bad == "missing_adjusted" and adjusted:
            row.pop("adjClose")
        if bad == "missing_volume" and not adjusted:
            row["volume"] = None
        if bad == "different_dates" and not adjusted:
            row["date"] = "2026-06-09"
        if bad == "different_symbol":
            row["symbol"] = "MSFT"
        return 200, [row]

    fmp_http.respond = respond
    with pytest.raises(VendorError):
        fmp.get_ohlcv("AAPL", "2026-06-01", "2026-06-10")


@pytest.mark.parametrize("method,endpoint", [("get_balance_sheet", "balance-sheet-statement"),
                                           ("get_income_statement", "income-statement"),
                                           ("get_cashflow", "cash-flow-statement")])
@pytest.mark.parametrize("freq,period", [("quarterly", "quarter"), ("annual", "annual")])
def test_fmp_statements_filter_disclosure_not_only_period(fmp_http, method, endpoint, freq, period):
    from tradingagents.dataflows import fmp

    def respond(path, params):
        assert path == endpoint and params["period"] == period and params["limit"] == 1000
        return 200, [
            {"symbol": "AAPL", "date": "2025-12-31", "filingDate": "2026-02-01", "reportedCurrency": "USD", "revenue": 123},
            {"symbol": "AAPL", "date": "2026-03-31", "filingDate": "2026-05-10", "revenue": 999},
            {"symbol": "AAPL", "date": "2025-09-30", "revenue": 888},
            {"symbol": "AAPL", "date": "2025-06-30", "filingDate": "2025-07-30", "acceptedDate": "2026-05-11 01:00:00", "revenue": 777},
        ]

    fmp_http.respond = respond
    result = getattr(fmp, method)("AAPL", freq, "2026-05-09")
    assert "123" in result and "USD" in result and "filingDate" in result and "fmp" in result
    assert all(value not in result for value in ("999", "888", "777"))


def test_fmp_news_pages_filter_and_deduplicate_before_limit(fmp_http):
    from tradingagents.dataflows import fmp

    def respond(endpoint, params):
        assert endpoint in {"news/stock", "news/general-latest"}
        if params["page"] == 0:
            # Filling a page with duplicates and out-of-range records must
            # not consume the caller's article limit before filtering.
            return 200, [{"symbol": "AAPL", "publishedDate": "2026-06-11 00:00:00", "title": "future", "url": "future"}] * 10 + [
                {"symbol": "AAPL", "publishedDate": "2026-06-10 23:59:59", "title": "last second", "url": "one"}] * 10
        return 200, [{"symbol": "AAPL", "publishedDate": "2026-06-01 00:00:00", "title": "first second", "url": "two"},
                     {"symbol": "AAPL", "title": "undated", "url": "three"}]

    fmp_http.respond = respond
    set_config({"news_article_limit": 2})
    for result in (fmp.get_news("AAPL", "2026-06-01", "2026-06-10"), fmp.get_global_news("2026-06-10", 9, 2)):
        assert result.count("### ") == 2 and "first second" in result and "last second" in result
        assert "future" not in result and "undated" not in result
        assert "Provider: fmp" in result and "Published:" in result
    assert [params["page"] for _, params in fmp_http.calls] == [0, 1, 0, 1]


def test_fmp_current_overview_cannot_leak_into_historical_analysis(fmp_http):
    from tradingagents.dataflows import fmp
    from tradingagents.dataflows.errors import NoMarketDataError

    with pytest.raises(NoMarketDataError, match="current snapshot"):
        fmp.get_fundamentals("AAPL", "2020-01-01")
    assert not fmp_http.calls


def test_fmp_insider_empty_and_latest_record_dates(fmp_http):
    from tradingagents.dataflows import fmp

    assert "No insider transactions" in fmp.get_insider_transactions("AAPL")
    fmp_http.respond = lambda endpoint, params: (200, [{"symbol": "AAPL", "transactionDate": "2026-06-01",
                                                       "filingDate": "2026-06-03", "price": 123}])
    out = fmp.get_insider_transactions("AAPL")
    assert "transactionDate" in out and "filingDate" in out and "not a historical as-of query" in out
    assert "2026-06-01" in out and "2026-06-03" in out


@pytest.mark.unit
def test_get_yfin_requests_inclusive_end(monkeypatch):
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, start, end, **kwargs):
            captured["start"] = start
            captured["end"] = end
            idx = pd.to_datetime(["2025-05-08", "2025-05-09"])
            return pd.DataFrame(
                {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                 "Close": [1.0, 2.0], "Volume": [1, 2]},
                index=idx,
            )

    monkeypatch.setattr(yfin.yf, "Ticker", FakeTicker)
    out = yfin.get_YFin_data_online("AAPL", "2025-05-01", "2025-05-09")

    # end is requested one day past end_date so 2025-05-09 is included (#987).
    assert captured["end"] == "2025-05-10"
    # Header still reflects the requested range, not the internal +1 day.
    assert "to 2025-05-09" in out


@pytest.mark.unit
def test_load_ohlcv_requests_inclusive_end(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    captured = {}

    def fake_download(symbol, start, end, **kwargs):
        captured["end"] = end
        idx = pd.to_datetime([pd.Timestamp.today().normalize()])
        return pd.DataFrame(
            {"Open": [100.0], "High": [100.0], "Low": [100.0],
             "Close": [100.0], "Volume": [1]},
            index=idx,
        )

    monkeypatch.setattr(yfin.yf, "Ticker", lambda symbol: type("Ticker", (), {"history": lambda self, **kw: fake_download(symbol, **kw)})())
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    su.load_ohlcv("AAPL", today)

    expected_end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert captured["end"] == expected_end  # tomorrow -> today's row included (#986)


def test_marketstack_date_sorting_deduplication_and_future_exclusion(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tradingagents.dataflows import marketstack

    monkeypatch.setenv("MARKETSTACK_API_KEY", "key")
    set_config({"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "marketstack"}})
    rows = [{"date": date, "symbol": "AAPL", "exchange": "XNAS", "adj_open": 10, "adj_high": 12,
             "adj_low": 9, "adj_close": value, "volume": 100}
            for date, value in [("2026-06-10T00:00:00+0000", 11), ("2026-06-09T00:00:00+0000", 10),
                                ("2026-06-09T00:00:00+0000", 10), ("2026-06-11T00:00:00+0000", None)]]
    monkeypatch.setattr(marketstack.requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, headers={}, json=lambda: {"data": rows, "pagination": {"offset": 0, "total": 4}},
    ))
    data = market_data.get_ohlcv("AAPL", "2026-06-09", "2026-06-10")
    assert list(data["Date"].dt.strftime("%Y-%m-%d")) == ["2026-06-09", "2026-06-10"]
    assert list(data["Close"]) == [10, 11]


def test_historical_indicators_request_five_years_ending_at_cutoff(monkeypatch, tmp_path):
    from unittest.mock import Mock

    set_config({"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "marketstack"}})
    fetch = Mock(return_value=pd.DataFrame({"Date": ["2020-05-08"], "Close": [10.]}))
    monkeypatch.setitem(market_data.OHLCV_ADAPTERS, "marketstack", fetch)
    su.load_ohlcv("AAPL", "2020-05-09")
    fetch.assert_called_once_with("AAPL", "2015-05-09", "2020-05-09")
