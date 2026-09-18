"""Vendor router must respect the configured chain and never silently hide a
broken primary.

Regressions for #988 (explicit single-vendor config still fell back to others),
#289 (fallback ran for unchosen vendors), and #989 (serious primary failures
were swallowed without a trace).
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _reset_config():
    # Hard reset: set_config() merges, so empty DEFAULT dicts (e.g. tool_vendors)
    # don't clear keys leaked by other tests. Replace the global outright.
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _no_data(symbol, *a, **k):
    raise NoMarketDataError(symbol, symbol, "no rows")


def _returns(value):
    def impl(symbol, *a, **k):
        return value
    return impl


def _raises(exc):
    def impl(symbol, *a, **k):
        raise exc
    return impl


@pytest.mark.unit
class VendorRoutingTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route(self, vendors_for_get_stock_data):
        return mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": vendors_for_get_stock_data},
            clear=False,
        )

    def test_explicit_single_vendor_does_not_fall_back(self):
        # #988: with yfinance pinned, a healthy alpha_vantage must NOT be used.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        av = mock.Mock(side_effect=_returns("AV_DATA"))
        with self._route({"yfinance": _no_data, "alpha_vantage": av}):
            result = interface.route_to_vendor("get_stock_data", "FAKE", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        av.assert_not_called()  # the unchosen vendor was never tried

    def test_explicit_multi_vendor_falls_back_within_chain(self):
        # Listing both vendors opts in to ordered fallback.
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def test_primary_error_is_logged_not_masked(self):
        # #989: primary errors + fallback no-data -> NO_DATA, but the failure
        # must be visible in logs (broken primary not hidden).
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _raises(ValueError("boom")), "alpha_vantage": _no_data}), \
                self.assertLogs("tradingagents.dataflows.interface", level="WARNING") as cm:
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        joined = "\n".join(cm.output)
        self.assertIn("boom", joined)            # the real error surfaced in logs
        self.assertIn("yfinance", joined)

    def test_unknown_configured_vendor_raises(self):
        set_config({"data_vendors": {"core_stock_apis": "bogus_vendor"}})
        with self.assertRaises(ValueError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("bogus_vendor", str(ctx.exception))

    def test_default_sentinel_uses_all_vendors(self):
        # No explicit choice ("default") keeps the resilient full-chain behavior.
        set_config({"data_vendors": {"core_stock_apis": "default"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def _route_method(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_optional_category_degrades_instead_of_raising(self):
        # An optional enrichment vendor (FRED macro) that raises must NOT abort
        # the run — the router returns a sentinel so the analysis proceeds.
        set_config({"data_vendors": {"macro_data": "fred"}})
        with self._route_method(
            "get_macro_indicators", {"fred": _raises(ValueError("FRED 400: bad series"))}
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("macro_data", result)

    def test_core_category_still_raises_on_error(self):
        # A core category (single configured vendor) propagates the error so a
        # broken primary is loud, not silently degraded.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route({"yfinance": _raises(ValueError("boom"))}), \
                self.assertRaises(ValueError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


if __name__ == "__main__":
    unittest.main()


def test_marketstack_supplies_indicators_snapshot_prices_and_returns_without_yahoo(monkeypatch, tmp_path):
    """Exercise all price consumers through HTTP pagination and one shared cache."""
    from types import SimpleNamespace

    import pandas as pd

    from tradingagents.dataflows import marketstack, yahoo
    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    set_config({"data_cache_dir": str(tmp_path), "data_vendors": {
        "core_stock_apis": "marketstack", "technical_indicators": "local", "instrument_data": "marketstack",
    }})
    monkeypatch.setenv("MARKETSTACK_API_KEY", "test-secret")
    monkeypatch.setattr(yahoo.yf, "Ticker", mock.Mock(side_effect=AssertionError("Unexpected Yahoo call")))
    dates = pd.bdate_range(end="2026-06-10", periods=1005)
    calls = []

    def respond(url, params, timeout):
        assert url == "https://api.marketstack.com/v2/eod"
        assert params["limit"] == 1000
        calls.append((params["symbols"], params["offset"]))
        rows = [{"date": str(day.date()), "symbol": params["symbols"], "exchange": "XNAS",
                 "open": 9999, "close": 9999, "adj_open": 100 + i, "adj_high": 102 + i,
                 "adj_low": 99 + i, "adj_close": 101 + i, "volume": 1000000, "adj_volume": 500000}
                for i, day in enumerate(dates)]
        page = rows[params["offset"]:params["offset"] + 1000]
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {
            "pagination": {"offset": params["offset"], "total": len(rows)}, "data": page,
        })

    monkeypatch.setattr(marketstack.requests, "get", respond)
    indicators = interface.route_to_vendor("get_indicators", "AAPL", "rsi", "2026-06-10", 7)
    assert "marketstack" in indicators and "2026-06-10" in indicators
    prices = interface.route_to_vendor("get_stock_data", "AAPL", "2026-06-01", "2026-06-10")
    snapshot = build_verified_market_snapshot("AAPL", "2026-06-10")
    assert "marketstack" in prices and "marketstack" in snapshot
    assert "9999" not in prices and "1000000" in prices
    assert calls == [("AAPL", 0), ("AAPL", 1000)]
    graph = SimpleNamespace(config=get_config())
    result = TradingAgentsGraph._fetch_returns(graph, "AAPL", "2026-05-04", benchmark="SPY")
    assert result[0] is not None and result[1] == 0.0 and result[2] == 5
    assert {symbol for symbol, _ in calls} == {"AAPL", "SPY"}
    yahoo.yf.Ticker.assert_not_called()


def test_structured_route_honors_tool_override_and_explicit_fallback(monkeypatch, tmp_path):
    import pandas as pd

    from tradingagents.dataflows import market_data
    from tradingagents.dataflows.errors import VendorRateLimitError

    set_config({"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "yfinance"},
                "tool_vendors": {"get_stock_data": "marketstack"}})
    yahoo = mock.Mock(side_effect=AssertionError("Unexpected Yahoo"))
    marketstack = mock.Mock(side_effect=VendorRateLimitError("quota"))
    monkeypatch.setitem(market_data.OHLCV_ADAPTERS, "yfinance", yahoo)
    monkeypatch.setitem(market_data.OHLCV_ADAPTERS, "marketstack", marketstack)
    with pytest.raises(VendorRateLimitError):
        market_data.get_ohlcv("MSFT", "2026-06-01", "2026-06-10")
    yahoo.assert_not_called()
    set_config({"tool_vendors": {"get_stock_data": "marketstack,yfinance"}})
    yahoo.side_effect = None
    yahoo.return_value = pd.DataFrame({"Date": ["2026-06-10"], "Close": [10.]})
    frame = market_data.get_ohlcv("MSFT", "2026-06-01", "2026-06-10")
    assert frame.attrs["provider"] == "yfinance"
    yahoo.assert_called_once()


def test_vendor_cache_isolation_and_old_yahoo_cache_is_not_reused(monkeypatch, tmp_path):
    import pandas as pd

    from tradingagents.dataflows import market_data

    set_config({"data_cache_dir": str(tmp_path)})
    (tmp_path / "AAPL-YFin-data-2021-06-10-2026-06-11.csv").write_text("Date,Close\n2026-06-10,9999\n")
    for provider, value in [("yfinance", 10.), ("marketstack", 20.)]:
        fetch = mock.Mock(return_value=pd.DataFrame({"Date": ["2026-06-10"], "Close": [value]}))
        monkeypatch.setitem(market_data.OHLCV_ADAPTERS, provider, fetch)
        first = market_data.get_ohlcv("AAPL", "2026-06-01", "2026-06-10", vendor=provider)
        second = market_data.get_ohlcv("AAPL", "2026-06-05", "2026-06-10", vendor=provider)
        assert first["Close"].iloc[-1] == second["Close"].iloc[-1] == value
        fetch.assert_called_once()


def test_alpha_vantage_structured_adjustment_uses_raw_volume(monkeypatch):
    from tradingagents.dataflows import alpha_vantage_stock

    monkeypatch.setattr(alpha_vantage_stock, "_make_api_request", lambda *args: (
        "timestamp,open,high,low,close,adjusted_close,volume\n"
        "2026-06-10,100,110,90,100,50,1000\n"
    ))
    frame = alpha_vantage_stock.get_ohlcv("AAPL", "2026-06-01", "2026-06-10")
    assert frame.iloc[0]["Open"] == 50 and frame.iloc[0]["High"] == 55
    assert frame.iloc[0]["Close"] == 50 and frame.iloc[0]["Volume"] == 1000


def test_fmp_all_report_routes_indicators_identity_and_returns_without_yahoo(fmp_http, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    from tradingagents.agents.utils.agent_utils import resolve_instrument_identity
    from tradingagents.dataflows import market_data, yahoo
    from tradingagents.dataflows.config import get_config
    from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    set_config({"data_cache_dir": str(tmp_path), "data_vendors": {
        "core_stock_apis": "fmp", "technical_indicators": "local", "instrument_data": "fmp",
        "fundamental_data": "fmp", "news_data": "fmp",
    }})
    yahoo_call = mock.Mock(side_effect=AssertionError("Unexpected Yahoo"))
    monkeypatch.setattr(yahoo.yf, "Ticker", yahoo_call)
    monkeypatch.setattr(yahoo.yf, "Search", yahoo_call)

    def respond(endpoint, params):
        symbol = params.get("symbol", "AAPL")
        if endpoint.startswith("historical-price-eod/"):
            dates = pd.bdate_range("2021-06-10", "2026-06-10")
            return 200, [{"symbol": symbol, "date": str(day.date()), "adjOpen": 100 + i,
                          "adjHigh": 102 + i, "adjLow": 99 + i, "adjClose": 101 + i,
                          "volume": 999 if endpoint.endswith("dividend-adjusted") else 1000000}
                         for i, day in enumerate(dates) if params["from"] <= str(day.date()) <= params["to"]]
        if endpoint == "profile":
            return 200, [{"symbol": symbol, "companyName": "Apple", "exchange": "NASDAQ",
                          "isEtf": False, "isFund": False, "isActivelyTrading": True}]
        if endpoint in {"quote", "ratios-ttm", "key-metrics-ttm"}:
            return 200, [{"symbol": symbol, "priceToEarningsRatioTTM": 20, "marketCap": 12345}]
        if endpoint.endswith("statement"):
            return 200, [{"symbol": symbol, "date": "2026-03-31", "filingDate": "2026-05-01", "reportedCurrency": "USD"}]
        if endpoint.startswith("news/"):
            return 200, [{"symbol": "AAPL", "title": "News", "publishedDate": "2026-06-10", "url": "https://example.com/story"}]
        assert endpoint == "insider-trading/search"
        return 200, []

    fmp_http.respond = respond
    assert "fmp" in interface.route_to_vendor("get_indicators", "AAPL", "rsi", "2026-06-10", 7)
    assert len(fmp_http.calls) == 2
    assert fmp_http.calls[0][1]["from"] == "2021-06-10"
    assert "fmp" in interface.route_to_vendor("get_stock_data", "AAPL", "2026-06-01", "2026-06-10")
    assert "fmp" in build_verified_market_snapshot("AAPL", "2026-06-10")
    assert len(fmp_http.calls) == 2  # one five-year snapshot serves all consumers
    assert resolve_instrument_identity("AAPL")["company_name"] == "Apple"
    assert market_data.get_instrument_info("AAPL")["provider"] == "fmp"
    for method, args in (("get_fundamentals", ("AAPL",)),
                         ("get_balance_sheet", ("AAPL", "quarterly", "2026-06-10")),
                         ("get_cashflow", ("AAPL", "annual", "2026-06-10")),
                         ("get_income_statement", ("AAPL", "quarterly", "2026-06-10")),
                         ("get_news", ("AAPL", "2026-06-01", "2026-06-10")),
                         ("get_global_news", ("2026-06-10",)), ("get_insider_transactions", ("AAPL",))):
        assert "Provider: fmp" in interface.route_to_vendor(method, *args)
    result = TradingAgentsGraph._fetch_returns(SimpleNamespace(config=get_config()), "AAPL", "2026-05-04", benchmark="SPY")
    assert result[0] is not None and result[1] == 0.0 and result[2] == 5
    assert {params["symbol"] for endpoint, params in fmp_http.calls if endpoint.startswith("historical")} == {"AAPL", "SPY"}
    yahoo_call.assert_not_called()


def test_fmp_caches_and_explicit_fallback_are_provider_isolated(monkeypatch, tmp_path):
    import pandas as pd

    from tradingagents.dataflows import market_data
    from tradingagents.dataflows.errors import VendorNotConfiguredError

    config = {"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "yfinance"},
              "tool_vendors": {"get_stock_data": "fmp"}}
    fetches = {}
    for provider, value in (("fmp", 10), ("marketstack", 20), ("yfinance", 30)):
        fetches[provider] = mock.Mock(return_value=pd.DataFrame({"Date": ["2026-06-10"], "Close": [value]}))
        monkeypatch.setitem(market_data.OHLCV_ADAPTERS, provider, fetches[provider])
        assert market_data.get_ohlcv("BRK.B", "2026-06-01", "2026-06-10", config=config, vendor=provider)["Close"].iloc[0] == value
    assert market_data.get_ohlcv("BRK-B", "2026-06-05", "2026-06-10", config=config)["Close"].iloc[0] == 10
    assert fetches["fmp"].call_count == 1
    fetches["fmp"].side_effect = VendorNotConfiguredError("FMP unavailable")
    config["tool_vendors"]["get_stock_data"] = "fmp,marketstack"
    assert market_data.get_ohlcv("MSFT", "2026-06-01", "2026-06-10", config=config).attrs["provider"] == "marketstack"
    assert fetches["yfinance"].call_count == 1
