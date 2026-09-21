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


def test_etf_fmp_profile_holdings_route_without_yahoo(fmp_http, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from tradingagents.dataflows import yahoo

    today = datetime.now(ZoneInfo("America/New_York")).date()
    future = str(today + timedelta(days=1))
    yahoo_call = mock.Mock(side_effect=AssertionError("Unexpected Yahoo"))
    monkeypatch.setattr(yahoo.yf, "Ticker", yahoo_call)
    set_config({"data_vendors": {"fundamental_data": "fmp"}})

    def respond(endpoint, params):
        assert params == {"symbol": "SPY"}
        if endpoint == "etf/info":
            return 200, [{"symbol": "SPY", "name": "Test fund", "assetClass": "Equity",
                          "expenseRatio": 0.09, "nav": 600, "navCurrency": "USD",
                          "updatedAt": str(today), "holdingsCount": 500,
                          "sectorsList": [{"industry": "Technology", "exposure": 30}]}]
        assert endpoint == "etf/holdings"
        return 200, [{"symbol": "SPY", "asset": f"H{i}", "weightPercentage": i,
                      "updatedAt": str(today)} for i in range(1, 12)] + [
                          {"symbol": "SPY", "asset": "FUTURE", "weightPercentage": 99, "updatedAt": future}]

    fmp_http.respond = respond
    report = interface.route_to_vendor("get_etf_fundamentals", "spy", str(today))
    assert "Provider: fmp" in report and "Test fund" in report
    assert "0.09" in report and "USD" in report and "Technology" in report
    assert "10 of 11 returned" in report and "H11,N/A,11" in report
    assert "H1,N/A,1," not in report and "FUTURE" not in report
    assert "Tracking error: N/A" in report
    assert "Current snapshot only" in report and str(today) in report
    assert len(fmp_http.calls) == 2
    yahoo_call.assert_not_called()


@pytest.mark.parametrize("problem", ["historical", "future_profile", "wrong_symbol", "empty_holdings", "bad_weight", "permission"])
def test_etf_fmp_unavailable_data_and_permissions_fail_without_yahoo(fmp_http, monkeypatch, problem):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from tradingagents.dataflows import fmp, yahoo
    from tradingagents.dataflows.errors import VendorError

    today = datetime.now(ZoneInfo("America/New_York")).date()
    monkeypatch.setattr(yahoo.yf, "Ticker", mock.Mock(side_effect=AssertionError("Unexpected Yahoo")))

    def respond(endpoint, params):
        if endpoint == "etf/info":
            return 200, [{"symbol": "QQQ" if problem == "wrong_symbol" else "SPY", "name": "Test fund",
                          "updatedAt": str(today + timedelta(days=1) if problem == "future_profile" else today)}]
        if problem == "permission":
            return 403, {"Error Message": "FMP-TEST-SECRET subscription required"}
        if problem == "empty_holdings":
            return 200, []
        return 200, [{"symbol": "SPY", "asset": "AAPL", "weightPercentage": None, "updatedAt": str(today)}]

    fmp_http.respond = respond
    with pytest.raises((NoMarketDataError, VendorError)) as error:
        fmp.get_etf_fundamentals("SPY", str(today - timedelta(days=1) if problem == "historical" else today))
    assert "FMP-TEST-SECRET" not in str(error.value)
    assert len(fmp_http.calls) <= 2  # no retry for missing data or denied plan
    if problem == "historical":
        assert not fmp_http.calls
    yahoo.yf.Ticker.assert_not_called()


def test_etf_tool_override_and_explicit_fallback(monkeypatch):
    from tradingagents.dataflows.errors import VendorNotConfiguredError

    fmp_call = mock.Mock(side_effect=VendorNotConfiguredError("FMP permissions unavailable"))
    yahoo_call = mock.Mock(return_value="Yahoo ETF data")
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_etf_fundamentals", {"fmp": fmp_call, "yfinance": yahoo_call})
    set_config({"data_vendors": {"fundamental_data": "fmp"}})
    with pytest.raises(VendorNotConfiguredError):
        interface.route_to_vendor("get_etf_fundamentals", "SPY", "2026-09-19")
    yahoo_call.assert_not_called()
    set_config({"tool_vendors": {"get_etf_fundamentals": "fmp,yfinance"}})
    assert interface.route_to_vendor("get_etf_fundamentals", "SPY", "2026-09-19") == "Yahoo ETF data"
    assert fmp_call.call_count == 2 and yahoo_call.call_count == 1


def test_etf_yahoo_fund_data_for_local_default(monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from zoneinfo import ZoneInfo

    import pandas as pd

    from tradingagents.dataflows import yahoo

    today = datetime.now(ZoneInfo("America/New_York")).date()
    ticker = mock.Mock()
    ticker.get_info.return_value = {"quoteType": "ETF", "longName": "Test fund", "currency": "USD"}
    ticker.funds_data = SimpleNamespace(
        description="Index exposure", fund_overview={"family": "Manager"},
        fund_operations=pd.DataFrame({"SPY": [0.0009]}, index=["Annual Report Expense Ratio"]),
        top_holdings=pd.DataFrame({"Name": ["Apple"], "Holding Percent": [0.07]}, index=["AAPL"]),
        sector_weightings={"technology": 0.3}, asset_classes={"stockPosition": 0.99},
    )
    fetch = mock.Mock(return_value=ticker)
    monkeypatch.setattr(yahoo.yf, "Ticker", fetch)
    report = interface.route_to_vendor("get_etf_fundamentals", "SPY", str(today))
    assert "Provider: yfinance" in report and "0.0009" in report and "AAPL" in report
    assert "dates unavailable" in report and "fraction" in report
    with pytest.raises(NoMarketDataError):
        yahoo.get_etf_fundamentals("SPY", str(today - timedelta(days=1)))
    fetch.assert_called_once_with("SPY")


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


def test_industry_sec_cutoff_units_and_original_filing_metadata():
    import json

    from tradingagents.dataflows.industry.documents import sec_disclosures

    calls = []
    paragraph = "Our business segment manufactures products for customers with supplier capacity and inventory risks. " * 4
    index = {"cik": "1045810", "name": "NVIDIA", "filings": {"recent": {
        "form": ["10-Q", "10-K", "10-Q"], "filingDate": ["2026-08-26", "2026-02-25", "2026-05-20"],
        "reportDate": ["2026-07-26", "2026-01-25", "2026-04-26"],
        "accessionNumber": ["0001045810-26-000075", "0001045810-26-000010", "0001045810-26-000045"],
        "primaryDocument": ["future.htm", "annual.htm", "quarter.htm"],
    }, "files": []}}

    class Client:
        blocked = set()

        def fetch(self, url, **kwargs):
            calls.append(url)
            if "submissions" in url:
                body = json.dumps(index).encode()
            elif "companyfacts" in url:
                body = json.dumps({"cik": 1045810, "facts": {"us-gaap": {"InventoryNet": {"units": {"USD": [
                    {"end": "2026-04-26", "filed": "2026-05-20", "val": 100},
                    {"end": "2026-04-26", "filed": "2026-08-26", "val": 999},
                ]}}}}}).encode()
            else:
                body = f"<html><body><p>{paragraph}</p></body></html>".encode()
            return {"body": body, "retrieved_at": "2026-09-20T00:00:00Z"}

    result = sec_disclosures(Client(), "1045810", "2026-06-01")
    assert not any("future.htm" in url for url in calls)
    assert len(result["evidence"]) == 3
    assert result["evidence"][0]["published_at"] == "2026-02-25"
    metric = result["evidence"][-1]["data"][0]
    assert metric["unit"] == "USD" and [row["val"] for row in metric["observations"]] == [100]


def test_industry_sec_older_submissions_shard():
    import json

    from tradingagents.dataflows.industry.documents import sec_disclosures

    class Client:
        blocked = set()

        def fetch(self, url, **kwargs):
            if "-submissions-" in url:
                body = {"form": ["10-K"], "filingDate": ["2020-02-20"], "reportDate": ["2020-01-01"],
                        "accessionNumber": ["0001045810-20-000001"], "primaryDocument": ["old.htm"]}
            elif "submissions" in url:
                body = {"cik": 1045810, "filings": {"recent": {"form": [], "filingDate": []},
                        "files": [{"name": "CIK0001045810-submissions-001.json", "filingFrom": "2019-01-01", "filingTo": "2021-01-01"}]}}
            else:
                return {"body": b"<p>Our business segment has long term supplier arrangements and customer concentration risks in manufacturing.</p>", "retrieved_at": "now"}
            return {"body": json.dumps(body).encode(), "retrieved_at": "now"}

    result = sec_disclosures(Client(), "1045810", "2020-06-01", include_facts=False)
    assert len(result["evidence"]) == 1 and result["evidence"][0]["period"] == "2020-01-01"


def test_industry_public_file_parsers_preserve_units_and_revisions():
    import io

    from openpyxl import Workbook

    from tradingagents.dataflows.industry.indicators import parse_census, parse_eia, parse_wsts

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Monthly Data"
    sheet.append(["All numbers are in 1000 US$."])
    sheet.append([2026])
    sheet.append(["Worldwide", 100, 120, None])
    sheet.append(["Americas", 20, 25, None])
    stream = io.BytesIO()
    workbook.save(stream)
    result = parse_wsts(stream.getvalue(), "2026-02-28")
    assert result["unit"] == "1000 USD" and result["latest_period"] == "2026-02"
    assert len(result["observations"]) == 4
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Estimates in millions of dollars"])
    sheet.append(["Industry", "", "Seasonally Adjusted", "", "Not Seasonally Adjusted"])
    sheet.append(["", "", "July", "June", "May"])
    sheet.append(["", "", "2026p", "2026r", "2026"])
    sheet.append(["Construction machinery", "", 100, 90, 80])
    sheet.append(["p Preliminary; r Revised. Semiconductor new orders excluded."])
    stream = io.BytesIO()
    workbook.save(stream)
    result = parse_census(stream.getvalue(), "new_orders", "2026-09-20", "construction machinery")
    assert result["latest_period"] == "2026-07" and len(result["rows"]) == 1
    assert "2026p" in str(result["headers_by_column_position"])
    assert "Semiconductor" in result["footnotes"][0]
    data = b'STUB_1,STUB_2,9/11/26,9/4/26\nRefiner Inputs,Crude Oil Inputs,"17,330","17,586"\nRefiner Inputs,Percent Utilization,96.8,97.8\n'
    result = parse_eia(data, "refining", "2026-09-20")
    assert result["metrics"][0]["unit"] == "thousand barrels per day"
    assert result["metrics"][1]["unit"] == "percent"
    assert result["metrics"][0]["observations"][0]["value"] == 17330


def test_industry_historical_official_files_do_not_fetch_current_vintages():
    from unittest.mock import Mock

    from tradingagents.dataflows.industry.indicators import collect_indicators

    client = Mock()
    result = collect_indicators(client, "Semiconductors", "2020-01-15", {"wsts"}, live=False)
    assert not result["evidence"] and result["failures"][0]["source"] == "wsts"
    client.fetch.assert_not_called()
    result = collect_indicators(client, "Software - Infrastructure", "2026-09-20", {"eia", "census", "wsts"}, live=True)
    assert not result["evidence"]
    client.fetch.assert_not_called()


def test_industry_related_companies_are_bounded_across_calls(monkeypatch, tmp_path):
    from tradingagents.dataflows.industry import IndustryResearch
    from tradingagents.dataflows.industry.transport import SourceUnavailable

    research = IndustryResearch("NVDA", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": [], "industry_max_related_companies": 3})
    research.context = {"evidence": [], "candidate_peers": [{"symbol": s} for s in ("AMD", "TSM", "MU", "AAPL")]}
    monkeypatch.setattr(research, "_profile", lambda symbol: {"companyName": symbol})
    monkeypatch.setattr(research, "_company", lambda symbol, **kwargs: {"ticker": symbol, "evidence": []})
    assert len(research.get_related_company_evidence(["AMD", "TSM"])["companies"]) == 2
    second = research.get_related_company_evidence(["MU", "AAPL"])
    assert second["checked_symbols"] == ["AMD", "MU", "TSM"]
    assert second["rejected"][0]["symbol"] == "AAPL"
    assert len(research.get_related_company_evidence(["AMD"])["companies"]) == 1
    with pytest.raises(SourceUnavailable):
        research.get_related_company_evidence(["AMD", "TSM", "MU", "AAPL"])


@pytest.mark.parametrize("status", [403, 429])
def test_industry_sec_denial_stops_requests_and_does_not_log_contact(monkeypatch, tmp_path, caplog, status):
    import requests

    from tradingagents.dataflows.industry.transport import DocumentClient, SourceUnavailable

    monkeypatch.setenv("SEC_USER_AGENT", "IndustryTests private-contact@example.com")
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 443))])
    calls = []

    def respond(session, url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = status
        response._content = b"denied"
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "get", respond)
    client = DocumentClient({"data_cache_dir": str(tmp_path)})
    for _ in range(2):
        with pytest.raises(SourceUnavailable):
            client.fetch("https://data.sec.gov/submissions/CIK0001045810.json", source="sec")
    assert len(calls) == 1 and "private-contact" not in caplog.text


def test_industry_extract_preserves_named_relationships_in_later_sections():
    import json

    from tradingagents.dataflows.industry.documents import extract_document

    noise = '<p>General supplier inventory risk and customer demand may change over time in our business operations.</p>' * 100
    html = (noise + '<h2>Manufacturing</h2><p>We utilize foundries such as Named Foundry Company to produce wafers. '
            'We purchase memory from Named Memory Corporation and engage independent contract manufacturers.</p>'
            '<h2>Our current competitors include:</h2><p>Suppliers of hardware and software for processors such as Named Competitor '
            'Corporation provide competing platforms for our principal markets.</p>')
    result = extract_document({"body": html.encode()}, budget=8000)
    assert "Named Foundry Company" in json.dumps(result["supply_chain"])
    assert "Named Competitor" in json.dumps(result["competition"])


def test_industry_fmp_entitlement_failure_is_per_dataset(fmp_http, tmp_path):
    from tradingagents.dataflows.errors import VendorNotConfiguredError
    from tradingagents.dataflows.industry import IndustryResearch

    def respond(endpoint, params):
        if endpoint == "revenue-product-segmentation":
            return 402, {"error": "subscription upgrade required"}
        return 200, [{"symbol": "CAT", "companyName": "Caterpillar"}]

    fmp_http.respond = respond
    research = IndustryResearch("CAT", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": ["fmp"]})
    with pytest.raises(VendorNotConfiguredError):
        research._fmp_rows("statements", "revenue_product_segmentation", symbol="CAT")
    assert research._profile("CAT")["companyName"] == "Caterpillar"
    assert research._fmp_rows("company", "stock_peers", symbol="CAT")

    research = IndustryResearch("CAT", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": ["fmp"]})
    research.live = True
    context = research.get_industry_context()
    failure = next(row for row in context["failures"] if row["dataset"] == "revenue-product-segmentation")
    assert failure["code"] == "subscription_required" and "HTTP 402" in failure["reason"]
    assert "not configured" not in failure["reason"]
    assert context["company"] == "Caterpillar"


def test_industry_fmp_diagnostics_distinguish_config_and_permissions(fmp_http, monkeypatch, tmp_path, caplog):
    from tradingagents.dataflows.errors import VendorError, VendorNotConfiguredError
    from tradingagents.dataflows.industry import IndustryResearch, source_failure

    for status, code in ((401, "authentication_failed"), (402, "subscription_required"), (403, "access_denied")):
        fmp_http.respond = lambda endpoint, params, status=status: (status, {"error": "apikey=upstream-secret"})
        research = IndustryResearch("CRCL", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": ["fmp"]})
        context = research.get_industry_context()
        failure = next(row for row in context["failures"] if row["dataset"] == "profile")
        assert failure["code"] == code and str(status) in failure["reason"]
        assert "upstream-secret" not in str(context) + caplog.text

    monkeypatch.delenv("FMP_API_KEY")
    calls_before = len(fmp_http.calls)
    research = IndustryResearch("CRCL", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": ["fmp"]})
    context = research.get_industry_context()
    failure = next(row for row in context["failures"] if row["dataset"] == "profile")
    assert failure["code"] == "missing_api_key" and len(fmp_http.calls) == calls_before
    assert source_failure("fmp", VendorNotConfiguredError("Install the project's pinned fmpsdk dependency"), dataset="profile")["code"] == "missing_dependency"
    assert source_failure("fmp", VendorError("secret"), dataset="profile")["reason"] == "VendorError"


def test_industry_sec_only_related_lookup_and_etf_guard(monkeypatch, tmp_path):
    import json

    from tradingagents.dataflows.industry import IndustryResearch, UnsupportedIndustryAsset

    research = IndustryResearch("NVDA", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": ["sec"]})
    research.context = {"evidence": [{"data": {"excerpts": {"supply_chain": [
        {"text": "We purchase memory from Micron Technology, Inc."},
    ]}}}], "candidate_peers": []}
    monkeypatch.setattr(research.client, "fetch", lambda url, **kwargs: {"body": json.dumps({
        "0": {"ticker": "MU", "cik_str": 723125, "title": "MICRON TECHNOLOGY INC"},
    }).encode()})
    monkeypatch.setattr("tradingagents.dataflows.industry.sec_disclosures", lambda *a, **k: {
        "issuer": "MICRON", "cik": "0000723125", "sic": "3674", "sic_description": "Semiconductors",
        "evidence": [], "failures": [],
    })
    result = research.get_related_company_evidence(["MU"])
    assert not result["rejected"]
    assert result["companies"][0]["cik"] == "0000723125"
    assert not result["companies"][0]["failures"]
    research = IndustryResearch("SPY", "2026-01-15", {"data_cache_dir": str(tmp_path), "industry_sources": ["fmp", "sec"]})
    monkeypatch.setattr(research, "_fmp_rows", lambda *a, **k: [{"symbol": "SPY", "isEtf": True}])
    monkeypatch.setattr(research.client, "fetch", lambda *a, **k: pytest.fail("ETF must not fetch SEC documents"))
    with pytest.raises(UnsupportedIndustryAsset):
        research.get_industry_context()


@pytest.mark.parametrize("failure", ["timeout", "server_error"])
def test_industry_transport_retries_once_and_caches_without_contact(monkeypatch, tmp_path, failure):
    import requests

    from tradingagents.dataflows.industry.transport import DocumentClient, SourceUnavailable

    monkeypatch.setenv("SEC_USER_AGENT", "IndustryTests contact-private@example.com")
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("8.8.8.8", 443))])
    monkeypatch.setattr("tradingagents.dataflows.industry.transport.time.sleep", lambda seconds: None)
    statuses = [failure, 200, failure, failure]
    calls = []

    def respond(session, url, **kwargs):
        calls.append(url)
        status = statuses.pop(0)
        if status == "timeout":
            raise requests.Timeout("upstream-private-details")
        response = requests.Response()
        response.status_code = 503 if status == "server_error" else status
        response._content = b'{"public": true}'
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "get", respond)
    client = DocumentClient({"data_cache_dir": str(tmp_path)})
    url = "https://data.sec.gov/submissions/CIK0001045810.json"
    first = client.fetch(url, source="sec")
    assert first["body"] == b'{"public": true}' and len(calls) == 2
    assert client.fetch(url, source="sec")["cache_hit"] and len(calls) == 2
    for path in (tmp_path / "industry").glob("*.json"):
        assert "contact-private" not in path.read_text()
    with pytest.raises(SourceUnavailable) as error:
        client.fetch("https://data.sec.gov/submissions/CIK0000789019.json", source="sec")
    assert len(calls) == 4 and "upstream-private-details" not in str(error.value)


def test_industry_missing_sec_contact_does_not_make_requests(monkeypatch, tmp_path):
    import requests

    from tradingagents.dataflows.industry.transport import DocumentClient, SourceUnavailable

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("COMMERCE_SUPPORT_EMAIL", raising=False)
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: pytest.fail("Missing SEC contact must fail before I/O"))
    with pytest.raises(SourceUnavailable, match="contact email"):
        DocumentClient({"data_cache_dir": str(tmp_path)}).fetch(
            "https://data.sec.gov/submissions/CIK0001045810.json", source="sec")
