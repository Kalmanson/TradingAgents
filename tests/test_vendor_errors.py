"""The vendor data-error hierarchy: every "vendor couldn't return usable data"
condition derives from VendorError, so the router catches base types and any
vendor slots in without new handling.
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.alpha_vantage_common import (
    AlphaVantageNotConfiguredError,
    AlphaVantageRateLimitError,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from tradingagents.dataflows.fred import FredNotConfiguredError


@pytest.mark.parametrize("status,body,expected,attempts", [
    (401, {"error": "invalid api key"}, VendorNotConfiguredError, 1),
    (402, {"error": "premium"}, VendorNotConfiguredError, 1),
    (403, {"error": "permission denied"}, VendorNotConfiguredError, 1),
    (429, {"error": "daily limit reached"}, VendorRateLimitError, 1),
    (429, {"error": "monthly quota exhausted"}, VendorRateLimitError, 1),
    (200, {"Error Message": "Limit Reach. Please upgrade your plan"}, VendorRateLimitError, 1),
    (200, [{"Error Message": "Invalid API KEY"}], VendorNotConfiguredError, 1),
    (200, {"Error Message": "too many requests"}, VendorRateLimitError, 3),
    (429, {"error": "rate limited"}, VendorRateLimitError, 3),
    (503, {}, VendorError, 3),
    (400, {"error": "invalid request"}, VendorError, 1),
    (404, {}, NoMarketDataError, 1),
    (200, "not a record list", VendorError, 1),
])
def test_fmp_real_sdk_errors_retries_and_redaction(fmp_http, monkeypatch, caplog, status, body, expected, attempts):
    import json
    import logging

    from tradingagents.dataflows import fmp

    if isinstance(body, dict):
        body = {**body, "debug": "FMP-TEST-SECRET"}
    fmp_http.respond = lambda endpoint, params: (status, body)
    sleep = mock.Mock()
    monkeypatch.setattr(fmp.time, "sleep", sleep)
    caplog.set_level(logging.INFO)
    with pytest.raises(expected) as exc:
        fmp.get_instrument_info("AAPL")
    assert len(fmp_http.calls) == attempts
    assert sleep.call_count == attempts - 1
    assert "FMP-TEST-SECRET" not in str(exc.value) + caplog.text
    assert json.dumps(body) not in caplog.text
    assert "provider=fmp" in caplog.text


def test_fmp_timeout_recovery_and_missing_key(fmp_http, monkeypatch):
    import requests

    from tradingagents.dataflows import fmp

    sleep = mock.Mock()
    monkeypatch.setattr(fmp.time, "sleep", sleep)

    def respond(endpoint, params):
        if len(fmp_http.calls) <= 2:
            raise requests.Timeout("secret: FMP-TEST-SECRET")
        return 200, [{"symbol": "AAPL", "companyName": "Apple", "exchange": "NASDAQ"}]

    fmp_http.respond = respond
    assert fmp.get_instrument_info("AAPL")["company_name"] == "Apple"
    assert len(fmp_http.calls) == 3 and sleep.call_count == 2
    fmp_http.calls.clear()
    monkeypatch.delenv("FMP_API_KEY")
    with pytest.raises(VendorNotConfiguredError):
        fmp.get_instrument_info("AAPL")
    assert not fmp_http.calls


def test_fmp_single_vendor_failures_never_call_yahoo(fmp_http, monkeypatch):
    from tradingagents.dataflows import market_data, yahoo

    set_config({"data_vendors": {"core_stock_apis": "fmp", "technical_indicators": "local",
                                "instrument_data": "fmp", "fundamental_data": "fmp", "news_data": "fmp"}})
    yahoo_call = mock.Mock(side_effect=AssertionError("Unexpected Yahoo"))
    monkeypatch.setattr(yahoo.yf, "Ticker", yahoo_call)
    monkeypatch.setattr(yahoo.yf, "Search", yahoo_call)
    fmp_http.respond = lambda endpoint, params: (402, {"Error Message": "FMP-TEST-SECRET"})
    for method, args in (("get_stock_data", ("AAPL", "2026-06-01", "2026-06-10")),
                         ("get_indicators", ("AAPL", "rsi", "2026-06-10", 7)),
                         ("get_fundamentals", ("AAPL",)),
                         ("get_income_statement", ("AAPL",)),
                         ("get_balance_sheet", ("AAPL",)), ("get_cashflow", ("AAPL",)),
                         ("get_news", ("AAPL", "2026-06-01", "2026-06-10")),
                         ("get_global_news", ("2026-06-10",)),
                         ("get_insider_transactions", ("AAPL",))):
        with pytest.raises(VendorNotConfiguredError):
            interface.route_to_vendor(method, *args)
    with pytest.raises(VendorNotConfiguredError):
        market_data.get_instrument_info("AAPL")
    yahoo_call.assert_not_called()


@pytest.mark.unit
class HierarchyTests(unittest.TestCase):
    def test_all_conditions_derive_from_vendor_error(self):
        for cls in (NoMarketDataError, VendorRateLimitError, VendorNotConfiguredError):
            self.assertTrue(issubclass(cls, VendorError))

    def test_not_configured_is_still_a_value_error(self):
        # Back-compat: existing `except ValueError` callers keep working.
        self.assertTrue(issubclass(VendorNotConfiguredError, ValueError))

    def test_vendor_named_errors_subclass_the_generic_bases(self):
        self.assertTrue(issubclass(AlphaVantageRateLimitError, VendorRateLimitError))
        self.assertTrue(issubclass(AlphaVantageNotConfiguredError, VendorNotConfiguredError))
        self.assertTrue(issubclass(FredNotConfiguredError, VendorNotConfiguredError))
        # ... and therefore still ValueErrors
        self.assertTrue(issubclass(FredNotConfiguredError, ValueError))

    def test_symbol_utils_reexports_no_market_data_error(self):
        from tradingagents.dataflows.symbol_utils import (
            NoMarketDataError as ReExported,
        )
        self.assertIs(ReExported, NoMarketDataError)


@pytest.mark.unit
class RouterHandlesBaseTypesTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_rate_limit_subclass_caught_by_base(self):
        # A vendor-named rate-limit error skips to the next vendor in the chain.
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})

        def _throttled(*a, **k):
            raise AlphaVantageRateLimitError("slow down")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _throttled, "yfinance": lambda *a, **k: "YF"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(out, "YF")

    def test_not_configured_falls_through_to_next_vendor(self):
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage,yfinance"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("no key")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _unconfigured, "yfinance": lambda *a, **k: "YF"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(out, "YF")

    def test_sole_unconfigured_vendor_surfaces_the_error(self):
        # With no fallback, the not-configured condition must surface (not vanish).
        set_config({"data_vendors": {"core_stock_apis": "alpha_vantage"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("no key")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"alpha_vantage": _unconfigured}},
            clear=False,
        ), self.assertRaises(AlphaVantageNotConfiguredError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("status,error,expected,attempts", [
    (401, {"code": 101, "type": "invalid_access_key"}, VendorNotConfiguredError, 1),
    (403, {"type": "function_access_restricted"}, VendorNotConfiguredError, 1),
    (429, {"type": "usage_limit_reached"}, VendorRateLimitError, 1),
    (200, {"code": 104}, VendorRateLimitError, 1),
    (429, {"type": "rate_limit_reached"}, VendorRateLimitError, 3),
    (503, {}, VendorError, 3),
])
def test_marketstack_retry_policy_and_redaction(monkeypatch, caplog, status, error, expected, attempts):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tradingagents.dataflows import marketstack

    monkeypatch.setenv("MARKETSTACK_API_KEY", "DO-NOT-LOG-THIS-KEY")
    response = SimpleNamespace(status_code=status, headers={"Retry-After": "1"},
                               json=lambda: {"error": {**error, "info": "DO-NOT-LOG-THIS-KEY"}} if error else {})
    request = Mock(return_value=response)
    sleep = Mock()
    monkeypatch.setattr(marketstack.requests, "get", request)
    monkeypatch.setattr(marketstack.time, "sleep", sleep)
    with pytest.raises(expected) as failure:
        marketstack.get_ohlcv("AAPL", "2026-06-01", "2026-06-10")
    assert request.call_count == attempts and sleep.call_count == attempts - 1
    assert "DO-NOT-LOG-THIS-KEY" not in str(failure.value) + caplog.text


def test_marketstack_missing_key_does_not_make_request(monkeypatch):
    from unittest.mock import Mock

    from tradingagents.dataflows import marketstack

    monkeypatch.delenv("MARKETSTACK_API_KEY", raising=False)
    request = Mock(side_effect=AssertionError("Should not request without a key"))
    monkeypatch.setattr(marketstack.requests, "get", request)
    with pytest.raises(VendorNotConfiguredError):
        marketstack.get_instrument_info("MSFT")
    request.assert_not_called()


def test_marketstack_timeout_retries_are_bounded_and_sanitized(monkeypatch, caplog):
    from unittest.mock import Mock

    import requests

    from tradingagents.dataflows import marketstack

    monkeypatch.setenv("MARKETSTACK_API_KEY", "secret-key")
    request = Mock(side_effect=requests.Timeout("https://api.marketstack.com?access_key=secret-key"))
    monkeypatch.setattr(marketstack.requests, "get", request)
    monkeypatch.setattr(marketstack.time, "sleep", Mock())
    with pytest.raises(VendorError) as failure:
        marketstack.get_instrument_info("AAPL")
    assert request.call_count == 3
    assert "secret-key" not in str(failure.value) + caplog.text


@pytest.mark.parametrize("mutation", ["empty", "missing_adjustment", "wrong_symbol", "foreign_exchange", "infinity", "pagination_gap"])
def test_marketstack_rejects_unusable_data(monkeypatch, mutation):
    from types import SimpleNamespace

    from tradingagents.dataflows import marketstack

    row = {"date": "2026-06-10", "symbol": "AAPL", "exchange": "XNAS", "adj_open": 10,
           "adj_high": 11, "adj_low": 9, "adj_close": 10, "volume": 100, "close": 20}
    if mutation == "missing_adjustment":
        row["adj_close"] = None
    elif mutation == "wrong_symbol":
        row["symbol"] = "MSFT"
    elif mutation == "foreign_exchange":
        row["exchange"] = "XLON"
    elif mutation == "infinity":
        row["adj_close"] = float("inf")
    data = [] if mutation in {"empty", "pagination_gap"} else [row]
    body = {"pagination": {"offset": 0, "total": 1 if mutation == "pagination_gap" else len(data)}, "data": data}
    monkeypatch.setenv("MARKETSTACK_API_KEY", "key")
    monkeypatch.setattr(marketstack.requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, headers={}, json=lambda: body,
    ))
    with pytest.raises(VendorError):
        marketstack.get_ohlcv("AAPL", "2026-06-01", "2026-06-10")


def test_marketstack_quota_error_survives_text_tool_routing(monkeypatch):
    from tradingagents.dataflows import market_data

    set_config({"data_vendors": {"core_stock_apis": "marketstack"}})

    def quota(*args):
        raise VendorRateLimitError("Marketstack monthly request quota exhausted")

    monkeypatch.setitem(market_data.OHLCV_ADAPTERS, "marketstack", quota)
    with pytest.raises(VendorRateLimitError, match="quota exhausted"):
        interface.route_to_vendor("get_stock_data", "AAPL", "2026-06-01", "2026-06-10")
