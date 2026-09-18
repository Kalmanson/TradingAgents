"""Tests for deterministic instrument-identity resolution (#814) and the
context-anchored message placeholder (#888)."""

import unittest
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    create_msg_delete,
    get_instrument_context_from_state,
    resolve_instrument_identity,
)


@pytest.mark.parametrize("symbol,changes,kind,exchange", [
    ("AAPL", {}, "EQUITY", "XNAS"),
    ("MSFT", {}, "EQUITY", "XNAS"),
    ("BRK.B", {"exchange": "NYSE"}, "EQUITY", "XNYS"),
    ("BRK-B", {"exchange": "NYSE"}, "EQUITY", "XNYS"),
    ("SPY", {"isEtf": True, "exchange": "AMEX"}, "ETF", "XASE"),
    ("TSM", {"isAdr": True, "country": "TW", "exchange": "NYSE"}, "EQUITY", "XNYS"),
    ("AAPL", {"isEtf": None}, None, "XNAS"),
    ("AAPL", {"isEtf": "false"}, None, "XNAS"),
    ("AAPL", {"isActivelyTrading": False}, None, "XNAS"),
    ("AAPL", {"exchange": "OTC"}, None, None),
])
def test_fmp_instrument_mapping_is_exact_and_purchase_safe(fmp_http, symbol, changes, kind, exchange):
    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.profile import validate_us_equity
    from tradingagents.dataflows import market_data

    profile = {"symbol": symbol.replace(".", "-"), "companyName": "Company", "exchange": "NASDAQ",
               "isEtf": False, "isFund": False, "isActivelyTrading": True, **changes}
    fmp_http.respond = lambda endpoint, params: (200, [profile])
    config = {"data_vendors": {"instrument_data": "fmp"}}
    info = market_data.get_instrument_info(symbol, config=config)
    assert info["quote_type"] == kind and info["exchange"] == exchange
    assert info["country"] == ("US" if exchange else None)
    if kind == "EQUITY":
        validate_us_equity(symbol, config=config)
    else:
        with pytest.raises((ValueError, ProviderUnavailable)):
            validate_us_equity(symbol, config=config)


def test_fmp_identity_failure_remains_ticker_only_and_does_not_reuse_yahoo(fmp_http, monkeypatch):
    from tradingagents.dataflows import market_data
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.errors import VendorError

    monkeypatch.setitem(market_data.INSTRUMENT_ADAPTERS, "yfinance", lambda symbol: {"company_name": "Yahoo Cached"})
    market_data.get_instrument_info("AAPL", vendor="yfinance")
    set_config({"data_vendors": {"instrument_data": "fmp"}})
    fmp_http.respond = lambda endpoint, params: (200, [{"symbol": "MSFT", "companyName": "Wrong Company"}])
    with pytest.raises(VendorError):
        market_data.get_instrument_info("AAPL")
    assert resolve_instrument_identity("AAPL") == {}


@pytest.mark.unit
class ResolveInstrumentIdentityTests(unittest.TestCase):
    def setUp(self):
        from tradingagents.dataflows.market_data import _instrument_cache
        _instrument_cache.clear()

    def test_resolves_company_metadata_from_yfinance(self):
        with patch("tradingagents.dataflows.yahoo.yf.Ticker") as mock:
            mock.return_value.get_info.return_value = {
                "longName": "TOTO LTD.",
                "shortName": "TOTO",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
                "quoteType": "EQUITY",
            }
            identity = resolve_instrument_identity("totdy")
        mock.assert_called_once_with("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO LTD.")
        self.assertEqual(identity["sector"], "Industrials")
        self.assertEqual(identity["industry"], "Building Products & Equipment")
        self.assertEqual(identity["exchange"], "PNK")

    def test_falls_back_to_short_name(self):
        with patch("tradingagents.dataflows.yahoo.yf.Ticker") as mock:
            mock.return_value.get_info.return_value = {"shortName": "TOTO", "sector": "Industrials"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO")

    def test_skips_placeholder_values(self):
        with patch("tradingagents.dataflows.yahoo.yf.Ticker") as mock:
            mock.return_value.get_info.return_value = {"longName": "  ", "sector": "None", "industry": "n/a"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity, {})

    def test_fails_open_on_exception(self):
        with patch(
            "tradingagents.dataflows.yahoo.yf.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            self.assertEqual(resolve_instrument_identity("TOTDY"), {})

    def test_result_is_cached(self):
        with patch("tradingagents.dataflows.yahoo.yf.Ticker") as mock:
            mock.return_value.get_info.return_value = {"longName": "TOTO LTD."}
            first = resolve_instrument_identity("TOTDY")
            second = resolve_instrument_identity("TOTDY")
        mock.assert_called_once()  # second call served from cache
        self.assertEqual(first, second)


@pytest.mark.unit
class BuildInstrumentContextTests(unittest.TestCase):
    def test_mentions_exact_symbol_without_identity(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)
        self.assertNotIn("Resolved identity", context)

    def test_injects_resolved_identity(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {
                "company_name": "TOTO LTD.",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
            },
        )
        self.assertIn("Company: TOTO LTD.", context)
        self.assertIn("Industrials / Building Products & Equipment", context)
        self.assertIn("Exchange: PNK", context)
        self.assertIn("Do not substitute a different company", context)

    def test_crypto_uses_name_label_and_keeps_hint(self):
        context = build_instrument_context(
            "BTC-USD", "crypto", {"company_name": "Bitcoin USD"}
        )
        self.assertIn("Name: Bitcoin USD", context)
        self.assertIn("crypto asset rather than a company", context)


@pytest.mark.unit
class GetInstrumentContextFromStateTests(unittest.TestCase):
    def test_prefers_precomputed_context(self):
        state = {"company_of_interest": "TOTDY", "instrument_context": "PRECOMPUTED"}
        self.assertEqual(get_instrument_context_from_state(state), "PRECOMPUTED")

    def test_fallback_is_network_free_ticker_only(self):
        # No instrument_context and no yfinance call — must not hit the network.
        with patch("tradingagents.dataflows.yahoo.yf.Ticker") as mock:
            context = get_instrument_context_from_state(
                {"company_of_interest": "NVDA", "asset_type": "stock"}
            )
        mock.assert_not_called()
        self.assertIn("NVDA", context)

    def test_fallback_respects_asset_type(self):
        context = get_instrument_context_from_state(
            {"company_of_interest": "BTC-USD", "asset_type": "crypto"}
        )
        self.assertIn("crypto asset", context)


@pytest.mark.unit
class ContextAnchoredPlaceholderTests(unittest.TestCase):
    """#888 — the message-clear placeholder must not be a bare 'Continue'."""

    def _run(self, state_extra):
        state = {
            "messages": [
                HumanMessage(content="old", id="h1"),
                AIMessage(content="reply", id="a1"),
            ],
            **state_extra,
        }
        return create_msg_delete()(state)

    def test_placeholder_is_not_bare_continue(self):
        result = self._run(
            {"company_of_interest": "EC", "asset_type": "stock", "trade_date": "2026-05-28"}
        )
        placeholder = result["messages"][-1]
        self.assertIsInstance(placeholder, HumanMessage)
        self.assertNotEqual(placeholder.content.strip(), "Continue")

    def test_placeholder_carries_resolved_identity(self):
        result = self._run(
            {
                "company_of_interest": "EC",
                "instrument_context": "The instrument to analyze is `EC`. Resolved identity: Company: Ecopetrol.",
                "trade_date": "2026-05-28",
            }
        )
        content = result["messages"][-1].content
        self.assertIn("Ecopetrol", content)
        self.assertIn("2026-05-28", content)

    def test_old_messages_are_removed(self):
        result = self._run({"company_of_interest": "EC", "trade_date": "2026-05-28"})
        removals = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        humans = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        self.assertEqual(len(removals), 2)
        self.assertEqual(len(humans), 1)

    def test_safe_defaults_when_state_minimal(self):
        result = create_msg_delete()({"messages": [], "company_of_interest": "EC"})
        placeholder = result["messages"][-1]
        self.assertNotEqual(placeholder.content.strip(), "Continue")
        self.assertIn("EC", placeholder.content)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("ticker,kind,exchange,country,outcome", [
    ("AAPL", "equity", "XNAS", "USA", "ok"),
    ("MSFT", "equity", "XNAS", "US", "ok"),
    ("BRK-B", "equity", "XNYS", "USA", "ok"),
    ("BRK.B", "equity", "XNYS", "USA", "ok"),
    ("SPY", "etf", "ARCX", "USA", "unsupported"),
    ("AAPL", "equity", "XLON", "GBR", "unsupported"),
    ("AAPL", None, "XNAS", "USA", "unverifiable"),
    ("AAPL", "equity", None, "USA", "unverifiable"),
])
def test_marketstack_purchase_validation_uses_own_configuration(monkeypatch, ticker, kind, exchange, country, outcome):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.profile import validate_us_equity
    from tradingagents.dataflows import marketstack, yahoo
    from tradingagents.dataflows.config import set_config

    # Global research defaults must not determine storefront eligibility.
    set_config({"data_vendors": {"instrument_data": "yfinance"}})
    config = {"data_vendors": {"instrument_data": "marketstack"}}
    monkeypatch.setenv("MARKETSTACK_API_KEY", "key")
    monkeypatch.setattr(yahoo.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected Yahoo")))
    requested = []

    def respond(url, **kwargs):
        requested.append(url)
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {
            "symbol": ticker.replace("-", "."), "name": "Verified company", "item_type": kind,
            "stock_exchange": {"mic": exchange, "country_code": country},
        })

    monkeypatch.setattr(marketstack.requests, "get", respond)
    if outcome == "ok":
        validate_us_equity(ticker, config=config)
    else:
        with pytest.raises(ProviderUnavailable if outcome == "unverifiable" else ValueError):
            validate_us_equity(ticker, config=config)
    assert requested == [f"https://api.marketstack.com/v2/tickers/{ticker.replace('-', '.')}" ]
    yahoo.yf.Ticker.assert_not_called()


def test_identity_cache_is_provider_specific_and_does_not_hide_purchase_outage(monkeypatch):
    from unittest.mock import Mock

    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.profile import validate_us_equity
    from tradingagents.dataflows import market_data
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.errors import VendorError

    yahoo = Mock(return_value={"provider": "yfinance", "company_name": "Yahoo name"})
    marketstack = Mock(return_value={"provider": "marketstack", "company_name": "Marketstack name",
                                   "quote_type": "EQUITY", "exchange": "XNAS", "country": "US"})
    monkeypatch.setitem(market_data.INSTRUMENT_ADAPTERS, "yfinance", yahoo)
    monkeypatch.setitem(market_data.INSTRUMENT_ADAPTERS, "marketstack", marketstack)
    assert resolve_instrument_identity("AAPL")["company_name"] == "Yahoo name"
    set_config({"data_vendors": {"instrument_data": "marketstack"}})
    assert resolve_instrument_identity("AAPL")["company_name"] == "Marketstack name"
    assert resolve_instrument_identity("AAPL")["company_name"] == "Marketstack name"
    marketstack.assert_called_once()
    marketstack.side_effect = VendorError("Service unavailable")
    with pytest.raises(ProviderUnavailable):
        validate_us_equity("AAPL")  # refresh, even though research has a cached identity
    assert marketstack.call_count == 2
