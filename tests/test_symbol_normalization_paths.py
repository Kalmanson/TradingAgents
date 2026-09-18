"""Symbol normalization must apply on every yfinance path, not just price fetch.

Regression tests for #983 (instrument identity), #984 (reflection returns), and
the news path: a broker symbol like XAUUSD must resolve to the same Yahoo symbol
(GC=F) that the price path uses, so identity, realized-return, and news lookups
hit the right instrument instead of failing/mismatching.
"""
from types import SimpleNamespace

import pandas as pd

import tradingagents.agents.utils.agent_utils as au
import tradingagents.dataflows.yfinance_news as ynews
from tradingagents.dataflows import market_data, yahoo
from tradingagents.dataflows.config import get_config
from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_identity_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def get_info(self):
            return {"longName": "Gold Futures", "quoteType": "FUTURE"}

    monkeypatch.setattr(yahoo.yf, "Ticker", FakeTicker)
    market_data._instrument_cache.clear()

    identity = au.resolve_instrument_identity("XAUUSD")

    assert seen["symbol"] == "GC=F"  # normalized, not the raw broker symbol
    assert identity.get("company_name") == "Gold Futures"


def test_fetch_returns_normalizes_symbol(monkeypatch):
    queried = []

    class FakeTicker:
        def __init__(self, symbol):
            queried.append(symbol)

        def history(self, *args, **kwargs):
            prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
            idx = pd.date_range(start="2025-01-02", periods=len(prices), freq="D")
            return pd.DataFrame({"Close": prices}, index=idx)

    monkeypatch.setattr(yahoo.yf, "Ticker", FakeTicker)

    # _fetch_returns does not use ``self``; call unbound to avoid building the graph.
    raw, alpha, days, resolved = TradingAgentsGraph._fetch_returns(
        SimpleNamespace(config=get_config()), "XAUUSD", "2025-01-02", holding_days=5, benchmark="SPY"
    )

    assert queried[0] == "GC=F"  # stock symbol normalized (#984)
    assert queried[1] == "SPY"   # benchmark left as the canonical symbol
    assert raw is not None and days is not None
    assert resolved == "2025-01-07"  # resolution date recorded (#1251)


def test_news_lookup_normalizes_symbol(monkeypatch):
    seen = {}

    class FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def get_news(self, count):
            return []

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(ynews, "yf_retry", lambda fn: fn())

    out = ynews.get_news_yfinance("XAUUSD", "2025-01-01", "2025-01-10")

    assert seen["symbol"] == "GC=F"   # news queried with the canonical symbol
    assert "XAUUSD" in out            # the user's ticker stays in the report
    assert "GC=F" in out              # provenance noted


def test_marketstack_input_does_not_apply_yahoo_asset_aliases():
    from tradingagents.application.runner import normalize_ticker
    from tradingagents.dataflows import marketstack
    from tradingagents.dataflows.symbol_utils import normalize_input_symbol

    config = {"data_vendors": {"core_stock_apis": "marketstack"}}
    assert normalize_ticker("GOLD", config) == "GOLD"
    assert normalize_input_symbol("XAUUSD", config) == "XAUUSD"
    assert marketstack.normalize_symbol("BRK-B") == marketstack.normalize_symbol("BRK.B")
    assert normalize_input_symbol("GOLD", {"data_vendors": {"core_stock_apis": "yfinance"}}) == "GC=F"
