"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib

import pytest

import tradingagents.default_config as default_config_module


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gpt-5.6"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.6-luna"
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="google",
        TRADINGAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
        TRADINGAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        TRADINGAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        TRADINGAGENTS_MAX_RISK_ROUNDS="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_reasoning_thinking_overrides(monkeypatch):
    """The provider reasoning/thinking knobs are env-configurable (non-interactive runs)."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_OPENAI_REASONING_EFFORT="high",
        TRADINGAGENTS_GOOGLE_THINKING_LEVEL="minimal",
        TRADINGAGENTS_ANTHROPIC_EFFORT="low",
    )
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] == "high"
    assert dc.DEFAULT_CONFIG["google_thinking_level"] == "minimal"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "low"


def test_reasoning_effort_defaults_to_none(monkeypatch):
    """Unset reasoning/thinking knobs stay None so each provider uses its own default."""
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["google_thinking_level"] is None
    assert dc.DEFAULT_CONFIG["anthropic_effort"] is None


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="",
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(monkeypatch, bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    monkeypatch.setenv("TRADINGAGENTS_CHECKPOINT_ENABLED", bad)
    with pytest.raises(ValueError, match="TRADINGAGENTS_CHECKPOINT_ENABLED"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("TRADINGAGENTS_CHECKPOINT_ENABLED", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG


def test_market_data_environment_overrides_do_not_change_news_or_fundamentals(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CORE_STOCK_VENDOR="marketstack",
                          TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR="local",
                          TRADINGAGENTS_INSTRUMENT_VENDOR="marketstack")
    try:
        assert dc.DEFAULT_CONFIG["data_vendors"]["core_stock_apis"] == "marketstack"
        assert dc.DEFAULT_CONFIG["data_vendors"]["technical_indicators"] == "local"
        assert dc.DEFAULT_CONFIG["data_vendors"]["instrument_data"] == "marketstack"
        assert dc.DEFAULT_CONFIG["data_vendors"]["news_data"] == "yfinance"
        assert dc.DEFAULT_CONFIG["data_vendors"]["fundamental_data"] == "yfinance"
    finally:
        _reload_with_env(monkeypatch)


def test_fmp_environment_selectors_and_key_are_independent(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "not-a-config-value")
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CORE_STOCK_VENDOR="fmp",
                          TRADINGAGENTS_INSTRUMENT_VENDOR="fmp",
                          TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR="local",
                          TRADINGAGENTS_FUNDAMENTAL_VENDOR=" fmp ", TRADINGAGENTS_NEWS_VENDOR="fmp")
    try:
        assert all(dc.DEFAULT_CONFIG["data_vendors"][key] == "fmp" for key in
                   ("core_stock_apis", "instrument_data", "fundamental_data", "news_data"))
        assert dc.DEFAULT_CONFIG["data_vendors"]["technical_indicators"] == "local"
        assert "not-a-config-value" not in repr(dc.DEFAULT_CONFIG)
    finally:
        dc = _reload_with_env(monkeypatch)
    assert all(dc.DEFAULT_CONFIG["data_vendors"][key] == "yfinance" for key in
               ("core_stock_apis", "instrument_data", "technical_indicators", "fundamental_data", "news_data"))


@pytest.mark.parametrize("raw,expected", [(None, False), ("", False), ("true", True), ("false", False), ("1", True), ("0", False)])
def test_etf_report_switch_defaults_off_and_parses_bool(monkeypatch, raw, expected):
    overrides = {} if raw is None else {"TRADINGAGENTS_ETF_REPORTS_ENABLED": raw}
    dc = _reload_with_env(monkeypatch, **overrides)
    try:
        assert dc.DEFAULT_CONFIG["etf_reports_enabled"] is expected
        assert dc.DEFAULT_CONFIG["etf_allowlist"] == "SPY,QQQ,VOO,IVV,VTI,DIA,IWM"
    finally:
        _reload_with_env(monkeypatch)


def test_etf_report_switch_rejects_invalid_boolean(monkeypatch):
    try:
        with pytest.raises(ValueError, match="TRADINGAGENTS_ETF_REPORTS_ENABLED"):
            _reload_with_env(monkeypatch, TRADINGAGENTS_ETF_REPORTS_ENABLED="enabled")
    finally:
        _reload_with_env(monkeypatch)
