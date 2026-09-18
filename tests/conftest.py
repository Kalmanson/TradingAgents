"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path):
    """Reset the global dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Replace the global outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config_module._config["data_cache_dir"] = str(tmp_path)
    from tradingagents.dataflows import market_data
    market_data._instrument_cache.clear()
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "tradingagents.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client


@pytest.fixture()
def fmp_http(monkeypatch):
    """Run the real pinned SDK against a controlled HTTP boundary, never the network."""
    import json
    from types import SimpleNamespace

    import requests

    monkeypatch.setenv("FMP_API_KEY", "FMP-TEST-SECRET")
    state = SimpleNamespace(calls=[], respond=lambda endpoint, params: (200, []))

    def respond(session, url, *, params, headers, timeout):
        assert url.startswith("https://financialmodelingprep.com/stable/")
        assert headers == {"apikey": "FMP-TEST-SECRET"}
        assert "apikey" not in params and timeout == (5, 30)
        endpoint = url.removeprefix("https://financialmodelingprep.com/stable/")
        state.calls.append((endpoint, params))
        status, body = state.respond(endpoint, params)
        response = requests.Response()
        response.status_code, response.url = status, url
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.Session, "get", respond)
    return state
