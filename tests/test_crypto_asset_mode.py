import unittest

from cli.models import AnalystType, AssetType
from cli.utils import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.graph.propagation import Propagator


class CryptoAssetModeTests(unittest.TestCase):
    def test_detects_crypto_pair_symbols(self):
        self.assertEqual(detect_asset_type("BTC-USD"), AssetType.CRYPTO)
        self.assertEqual(detect_asset_type("eth-usd"), AssetType.CRYPTO)

    def test_defaults_non_crypto_symbols_to_stock(self):
        self.assertEqual(detect_asset_type("AAPL"), AssetType.STOCK)
        self.assertEqual(detect_asset_type("SPY"), AssetType.STOCK)

    def test_filters_out_fundamentals_analyst_for_crypto(self):
        analysts = [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]

        self.assertEqual(
            filter_analysts_for_asset_type(analysts, AssetType.CRYPTO),
            [
                AnalystType.MARKET,
                AnalystType.SOCIAL,
                AnalystType.NEWS,
            ],
        )

    def test_keeps_all_analysts_for_stock(self):
        analysts = [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]

        self.assertEqual(
            filter_analysts_for_asset_type(analysts, AssetType.STOCK),
            analysts,
        )

    def test_propagator_includes_asset_type_in_initial_state(self):
        state = Propagator().create_initial_state(
            "BTC-USD", "2026-04-18", asset_type=AssetType.CRYPTO.value
        )

        self.assertEqual(state["asset_type"], AssetType.CRYPTO.value)


if __name__ == "__main__":
    unittest.main()


def test_fund_analyst_binds_only_etf_tool_and_preserves_company_branch():
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    observed = {}

    class LLM:
        def bind_tools(self, tools):
            observed["tools"] = [tool.name for tool in tools]

            def respond(prompt):
                observed["prompt"] = prompt.to_messages()[0].content
                return AIMessage(content="Evidence and limitations")

            return RunnableLambda(respond)

    node = create_fundamentals_analyst(LLM())
    state = Propagator().create_initial_state("SPY", "2026-09-18", asset_type="etf")
    assert node(state)["fundamentals_report"] == "Evidence and limitations"
    assert observed["tools"] == ["get_etf_fundamentals"]
    assert "operating company" in observed["prompt"] and "holdings" in observed["prompt"]
    state["asset_type"] = "stock"
    node(state)
    assert set(observed["tools"]) == {"get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"}
    graph = object.__new__(TradingAgentsGraph)
    assert "get_etf_fundamentals" in graph._create_tool_nodes()["fundamentals"].tools_by_name


def test_etf_researchers_use_fund_thesis_not_company_financials():
    from unittest.mock import Mock

    from langchain_core.messages import AIMessage

    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    state = Propagator().create_initial_state("SPY", "2026-09-18", asset_type="etf")
    for factory in (create_bull_researcher, create_bear_researcher):
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="Fund thesis")
        factory(llm)(state)
        prompt = llm.invoke.call_args.args[0]
        assert "ETF fundamentals report" in prompt and "fund" in prompt
        assert "company's market opportunities" not in prompt
        assert "declining innovation" not in prompt
