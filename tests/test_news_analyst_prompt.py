"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)`` while the tool takes a
``ticker``, tricking the LLM into hallucinating free-text query calls.
"""
import inspect
from unittest.mock import Mock

import pytest
from requests import Timeout

import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.utils import news_data_tools
from tradingagents.agents.utils.news_data_tools import get_news
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    VendorError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)


@pytest.mark.unit
def test_get_news_takes_ticker_not_query():
    arg_names = set(get_news.args.keys())
    assert "ticker" in arg_names
    assert "query" not in arg_names


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(ticker, start_date, end_date)" in src
    assert "get_news(query" not in src


@pytest.mark.parametrize("tool,args", [
    (get_news, ("AAPL", "2026-09-14", "2026-09-21")),
    (news_data_tools.get_global_news, ("2026-09-21", 7, 5)),
])
@pytest.mark.parametrize("error", [
    VendorNotConfiguredError, VendorRateLimitError, VendorError, Timeout,
])
def test_news_source_errors_degrade_without_exposing_provider_details(monkeypatch, caplog, tool, args, error):
    route = Mock(side_effect=error("https://provider.example?apikey=SECRET"))
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    result = tool.func(*args)
    assert result.startswith("NEWS_DATA_UNAVAILABLE:")
    assert "Continue with other available evidence" in result
    assert "SECRET" not in result + caplog.text
    assert "event=news_data_unavailable" in caplog.text
    route.assert_called_once_with(tool.name, *args)


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError])
def test_news_programming_and_routing_errors_still_surface(monkeypatch, error):
    monkeypatch.setattr(news_data_tools, "route_to_vendor", Mock(side_effect=error("bad config")))
    with pytest.raises(error):
        get_news.func("AAPL", "2026-09-14", "2026-09-21")


def test_news_uses_configured_fallback_before_degrading(monkeypatch):
    from tradingagents.dataflows import interface

    set_config({"data_vendors": {"news_data": "fmp,alpha_vantage"}})
    primary = Mock(side_effect=VendorNotConfiguredError("subscription"))
    fallback = Mock(return_value="Verified news from the configured fallback")
    unused = Mock(side_effect=AssertionError("Unconfigured vendor must not run"))
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_news", {
        "fmp": primary, "alpha_vantage": fallback, "yfinance": unused,
    })
    assert get_news.func("AAPL", "2026-09-14", "2026-09-21") == fallback.return_value
    primary.assert_called_once()
    fallback.assert_called_once()
    unused.assert_not_called()


@pytest.mark.parametrize("language,notice", [("English", "Data availability"), ("Chinese", "数据缺失提示")])
def test_fmp_402_allows_sentiment_news_and_downstream_report_to_complete(
    fmp_http, monkeypatch, tmp_path, language, notice,
):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import END, StateGraph

    from tradingagents.agents.analysts import sentiment_analyst as sentiment
    from tradingagents.agents.utils.agent_states import AgentState
    from tradingagents.agents.utils.agent_utils import create_msg_delete
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.propagation import Propagator
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.reporting import write_report_tree

    set_config({"data_vendors": {"news_data": "fmp"}, "output_language": language})
    fmp_http.respond = lambda endpoint, params: (402, {"Error Message": "subscription required"})
    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", Mock(return_value="Available social evidence"))
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", Mock(return_value="Available community evidence"))
    prompts = []

    class LLM:
        def invoke(self, messages):
            prompts.append("\n".join(message.content for message in messages))
            return AIMessage(content="Analysis using remaining evidence")

        def bind_tools(self, tools):
            def respond(prompt):
                messages = prompt.to_messages()
                if not any(isinstance(message, ToolMessage) for message in messages):
                    return AIMessage(content="", tool_calls=[
                        {"id": "stock-news", "name": "get_news", "args": {
                            "ticker": "AAPL", "start_date": "2026-09-14", "end_date": "2026-09-21",
                        }},
                        {"id": "global-news", "name": "get_global_news", "args": {
                            "curr_date": "2026-09-21", "look_back_days": 7, "limit": 5,
                        }},
                    ])
                assert all(message.content.startswith("NEWS_DATA_UNAVAILABLE:")
                           for message in messages if isinstance(message, ToolMessage))
                return self.invoke(messages)
            return RunnableLambda(respond)

    llm = LLM()
    graph = StateGraph(AgentState)
    graph.add_node("sentiment", sentiment.create_sentiment_analyst(llm))
    graph.add_node("clear_sentiment", create_msg_delete())
    graph.add_node("news", na.create_news_analyst(llm))
    graph.add_node("tools_news", TradingAgentsGraph._create_tool_nodes(None)["news"])

    def downstream(state):
        assert notice in state["sentiment_report"] and notice in state["news_report"]
        assert state["messages"][-1].content == state["news_report"]
        return {"trader_investment_plan": "Completed using available evidence"}

    graph.add_node("downstream", downstream)
    graph.set_entry_point("sentiment")
    graph.add_edge("sentiment", "clear_sentiment")
    graph.add_edge("clear_sentiment", "news")
    graph.add_conditional_edges("news", ConditionalLogic().should_continue_news, {
        "tools_news": "tools_news", "Msg Clear News": "downstream",
    })
    graph.add_edge("tools_news", "news")
    graph.add_edge("downstream", END)
    state = Propagator().create_initial_state("AAPL", "2026-09-21")
    final = graph.compile().invoke(state, {"recursion_limit": 12})
    assert final["trader_investment_plan"] == "Completed using available evidence"
    assert len(fmp_http.calls) == 3  # Sentiment fetch + two news tools; no 402 retries.
    assert len(prompts) == 2
    assert all("NEWS_DATA_UNAVAILABLE" in prompt for prompt in prompts)
    assert "Available social evidence" in prompts[0]
    report = write_report_tree(final, "AAPL", tmp_path).read_text()
    assert report.count(notice) == 2
    assert "Completed using available evidence" in report


def test_successful_news_does_not_add_coverage_warning():
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda

    from tradingagents.graph.propagation import Propagator

    llm = Mock()
    llm.bind_tools.return_value = RunnableLambda(lambda prompt: AIMessage(content="Supported news analysis"))
    state = Propagator().create_initial_state("AAPL", "2026-09-21")
    state["messages"] = [ToolMessage(name="get_news", content="Verified headline", tool_call_id="news")]
    result = na.create_news_analyst(llm)(state)
    assert result["news_report"] == "Supported news analysis"
    assert result["messages"][0].content == result["news_report"]


def test_stock_news_failure_keeps_available_global_news(fmp_http):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import MessagesState, StateGraph

    from tradingagents.graph.propagation import Propagator
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    set_config({"data_vendors": {"news_data": "fmp"}, "output_language": "English"})
    fmp_http.respond = lambda endpoint, params: (
        (402, {"Error Message": "subscription required"}) if "symbols" in params else
        (200, [{"title": "Verified macro headline", "publishedDate": "2026-09-20 10:00:00",
                "url": "https://example.com/news", "text": "Available macro evidence"}])
    )
    calls = AIMessage(content="", tool_calls=[
        {"id": "stock", "name": "get_news", "args": {
            "ticker": "AAPL", "start_date": "2026-09-14", "end_date": "2026-09-21",
        }},
        {"id": "global", "name": "get_global_news", "args": {"curr_date": "2026-09-21"}},
    ])
    graph = StateGraph(MessagesState)
    graph.add_node("tools", TradingAgentsGraph._create_tool_nodes(None)["news"])
    graph.set_entry_point("tools")
    graph.set_finish_point("tools")
    messages = graph.compile().invoke({"messages": [calls]})["messages"][1:]
    assert messages[0].content.startswith("NEWS_DATA_UNAVAILABLE:")
    assert "Verified macro headline" in messages[1].content
    llm = Mock()
    respond = Mock(return_value=AIMessage(content="Analysis of available macro evidence"))
    llm.bind_tools.return_value = RunnableLambda(respond)
    state = Propagator().create_initial_state("AAPL", "2026-09-21")
    state["messages"] = [calls, *messages]
    report = na.create_news_analyst(llm)(state)["news_report"]
    assert "asset-specific news" in report
    assert "global news" not in report.split("\n\n")[0]
    assert "Verified macro headline" in respond.call_args.args[0].to_string()
    assert len(fmp_http.calls) == 2
