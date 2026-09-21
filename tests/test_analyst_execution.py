import unittest

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"])

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].tool_node, "tools_news")
        self.assertEqual(plan.specs[0].clear_node, "Msg Clear News")

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "macro"])

    def test_get_initial_analyst_node_uses_plan_metadata(self):
        plan = build_analyst_execution_plan(["fundamentals", "news"])

        self.assertEqual(
            get_initial_analyst_node(plan),
            "Fundamentals Analyst",
        )

    def test_social_key_displays_as_sentiment_analyst(self):
        # The wire key stays "social" for saved-config back-compat, but the
        # user-visible agent_node label must match the v0.2.5 rename so the
        # wall-time summary and any future consumer of agent_node says
        # "Sentiment Analyst" rather than the legacy "Social Analyst".
        plan = build_analyst_execution_plan(["social"])
        spec = plan.specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")


class AnalystWallTimeTrackerTests(unittest.TestCase):
    def test_records_wall_time_when_analyst_completes(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=10.0)
        tracker.mark_completed("market", completed_at=13.5)

        self.assertEqual(tracker.get_wall_times(), {"market": 3.5})

    def test_formats_summary_in_plan_order(self):
        plan = build_analyst_execution_plan(["news", "market"])
        tracker = AnalystWallTimeTracker(plan)

        tracker.mark_started("market", started_at=20.0)
        tracker.mark_completed("market", completed_at=22.25)
        tracker.mark_started("news", started_at=10.0)
        tracker.mark_completed("news", completed_at=14.0)

        self.assertEqual(
            tracker.format_summary(),
            "Analyst wall time: News 4.00s | Market 2.25s",
        )

    def test_syncs_wall_time_from_sequential_chunks(self):
        plan = build_analyst_execution_plan(["market", "news"])
        tracker = AnalystWallTimeTracker(plan)

        sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
        self.assertEqual(tracker.get_wall_times(), {})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done"},
            now=13.0,
        )
        self.assertEqual(tracker.get_wall_times(), {"market": 3.0})

        sync_analyst_tracker_from_chunk(
            tracker,
            {"market_report": "done", "news_report": "done"},
            now=18.0,
        )
        self.assertEqual(
            tracker.get_wall_times(),
            {"market": 3.0, "news": 5.0},
        )


def test_industry_selection_and_tool_schema_exclude_nonstocks():
    from tradingagents.agents.utils.industry_data_tools import build_industry_tools
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.analyst_execution import applicable_analysts

    selected = ("market", "fundamentals", "industry")
    assert applicable_analysts(selected, "stock") == selected
    assert applicable_analysts(selected, "etf") == ("market", "fundamentals")
    assert applicable_analysts(selected, "crypto") == ("market",)
    spec = build_analyst_execution_plan(["industry"]).specs[0]
    assert spec.report_key == "industry_report" and spec.tool_node == "tools_industry"
    for tool in build_industry_tools(DEFAULT_CONFIG):
        assert "state" not in tool.tool_call_schema.model_json_schema()["properties"]


def test_industry_node_collects_evidence_before_model_and_reports_missing_data():
    import json
    from unittest.mock import Mock

    from langchain_core.messages import HumanMessage, ToolMessage

    from tradingagents.agents.analysts.industry_analyst import create_industry_analyst
    from tradingagents.agents.utils.industry_data_tools import build_industry_tools
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.propagation import Propagator

    llm = Mock()
    node = create_industry_analyst(llm, build_industry_tools(DEFAULT_CONFIG))
    state = Propagator().create_initial_state("NVDA", "2026-01-15")
    state["messages"] = [HumanMessage(content="NVDA")]
    first = node(state)
    assert {call["name"] for call in first["messages"][0].tool_calls} == {
        "get_industry_context", "get_industry_indicators",
    }
    assert not llm.bind_tools.called
    state["messages"] += first["messages"] + [
        ToolMessage(name=name, content=json.dumps({"primary_evidence_available": False}), tool_call_id=str(i))
        for i, name in enumerate(("get_industry_context", "get_industry_indicators"))
    ]
    assert node(state)["industry_report"]
    assert not llm.bind_tools.called
    for kind in ("etf", "crypto"):
        state["asset_type"] = kind
        assert node(state)["industry_report"] == ""
    assert not llm.bind_tools.called


def test_industry_real_toolnode_and_agent_loop(monkeypatch, tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    from tradingagents.agents.analysts.industry_analyst import create_industry_analyst
    from tradingagents.agents.utils.agent_states import AgentState
    from tradingagents.agents.utils.industry_data_tools import build_industry_tools
    from tradingagents.dataflows.industry import IndustryResearch
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.propagation import Propagator

    monkeypatch.setattr(IndustryResearch, "get_industry_context", lambda self: {
        "primary_evidence_available": True, "evidence": [{"source": "sec", "url": "https://www.sec.gov/example"}],
    })
    monkeypatch.setattr(IndustryResearch, "get_industry_indicators", lambda self: {"evidence": []})
    calls = []

    class LLM:
        def bind_tools(self, tools):
            def respond(prompt):
                calls.append(prompt.to_string())
                return AIMessage(content="Industry cycle and evidence https://www.sec.gov/example")
            return RunnableLambda(respond)

    tools = build_industry_tools({"data_cache_dir": str(tmp_path)})
    graph = StateGraph(AgentState)
    graph.add_node("industry", create_industry_analyst(LLM(), tools))
    graph.add_node("tools_industry", ToolNode(tools))
    graph.set_entry_point("industry")
    graph.add_conditional_edges("industry", ConditionalLogic().should_continue_industry,
                               {"tools_industry": "tools_industry", "Msg Clear Industry": END})
    graph.add_edge("tools_industry", "industry")
    state = Propagator().create_initial_state("NVDA", "2026-01-15")
    state["fundamentals_report"] = "FINANCIAL_CONTEXT_MARKER"
    result = graph.compile().invoke(state)
    assert result["industry_report"].startswith("Industry cycle")
    assert len(calls) == 1 and "FINANCIAL_CONTEXT_MARKER" in calls[0]
    assert "falsification" in calls[0] and "Anonymous customers remain anonymous" in calls[0]


def test_industry_report_consistency_checks_units_thresholds_and_citations():
    from tradingagents.agents.analysts.industry_analyst import industry_report_issues

    evidence = {"url": "https://www.wsts.org/example.xlsx", "data": [
        {"display_yi_usd": 185.7, "display_usd_billion": 18.57},
    ]}
    assert not industry_report_issues("WSTS 美洲销售 185.7 亿美元 / $18.57 billion。\n"
                                      "[数据](https://www.wsts.org/example.xlsx#table)\n"
                                      "## 证伪条件\n若订单转弱且库存累积，行业扩张判断将弱化。", evidence)
    assert industry_report_issues("WSTS 美洲销售 1857 亿美元。", evidence)
    assert industry_report_issues("WSTS Americas sales $185.7 billion.", evidence)
    assert not industry_report_issues("WSTS 美洲销售约186亿美元。公司收入962亿美元，库存316亿美元。", evidence)
    assert not industry_report_issues("WSTS Americas sales $18.6 billion. Company revenue $96 billion.", evidence)
    assert industry_report_issues("## 证伪条件\n毛利率低于65%时证伪。", evidence)
    assert industry_report_issues("若库存连续两个季度增长则证伪。", evidence)
    assert industry_report_issues("If margin falls below 65%, invalidate the thesis.", evidence)
    assert industry_report_issues("[虚构来源](https://www.wsts.org/missing.xlsx)", evidence)
    assert industry_report_issues("Sources: https://www.wsts.org/missing.xlsx", evidence)
    assert industry_report_issues("Incomplete report", evidence, "length")
    assert industry_report_issues("Incomplete report", evidence, "max_tokens")
    assert not industry_report_issues("## 催化与证伪\n订单转弱将削弱判断。\n"
                                      "## 已披露事实\n去年价格曾低于10美元。", evidence)
    assert not industry_report_issues("若成本高于同行且收入承压，这一判断会弱化。", evidence)
    assert not industry_report_issues("来源：`https://www.wsts.org/example.xlsx`", evidence)
    assert industry_report_issues("若库存连续三个季度增长则证伪。", evidence)


def test_industry_report_revises_once_then_withholds_invalid_narrative():
    import json

    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts.industry_analyst import create_industry_analyst
    from tradingagents.agents.utils.industry_data_tools import build_industry_tools
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.propagation import Propagator

    bad = "## Falsification\nIf margin falls below 65%, invalidate the thesis."

    class LLM(RunnableLambda):
        def bind_tools(self, tools):
            return self

    for revision in ("Industry demand remains uncertain; weakening orders would undermine the expansion thesis.", bad):
        calls = []

        def respond(prompt, calls=calls, revision=revision):
            calls.append(prompt.to_string())
            return AIMessage(content=bad if len(calls) == 1 else revision)

        state = Propagator().create_initial_state("NVDA", "2026-01-15")
        state["messages"] = [ToolMessage(name=name, content=json.dumps(payload), tool_call_id=name)
                             for name, payload in (
                                 ("get_industry_context", {"primary_evidence_available": True}),
                                 ("get_industry_indicators", {"evidence": []}),
                             )]
        result = create_industry_analyst(LLM(respond), build_industry_tools(DEFAULT_CONFIG))(state)
        assert len(calls) == 2 and "Evidence consistency check failed" in calls[1]
        assert "65%" not in result["industry_report"]
        if revision != bad:
            assert result["industry_report"] == revision
        else:
            assert "source evidence only" in result["industry_report"] or "来源资料摘要" in result["industry_report"]
        diagnostics = result["messages"][-1].response_metadata["industry_validation"]
        assert diagnostics["status"] == ("revised" if revision != bad else "evidence_only")
        assert diagnostics["attempts"][0]["issues"] == ["quantitative_condition"]


def test_industry_failed_revision_preserves_sources_and_specific_reasons(monkeypatch, caplog):
    import json

    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables import RunnableLambda

    from tradingagents.agents.analysts import industry_analyst as module
    from tradingagents.agents.utils.industry_data_tools import build_industry_tools
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.propagation import Propagator

    class LLM(RunnableLambda):
        def bind_tools(self, tools):
            return self

    evidence = [
        {"source": "sec", "status": "available", "url": "https://www.sec.gov/annual.htm",
         "published_at": "2026-01-10", "period": "2025-12-31", "retrieved_at": "2026-01-15",
         "data": {"form": "10-K", "excerpts": {"customers": [
             {"locator": "block 17", "text": "Customer A accounts for a significant share. Its name is not disclosed."},
         ]}}},
        {"source": "sec", "url": "https://data.sec.gov/facts.json", "data": [
            {"metric": "Revenues", "unit": "USD", "observations": [
                {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-01-10", "val": 123456},
            ]},
        ]},
    ]
    for language in ("Chinese", "English"):
        monkeypatch.setattr(module, "get_config", lambda language=language: {"output_language": language})
        for mode in ("citation", "exception", "empty", "tool_call", "content_blocks"):
            calls = []

            def respond(prompt, calls=calls, mode=mode):
                calls.append(prompt)
                if len(calls) == 1:
                    return AIMessage(content="REJECTED_DRAFT", response_metadata={"finish_reason": "length"})
                if mode == "exception":
                    raise RuntimeError("api_key=secret-must-not-appear")
                if mode == "empty":
                    return AIMessage(content="")
                if mode == "tool_call":
                    return AIMessage(content="", tool_calls=[{"name": "unexpected", "args": {}, "id": "x"}])
                if mode == "content_blocks":
                    return AIMessage(content=[{"type": "text", "text": "Completed supported narrative."}])
                return AIMessage(content="REJECTED_REVISION [invented](https://invalid.example/fake)")

            state = Propagator().create_initial_state("CRCL", "2026-01-15")
            state["messages"] = [ToolMessage(name=name, content=json.dumps(payload), tool_call_id=name)
                                 for name, payload in (
                                     ("get_industry_context", {"ticker": "CRCL", "primary_evidence_available": True,
                                      "evidence": evidence, "failures": [{"source": "fmp", "dataset": "revenue-product-segmentation",
                                                                          "reason": "subscription required (HTTP 402)"}]}),
                                     ("get_industry_indicators", {"evidence": []}),
                                 )]
            result = module.create_industry_analyst(LLM(respond), build_industry_tools(DEFAULT_CONFIG))(state)
            assert len(calls) == 2
            report = result["industry_report"]
            assert not any(value in report + caplog.text for value in ("REJECTED_DRAFT", "REJECTED_REVISION", "secret-must-not-appear", "invalid.example"))
            if mode == "content_blocks":
                assert report == "Completed supported narrative."
                continue
            for value in ("Customer A", "block 17", "123456", "USD", "2025-12-31", "2026-01-10", "HTTP 402", "https://www.sec.gov/annual.htm", "truncated"):
                assert value in report
            assert "|---|---:|---|---|---|---|\n| Revenues" in report
            diagnostics = result["messages"][-1].response_metadata["industry_validation"]
            assert diagnostics["status"] == "evidence_only"
            expected = {"citation": "unprovided_citation", "exception": "revision_failed", "empty": "empty_response", "tool_call": "unexpected_tool_call"}[mode]
            assert expected in diagnostics["attempts"][1]["issues"] and expected in report


def test_default_five_analyst_graph_runs_industry_before_research_and_risk(monkeypatch, tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.prebuilt import ToolNode

    import tradingagents.graph.setup as setup
    from tradingagents.agents.schemas import PortfolioDecision, ResearchPlan, TraderProposal
    from tradingagents.agents.utils.industry_data_tools import build_industry_tools
    from tradingagents.dataflows.industry import IndustryResearch
    from tradingagents.graph.conditional_logic import ConditionalLogic
    from tradingagents.graph.propagation import Propagator

    marker = "Industry evidence: disclosed catalyst; falsification and evidence gaps."
    prompts = []

    def respond(prompt):
        prompts.append(str(prompt))
        return AIMessage(content=marker)

    class LLM(RunnableLambda):
        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema, **kwargs):
            outputs = {
                ResearchPlan: {"recommendation": "Hold", "rationale": marker, "strategic_actions": "Monitor demand"},
                TraderProposal: {"action": "HOLD", "reasoning": marker},
                PortfolioDecision: {"rating": "Hold", "executive_summary": marker, "investment_thesis": marker},
            }

            def structured(prompt):
                prompts.append(str(prompt))
                return schema(**outputs[schema])

            return RunnableLambda(structured)

    for name, report_key in (("market", "market_report"), ("sentiment", "sentiment_report"),
                             ("news", "news_report"), ("fundamentals", "fundamentals_report")):
        monkeypatch.setattr(setup, f"create_{name}_analyst", lambda llm, key=report_key:
                            lambda state: {key: "Existing analyst context", "messages": [AIMessage(content="done")]})
    monkeypatch.setattr(IndustryResearch, "get_industry_context", lambda self: {
        "primary_evidence_available": True, "evidence": [],
    })
    monkeypatch.setattr(IndustryResearch, "get_industry_indicators", lambda self: {"evidence": []})
    llm = LLM(respond)
    nodes = {key: ToolNode([]) for key in ("market", "social", "news", "fundamentals")}
    nodes["industry"] = ToolNode(build_industry_tools({"data_cache_dir": str(tmp_path)}))
    graph = setup.GraphSetup(llm, llm, nodes, ConditionalLogic()).setup_graph().compile()
    state = Propagator().create_initial_state("NVDA", "2026-01-15")
    chunks = list(graph.stream(state, stream_mode="updates", config={"recursion_limit": 50}))
    order = [node for chunk in chunks for node in chunk]
    assert order.index("Fundamentals Analyst") < order.index("Industry and Supply Chain Analyst") < order.index("Bull Researcher")
    assert order[-1] == "Portfolio Manager"
    assert sum(marker in prompt for prompt in prompts) >= 7  # Two researchers, three risk analysts and two managers.
    assert marker in chunks[-1]["Portfolio Manager"]["final_trade_decision"]
