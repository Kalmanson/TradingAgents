"""Report parity plus persistence used by the local report workbench."""

import threading
import time
from types import SimpleNamespace

import pytest

from tradingagents.application.runner import (
    AnalysisEvent,
    AnalysisRequest,
    AnalysisResult,
    AnalysisRunner,
    CancellationToken,
)
from tradingagents.application.task_manager import AnalysisTaskManager
from tradingagents.application.task_store import TaskStore
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree


def _state():
    return {
        "market_report": "MKT",
        "news_report": "NEWS",
        "investment_debate_state": {"judge_decision": "RM PLAN"},
        "trader_investment_plan": "TRADE",
        "risk_debate_state": {"judge_decision": "PM DECISION"},
    }


@pytest.mark.unit
def test_write_report_tree_creates_files(tmp_path):
    out = write_report_tree(_state(), "AAPL", tmp_path)
    assert out.name == "complete_report.md"
    assert (tmp_path / "1_analysts" / "market.md").read_text() == "MKT"
    assert (tmp_path / "1_analysts" / "news.md").read_text() == "NEWS"
    assert (tmp_path / "2_research" / "manager.md").read_text() == "RM PLAN"
    assert (tmp_path / "3_trading" / "trader.md").read_text() == "TRADE"
    assert (tmp_path / "5_portfolio" / "decision.md").read_text() == "PM DECISION"
    complete = out.read_text()
    assert "Trading Analysis Report: AAPL" in complete
    assert "MKT" in complete and "PM DECISION" in complete


@pytest.mark.unit
def test_save_reports_explicit_path(tmp_path):
    # Unbound: with an explicit save_path, the method doesn't touch self/config.
    out = TradingAgentsGraph.save_reports(None, _state(), "AAPL", save_path=tmp_path)
    assert (tmp_path / "complete_report.md").exists()
    assert out == tmp_path / "complete_report.md"


@pytest.mark.unit
def test_save_reports_defaults_under_results_dir(tmp_path):
    mock_self = SimpleNamespace(config={"results_dir": str(tmp_path)})
    out = TradingAgentsGraph.save_reports(mock_self, _state(), "AAPL")
    assert out.exists()
    assert out.parent.parent.name == "reports"  # results_dir/reports/AAPL_<stamp>/...
    assert out.parent.name.startswith("AAPL_")


def _request(ticker="AAPL"):
    return AnalysisRequest(
        ticker=ticker,
        analysis_date="2026-01-15",
        analysts=("market", "news", "fundamentals"),
        research_depth=1,
        llm_provider="openai",
        quick_think_llm="quick",
        deep_think_llm="deep",
    )


@pytest.mark.unit
def test_analysis_request_normalizes_crypto_and_builds_isolated_config():
    request = _request("BTCUSD").normalized()
    assert request.ticker == "BTC-USD"
    assert request.asset_type == "crypto"
    assert request.analysts == ("market", "news")

    config = request.build_config()
    config["data_vendors"]["news_data"] = "changed"
    assert request.build_config()["data_vendors"]["news_data"] != "changed"


@pytest.mark.unit
def test_task_store_recovery_resume_and_legacy_dedup(tmp_path):
    store = TaskStore(tmp_path / "ui")
    run_id = store.create_run(_request())
    claimed = store.claim_next()
    assert claimed["id"] == run_id
    assert store.recover_interrupted() == 1
    assert store.get_run(run_id)["status"] == "interrupted"

    resumed = store.clone_run(run_id, resume=True)
    assert store.get_run(resumed)["parent_run_id"] == run_id

    report_dir = tmp_path / "reports" / "AAPL_20260115_120000"
    report_dir.mkdir(parents=True)
    (report_dir / "complete_report.md").write_text("# Trading Analysis Report: AAPL")
    assert store.import_legacy_reports(report_dir.parent) == 1
    assert store.import_legacy_reports(report_dir.parent) == 0


@pytest.mark.unit
def test_task_store_never_persists_environment_api_key(tmp_path, monkeypatch):
    secret = "unit-test-secret-that-must-not-be-written"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    store = TaskStore(tmp_path / "ui")
    store.create_run(_request())
    assert secret.encode() not in store.db_path.read_bytes()


class _FakeMemory:
    def __init__(self):
        self.stored = False

    def get_past_context(self, ticker, as_of):
        return "past"

    def store_decision(self, **kwargs):
        self.stored = True


class _FakePropagator:
    def create_initial_state(self, *args, **kwargs):
        return {"messages": []}

    def get_graph_args(self, callbacks=None):
        return {"stream_mode": "values", "config": {}}


class _FakeCompiledGraph:
    def stream(self, graph_input, **kwargs):
        yield {"messages": [], "market_report": "MARKET"}
        yield {
            "messages": [],
            "market_report": "MARKET",
            "investment_debate_state": {"judge_decision": "RESEARCH"},
            "trader_investment_plan": "TRADE",
            "risk_debate_state": {"judge_decision": "FINAL"},
            "final_trade_decision": "Hold",
        }


class _FakeTradingGraph:
    def __init__(self, *args, **kwargs):
        self.memory_log = _FakeMemory()
        self.propagator = _FakePropagator()
        self.graph = _FakeCompiledGraph()
        self.cleared = False
        self.ended = False
        self.logged = False

    def _resolve_pending_entries(self, ticker):
        return None

    def _memory_as_of(self, analysis_date):
        return None

    def resolve_instrument_context(self, ticker, asset_type):
        return "instrument"

    def begin_checkpoint(self, ticker, analysis_date, asset_type):
        return "thread-id"

    def checkpoint_input(self, initial_state):
        return initial_state

    def _log_state(self, analysis_date, final_state):
        self.logged = True

    def process_signal(self, final_decision):
        return "Hold"

    def clear_checkpoint_on_success(self, ticker, analysis_date, asset_type):
        self.cleared = True

    def end_checkpoint(self):
        self.ended = True


@pytest.mark.unit
def test_analysis_runner_streams_and_finalizes_success(tmp_path):
    graphs = []

    def factory(*args, **kwargs):
        graph = _FakeTradingGraph()
        graphs.append(graph)
        return graph

    runner = AnalysisRunner(_request(), artifact_dir=tmp_path, graph_factory=factory)
    events = list(runner.stream())
    completed = next(event for event in events if event.event_type == "completed")

    assert completed.result.signal == "Hold"
    assert completed.result.report_path == tmp_path / "complete_report.md"
    assert completed.result.report_path.exists()
    assert graphs[0].logged is True
    assert graphs[0].memory_log.stored is True
    assert graphs[0].cleared is True
    assert graphs[0].ended is True


@pytest.mark.unit
def test_analysis_runner_stops_without_clearing_checkpoint():
    graphs = []

    def factory(*args, **kwargs):
        graph = _FakeTradingGraph()
        graphs.append(graph)
        return graph

    token = CancellationToken()
    token.cancel()
    events = list(AnalysisRunner(_request(), graph_factory=factory).stream(token))

    assert events[-1].event_type == "stopped"
    assert graphs[0].cleared is False
    assert graphs[0].ended is True


@pytest.mark.unit
def test_task_manager_runs_requests_serially(tmp_path):
    active = 0
    maximum_active = 0
    execution_order = []
    lock = threading.Lock()

    class FakeRunner:
        def __init__(self, request, *, artifact_dir):
            self.request = request

        def stream(self, cancel_token):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                execution_order.append(self.request.ticker)
            time.sleep(0.03)
            with lock:
                active -= 1
            result = AnalysisResult({}, "Hold", None, {})
            yield AnalysisEvent("completed", result=result)

    store = TaskStore(tmp_path / "ui")
    manager = AnalysisTaskManager(store, runner_factory=FakeRunner)
    try:
        first = manager.submit(_request("AAPL"))
        second = manager.submit(_request("MSFT"))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if all(store.get_run(run_id)["status"] == "completed" for run_id in (first, second)):
                break
            time.sleep(0.02)
        assert [store.get_run(run_id)["status"] for run_id in (first, second)] == [
            "completed",
            "completed",
        ]
        assert execution_order == ["AAPL", "MSFT"]
        assert maximum_active == 1
    finally:
        manager.shutdown()
