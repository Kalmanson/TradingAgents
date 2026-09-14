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


# The public storefront is optional; these scenarios reuse this module's report
# and queue fixtures and never contact a payment, email, or model provider.
@pytest.fixture
def commerce(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("nh3")
    import uuid
    from copy import deepcopy
    from datetime import datetime, timezone

    import requests
    from fastapi.testclient import TestClient

    from tradingagents.commerce.config import CommerceSettings
    from tradingagents.commerce.creem import CreemClient
    from tradingagents.commerce.email import EmailClient
    from tradingagents.commerce.service import CommerceService
    from tradingagents.commerce.store import CommerceStore
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.mvp.app import create_app

    settings = CommerceSettings(
        data_dir=tmp_path / "commerce", public_url="http://testserver", product_id="prod_report",
        creem_api_key="merchant-test-key", webhook_secret="webhook-test-secret",
        resend_api_key="resend-test-key", email_from="reports@example.com",
        support_email="support@example.com", sales_enabled=True,
    )

    class Transport:
        def __init__(self):
            self.checkouts = []
            self.messages = {}
            self.requests = []
            self.ambiguous_checkout = False
            self.email_timeout = False
            self.email_fail = False
            self.lookup_checkout = None

        def request(self, method, url, **kwargs):
            self.requests.append((method, url, deepcopy(kwargs)))
            if url.endswith("/v1/products/prod_report"):
                body = {"id": "prod_report", "price": 599, "currency": "USD", "billing_type": "onetime",
                        "tax_mode": "inclusive", "mode": "test"}
            elif url.endswith("/v1/checkouts") and method == "POST":
                self.checkouts.append(kwargs["json"])
                if self.ambiguous_checkout:
                    raise requests.ReadTimeout("upstream-private-payload")
                body = {"id": "ch_" + uuid.uuid4().hex, "mode": "test", "product": "prod_report",
                        "request_id": kwargs["json"]["request_id"], "checkout_url": "https://checkout.creem.io/test-session"}
            elif url.endswith("/v1/checkouts") and method == "GET":
                body = deepcopy(self.lookup_checkout)
            elif url.endswith("/emails"):
                if self.email_fail:
                    return SimpleNamespace(status_code=500, json=lambda: {"error": "private"})
                key = kwargs["headers"]["Idempotency-Key"]
                self.messages.setdefault(key, {"id": "mail_" + uuid.uuid4().hex, "payload": deepcopy(kwargs["json"])})
                if self.email_timeout:
                    self.email_timeout = False
                    raise requests.ReadTimeout("accepted-but-response-lost")
                body = {"id": self.messages[key]["id"]}
            else:
                raise AssertionError(f"Unexpected provider request {method} {url}")
            return SimpleNamespace(status_code=200, json=lambda: body)

    transport = Transport()
    validated = []
    store = CommerceStore(settings)
    service = CommerceService(settings, store, creem=CreemClient(settings, transport),
                              email=EmailClient(settings, transport), stock_validator=validated.append,
                              base_config={**DEFAULT_CONFIG, "llm_provider": "openai", "backend_url": None})
    client = TestClient(create_app(settings, service=service, start_workers=False))

    def purchase(language="en", ticker="RXRX", key=None):
        key = key or uuid.uuid4().hex
        response = client.post("/api/orders", json={"ticker": ticker, "language": language}, headers={"Idempotency-Key": key})
        assert response.status_code == 200, response.text
        return store.get_order(key, by="idempotency_key")

    def event(order):
        return {"id": "evt_" + uuid.uuid4().hex, "eventType": "checkout.completed", "object": {
            "id": order["creem_checkout_id"], "request_id": order["id"], "status": "completed", "mode": "test", "units": 1,
            "product": {"id": "prod_report", "price": 599, "currency": "USD", "billing_type": "onetime", "tax_mode": "inclusive"},
            "order": {"id": "ord_" + order["id"], "transaction": "tran_" + order["id"],
                      "customer": "cust_payer", "product": "prod_report", "amount": 599,
                      "amount_paid": 599, "amount_due": 599, "sub_total": 499, "tax_amount": 100,
                      "currency": "USD", "status": "paid", "type": "onetime",
                      "created_at": datetime.now(timezone.utc).isoformat()},
            "customer": {"id": "cust_payer", "email": "payer@example.com"},
        }}

    def post_event(payload, signature=True):
        import hashlib
        import hmac
        import json

        raw = json.dumps(payload, ensure_ascii=False).encode()
        digest = hmac.new(settings.webhook_secret.encode(), raw, hashlib.sha256).hexdigest()
        return client.post("/api/webhooks/creem", content=raw, headers={
            "Content-Type": "application/json", "creem-signature": digest if signature else "0" * 64,
        })

    def finish(order, *, failed=False):
        import json

        current = store.get_order(order["id"])
        run_id = current["trade_run_id"]
        assert run_id
        if failed:
            store.tasks.finish(run_id, "failed", error_summary="private-model-error")
        else:
            directory = store.tasks.artifact_dir(run_id)
            language = json.loads(current["params_json"])["request"]["output_language"]
            path = directory / "complete_report.md"
            path.write_text(f"# {order['ticker']} report\n\nLanguage: {language}\n\n中文 / 日本語 / Français\n\n## Decision\nHold\n", encoding="utf-8")
            store.tasks.finish(run_id, "completed", report_path=path)
        store.reconcile()
        return store.get_order(order["id"])

    yield SimpleNamespace(settings=settings, store=store, service=service, client=client,
                          transport=transport, validated=validated, purchase=purchase, event=event,
                          post_event=post_event, finish=finish)
    client.close()


def test_commerce_profile_preserves_explicit_retry_limit():
    from tradingagents.commerce.profile import build_profile
    from tradingagents.default_config import DEFAULT_CONFIG

    config = {**DEFAULT_CONFIG, "llm_provider": "openai", "backend_url": None,
              "llm_max_retries": 0, "max_tokens": 4096}
    profile = build_profile("RXRX", "en", config)
    assert profile["config"]["llm_max_retries"] == 0
    assert profile["config"]["max_tokens"] == 4096
    defaults = build_profile("RXRX", "en", {**config, "llm_max_retries": None, "max_tokens": None})
    assert defaults["config"]["llm_max_retries"] == 2
    assert defaults["config"]["max_tokens"] == 8192


@pytest.mark.integration
def test_commerce_readiness_pauses_sales_at_capacity(commerce):
    c = commerce
    for _ in range(c.settings.max_pending_runs):
        order = c.purchase()
        assert c.post_event(c.event(order)).status_code == 200
    assert c.client.get("/health/ready").json() == {"ready": True, "acceptingOrders": False}
    assert "Report purchases opening soon" in c.client.get("/").text
    assert c.client.post("/api/orders", json={"ticker": "RXRX", "language": "en"},
                         headers={"Idempotency-Key": "capacity-request-key-12345"}).status_code == 503


@pytest.mark.integration
def test_paid_language_report_archive_and_offline_recovery(commerce, tmp_path, capsys):
    import json
    from pathlib import Path

    from tradingagents.commerce.store import CommerceStore
    from tradingagents.mvp.cli import main as commerce_cli

    order = commerce.purchase(language="ja")
    assert commerce.transport.checkouts[0]["request_id"] == order["id"]
    assert "customer" not in commerce.transport.checkouts[0]
    unpaid = commerce.client.get(f"/success/{order['status_token']}?signature=forged&order_id=paid")
    assert "Waiting for payment" in unpaid.text
    assert commerce.store.tasks.list_runs() == []
    event = commerce.event(order)
    assert commerce.post_event(event).json()["result"] == "PROCESSED"
    complete = commerce.finish(order)
    assert complete["status"] == "COMPLETED"
    assert "Language: Japanese" in complete["report_markdown"]
    assert complete["report_sha256"]
    archive = Path(complete["report_path"])
    assert archive.read_text() == complete["report_markdown"]
    assert json.loads(archive.with_name("order.json").read_text())["language"] == "ja"
    reopened = CommerceStore(commerce.settings)
    assert reopened.get_order(order["id"])["report_markdown"] == complete["report_markdown"]
    source = Path(commerce.store.tasks.get_run(complete["trade_run_id"])["report_path"])
    source.unlink()
    archive.unlink()
    report = commerce.client.get(f"/report/{order['report_token']}")
    assert report.status_code == 200
    assert 'lang="ja"' in report.text and "Language: Japanese" in report.text
    assert report.headers["cache-control"] == "no-store"
    assert report.headers["referrer-policy"] == "no-referrer"
    output = tmp_path / "restored.md"
    commerce_cli(["--data-dir", str(commerce.settings.data_dir), "export", "ord_" + order["id"], "--output", str(output)])
    assert output.read_text() == complete["report_markdown"]
    capsys.readouterr()
    commerce_cli(["--data-dir", str(commerce.settings.data_dir), "order", order["id"]])
    details = json.loads(capsys.readouterr().out)
    assert details["report_saved"] is True and details["language"] == "ja"
    assert "customer_email" not in details and "report_token" not in details
    commerce_cli(["--data-dir", str(commerce.settings.data_dir), "backup", "--output", str(tmp_path / "backup.sqlite3")])
    assert (tmp_path / "backup.sqlite3").stat().st_size > 0


@pytest.mark.integration
@pytest.mark.parametrize("language,expected", [("en", "English"), ("zh-CN", "Chinese"), ("fr", "French"), ("ar", "Arabic")])
def test_paid_language_reaches_runner_request(commerce, language, expected):
    import json

    order = commerce.purchase(language=language)
    commerce.post_event(commerce.event(order))
    row = commerce.store.tasks.list_runs()[0]
    request = AnalysisRequest.from_dict(json.loads(row["request_json"]))
    assert request.output_language == expected
    runner = commerce.service.runner_factory(request, artifact_dir=commerce.store.tasks.artifact_dir(row["id"]))
    assert runner.config["output_language"] == expected
    assert runner.config["checkpoint_enabled"] is False
    assert order["id"] in runner.config["memory_log_path"]


@pytest.mark.integration
def test_payment_gate_and_public_inputs(commerce):
    import uuid

    for extra in ({"price": 1}, {"customer_email": "attacker@example.com"}, {"backend_url": "http://localhost"}):
        response = commerce.client.post("/api/orders", json={"ticker": "RXRX", **extra}, headers={"Idempotency-Key": uuid.uuid4().hex})
        assert response.status_code == 422
    assert commerce.client.post("/api/orders", json={"ticker": "RXRX", "language": "injected prompt"}, headers={"Idempotency-Key": uuid.uuid4().hex}).status_code == 422
    response = commerce.client.post("/api/orders", json={"ticker": "RXRX"}, headers={"Origin": "https://attacker.example", "Idempotency-Key": uuid.uuid4().hex})
    assert response.status_code == 403
    assert not commerce.transport.checkouts


@pytest.mark.integration
def test_bad_signature_and_duplicate_payment_are_harmless(commerce):
    import uuid

    order = commerce.purchase()
    event = commerce.event(order)
    assert commerce.post_event(event, signature=False).status_code == 401
    assert commerce.store.tasks.list_runs() == []
    assert commerce.post_event(event).status_code == 200
    assert commerce.post_event(event).status_code == 200
    event["id"] = "evt_" + uuid.uuid4().hex
    assert commerce.post_event(event).status_code == 200
    assert len(commerce.store.tasks.list_runs()) == 1
    complete = commerce.finish(order)
    event["id"] = "evt_" + uuid.uuid4().hex
    assert commerce.post_event(event).status_code == 200
    assert commerce.store.get_order(order["id"])["status"] == "COMPLETED"
    assert commerce.store.get_order(order["id"])["report_sha256"] == complete["report_sha256"]
    assert commerce.service.deliver_once()
    assert not commerce.service.deliver_once()
    assert len(commerce.transport.messages) == 1
    message = next(iter(commerce.transport.messages.values()))["payload"]
    assert message["to"] == ["payer@example.com"]
    assert f"/report/{order['report_token']}" in message["text"]
    assert "Language: English" not in message["html"]  # only a link, not the report body


@pytest.mark.integration
@pytest.mark.parametrize("part,field,value", [
    ("order", "amount_paid", 1), ("order", "amount_due", 600),
    ("order", "currency", "EUR"), ("order", "status", "pending"),
    ("order", "customer", "cust_attacker"), ("product", "id", "prod_other"),
    ("product", "tax_mode", "exclusive"), ("checkout", "units", 2),
    ("checkout", "mode", "prod"), ("checkout", "status", "pending"),
])
def test_inconsistent_signed_payment_never_runs(commerce, part, field, value):
    order = commerce.purchase()
    event = commerce.event(order)
    target = event["object"] if part == "checkout" else event["object"][part]
    target[field] = value
    response = commerce.post_event(event)
    assert response.status_code == 200 and response.json()["result"] == "REJECTED"
    assert commerce.store.tasks.list_runs() == []
    assert commerce.store.get_order(order["id"])["status"] == "PENDING_PAYMENT"


@pytest.mark.integration
def test_expanded_checkout_is_resolved_by_server(commerce):
    from copy import deepcopy

    order = commerce.purchase()
    event = commerce.event(order)
    commerce.transport.lookup_checkout = deepcopy(event["object"])
    event["object"]["customer"] = "cust_payer"
    assert commerce.post_event(event).json()["result"] == "PROCESSED"
    assert commerce.store.get_order(order["id"])["customer_email"] == "payer@example.com"


@pytest.mark.integration
def test_concurrent_deliveries_create_one_job(commerce):
    import json
    from concurrent.futures import ThreadPoolExecutor

    order = commerce.purchase()
    payloads = [commerce.event(order) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda event: commerce.service.process_webhook(event, json.dumps(event).encode()), payloads))
    assert results == ["PROCESSED"] * 8
    assert len(commerce.store.tasks.list_runs()) == 1


@pytest.mark.unit
def test_task_creation_participates_in_caller_transaction(tmp_path):
    store = TaskStore(tmp_path / "atomic")
    with pytest.raises(RuntimeError), store.transaction() as connection:
        store.create_run(_request(), connection=connection)
        raise RuntimeError("Simulated order update failure")
    assert store.list_runs() == []


@pytest.mark.integration
def test_payment_transaction_rolls_back_event_and_run(commerce, monkeypatch):
    order = commerce.purchase()
    original = commerce.store.tasks.create_run

    def crash_after_insertion(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Crash between job insertion and order binding")

    monkeypatch.setattr(commerce.store.tasks, "create_run", crash_after_insertion)
    with pytest.raises(RuntimeError):
        commerce.post_event(commerce.event(order))
    assert commerce.store.tasks.list_runs() == []
    assert commerce.store.get_order(order["id"])["status"] == "PENDING_PAYMENT"
    monkeypatch.setattr(commerce.store.tasks, "create_run", original)
    assert commerce.post_event(commerce.event(order)).json()["result"] == "PROCESSED"


@pytest.mark.integration
def test_checkout_timeout_does_not_create_a_second_session(commerce):
    import uuid

    key = uuid.uuid4().hex
    commerce.transport.ambiguous_checkout = True
    for _ in range(2):
        response = commerce.client.post("/api/orders", json={"ticker": "RXRX"}, headers={"Idempotency-Key": key})
        assert response.status_code == 503
        assert "upstream-private-payload" not in response.text
    assert len(commerce.transport.checkouts) == 1
    assert commerce.store.get_order(key, by="idempotency_key")["checkout_status"] == "UNKNOWN"


@pytest.mark.integration
def test_failed_and_interrupted_analyses_notify_without_retry(commerce):
    order = commerce.purchase()
    commerce.post_event(commerce.event(order))
    failed = commerce.finish(order, failed=True)
    assert failed["status"] == "FAILED"
    assert "private-model-error" not in commerce.client.get(f"/api/orders/status/{order['status_token']}").text
    assert commerce.service.deliver_once()
    second = commerce.purchase(ticker="AAPL")
    commerce.post_event(commerce.event(second))
    commerce.store.tasks.claim_next()
    assert commerce.store.tasks.recover_interrupted() == 1
    commerce.store.reconcile()
    assert commerce.store.get_order(second["id"])["status"] == "FAILED"
    assert len(commerce.store.tasks.list_runs()) == 2


@pytest.mark.integration
def test_email_ambiguous_response_retries_same_payload_and_key(commerce):
    order = commerce.purchase()
    commerce.post_event(commerce.event(order))
    commerce.finish(order)
    commerce.transport.email_timeout = True
    assert commerce.service.deliver_once()
    assert len(commerce.transport.messages) == 1
    with commerce.store.tasks.transaction() as connection:
        connection.execute("UPDATE email_deliveries SET next_attempt_at='2000-01-01'")
    assert commerce.service.deliver_once()
    assert len(commerce.transport.messages) == 1
    calls = [call for call in commerce.transport.requests if call[1].endswith("/emails")]
    assert calls[0][2]["headers"]["Idempotency-Key"] == calls[1][2]["headers"]["Idempotency-Key"]
    assert calls[0][2]["json"] == calls[1][2]["json"]
    assert commerce.store.get_order(order["id"])["status"] == "COMPLETED"


@pytest.mark.integration
def test_expired_email_idempotency_window_requires_operator(commerce):
    order = commerce.purchase()
    commerce.post_event(commerce.event(order))
    commerce.finish(order)
    with commerce.store.tasks.transaction() as connection:
        connection.execute("UPDATE email_deliveries SET first_attempt_at='2000-01-01T00:00:00+00:00'")
    assert not commerce.service.deliver_once()
    assert not commerce.transport.messages


@pytest.mark.integration
@pytest.mark.parametrize("before_payment", [True, False])
def test_full_refund_never_reopens_access_but_preserves_report(commerce, before_payment):
    import uuid

    order = commerce.purchase()
    event = commerce.event(order)
    refund = {"id": "evt_" + uuid.uuid4().hex, "eventType": "refund.created", "object": {
        "status": "succeeded", "refund_currency": "USD", "refund_amount": 599,
        "transaction": {"id": "tran_" + order["id"], "order": "ord_" + order["id"], "status": "refunded", "mode": "test"},
    }}
    if not before_payment:
        commerce.post_event(event)
        commerce.finish(order)
    assert commerce.post_event(refund).status_code == 200
    event["id"] = "evt_" + uuid.uuid4().hex
    assert commerce.post_event(event).status_code == 200
    current = commerce.store.get_order(order["id"])
    assert current["status"] == "REFUNDED"
    assert commerce.client.get(f"/report/{order['report_token']}").status_code == 410
    if before_payment:
        assert not commerce.store.tasks.list_runs()
    else:
        assert current["report_markdown"]
    assert not commerce.service.deliver_once()


@pytest.mark.integration
def test_random_tokens_and_markdown_cannot_leak_or_execute(commerce):
    import secrets

    order = commerce.purchase()
    for token in (order["id"], "1", secrets.token_urlsafe(32)):
        assert commerce.client.get(f"/report/{token}").status_code == 404
        assert commerce.client.get(f"/api/orders/status/{token}").status_code == 404
    commerce.post_event(commerce.event(order))
    complete = commerce.finish(order)
    with commerce.store.tasks.transaction() as connection:
        connection.execute("UPDATE trade_orders SET report_markdown=? WHERE id=?", (
            complete["report_markdown"] + '\n<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n[bad](javascript:alert(1))', order["id"],
        ))
    report = commerce.client.get(f"/report/{order['report_token']}")
    assert "<script>alert" not in report.text and "<img src=x" not in report.text
    assert 'href="javascript:' not in report.text
    status = commerce.client.get(f"/api/orders/status/{order['status_token']}").json()
    assert "customer_email" not in status and "id" not in status and "params_json" not in status


@pytest.mark.integration
def test_real_worker_lifecycle_delivers_after_browser_closes(commerce):
    from fastapi.testclient import TestClient

    from tradingagents.mvp.app import create_app

    seen = []

    class Runner:
        def __init__(self, request, *, artifact_dir):
            self.request = request
            self.directory = artifact_dir

        def stream(self, token):
            seen.append(self.request.output_language)
            path = self.directory / "complete_report.md"
            path.write_text("# 报告\n\n研究结果。", encoding="utf-8")
            yield AnalysisEvent("completed", result=AnalysisResult({}, "Hold", path, {}))

    app = create_app(commerce.settings, service=commerce.service, runner_factory=Runner)
    with TestClient(app) as active:
        order = commerce.purchase(language="zh-CN")
        assert commerce.post_event(commerce.event(order)).status_code == 200
        # No success page or browser polling is needed to finish delivery.
        deadline = time.monotonic() + 5
        while not commerce.transport.messages and time.monotonic() < deadline:
            time.sleep(.05)
        assert seen == ["Chinese"]
        assert len(commerce.transport.messages) == 1
        assert commerce.store.get_order(order["id"])["status"] == "COMPLETED"
        assert active.get("/health/ready").json()["ready"]


@pytest.mark.integration
def test_commerce_logs_correlate_delivery_without_sensitive_data(commerce, caplog):
    import json
    import logging

    caplog.set_level(logging.DEBUG, logger="tradingagents.commerce")
    c = commerce
    order = c.purchase()
    event = c.event(order)
    c.post_event(event)
    c.post_event(event)
    run = c.store.tasks.claim_next()
    request = AnalysisRequest.from_dict(json.loads(run["request_json"]))
    c.service.runner_factory(request, artifact_dir=c.store.tasks.artifact_dir(run["id"]))
    c.store.reconcile()
    c.store.reconcile()  # 正常轮询不能重复打印同一次 RUNNING 状态变化。
    complete = c.finish(order)
    c.service.deliver_once()
    c.store.reconcile()
    records = [r.getMessage() for r in caplog.records if r.name.startswith("tradingagents.commerce")]
    output = "\n".join(records)
    assert sum("event=analysis_queued " in text for text in records) == 1
    assert sum("event=order_state_saved " in text for text in records) == 2
    for name in ("order_created", "checkout_ready", "payment_recorded", "analysis_starting",
                 "email_sending", "email_delivery_saved"):
        assert any(f"event={name} " in text and f"order_id={order['id']}" in text for text in records)
    assert f"run_id={run['id']}" in output
    assert "status=COMPLETED" in output and "status=SENT" in output
    for sensitive in (order["status_token"], order["report_token"], order["idempotency_key"],
                      order["checkout_url"], "payer@example.com", complete["report_markdown"],
                      c.settings.webhook_secret, c.settings.creem_api_key, c.settings.resend_api_key):
        assert sensitive not in output


@pytest.mark.integration
def test_commerce_failure_logs_are_private_and_follow_commit(commerce, caplog, monkeypatch):
    import logging

    import tradingagents.commerce.store as store_module

    caplog.set_level(logging.DEBUG, logger="tradingagents.commerce")
    caplog.set_level(logging.DEBUG, logger="tradingagents.mvp")
    c = commerce
    order = c.purchase()
    event = c.event(order)
    original = c.store.tasks.create_run
    private = f"private-response payer@example.com {order['report_token']}"

    def fail_transaction(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError(private)

    monkeypatch.setattr(c.store.tasks, "create_run", fail_transaction)
    with pytest.raises(RuntimeError):
        c.post_event(event)
    assert not any("event=analysis_queued " in r.getMessage() or "event=payment_recorded " in r.getMessage()
                   for r in caplog.records)
    assert not c.store.tasks.list_runs()

    monkeypatch.setattr(c.store.tasks, "create_run", original)
    c.post_event(event)

    def fail_archive(*args, **kwargs):
        raise OSError(private)

    monkeypatch.setattr(store_module, "atomic_write", fail_archive)
    assert c.finish(order)["status"] == "FAILED"
    c.transport.email_timeout = True
    c.service.deliver_once()
    with c.store.tasks.transaction() as connection:
        connection.execute("""UPDATE email_deliveries
            SET first_attempt_at='2000-01-01T00:00:00+00:00',
                next_attempt_at='2000-01-01T00:00:00+00:00'""")
    c.service.deliver_once()
    event["id"] = private
    assert c.post_event(event, signature=False).status_code == 401
    output = "\n".join(r.getMessage() for r in caplog.records
                       if r.name.startswith(("tradingagents.commerce", "tradingagents.mvp")))
    for marker in ("status=FAILED", "error_type=OSError", "event=email_send_failed",
                   "event=email_retry_expired", "event=webhook_signature_invalid"):
        assert marker in output
    for sensitive in (private, "payer@example.com", order["report_token"], "accepted-but-response-lost"):
        assert sensitive not in output


@pytest.mark.integration
def test_commerce_cli_configures_visible_logs_without_duplicate_handlers():
    import subprocess
    import sys

    pytest.importorskip("uvicorn")
    # 在独立进程验证真实 logging 配置，避免污染其他测试的日志处理器。
    code = '''
import logging
import logging.config
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from tradingagents.mvp.cli import main

def check_run(*args, **kwargs):
    assert kwargs["access_log"] is False
    assert kwargs["workers"] == 1
    assert kwargs["log_config"]["loggers"]["tradingagents.commerce"]["level"] == "INFO"
    assert "commerce" not in LOGGING_CONFIG["handlers"]
    logging.config.dictConfig(kwargs["log_config"])
    logging.getLogger("tradingagents.commerce.service").info("中文日志 event=smoke order_id=test-order")
    print("ok")

uvicorn.run = check_run
main(["serve"])
main(["serve", "--log-level", "info"])
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok\nok\n"
    assert result.stderr.count("event=smoke") == 2
    assert "INFO tradingagents.commerce.service" in result.stderr
