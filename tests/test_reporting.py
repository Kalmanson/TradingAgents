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
        support_email="support@example.com", recaptcha_enabled=False,
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
            self.product = {"id": "prod_report", "price": 599, "currency": "USD", "billing_type": "onetime",
                            "tax_mode": "inclusive", "mode": "test", "status": "active"}
            self.product_unavailable = False

        def request(self, method, url, **kwargs):
            self.requests.append((method, url, deepcopy(kwargs)))
            if url.endswith("/v1/products") and method == "GET":
                assert kwargs["params"] == {"product_id": "prod_report"}
                if self.product_unavailable:
                    return SimpleNamespace(status_code=503, json=lambda: {})
                body = deepcopy(self.product)
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
        tax = 100
        total = order["amount"] + tax if order["tax_mode"] == "exclusive" else order["amount"]
        return {"id": "evt_" + uuid.uuid4().hex, "eventType": "checkout.completed", "object": {
            "id": order["creem_checkout_id"], "request_id": order["id"], "status": "completed", "mode": "test", "units": 1,
            "product": {"id": order["product_id"], "price": order["amount"], "currency": order["currency"],
                        "billing_type": "onetime", "tax_mode": order["tax_mode"]},
            "order": {"id": "ord_" + order["id"], "transaction": "tran_" + order["id"],
                      "customer": "cust_payer", "product": order["product_id"], "amount": order["amount"],
                      "amount_paid": total, "amount_due": total, "sub_total": total - tax, "tax_amount": tax,
                      "currency": order["currency"], "status": "paid", "type": "onetime",
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
    assert "Reports temporarily unavailable" in c.client.get("/").text
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
    ("order", "sub_total", 1), ("order", "tax_amount", -1), ("checkout", "units", 2),
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
        "transaction": {"id": "tran_" + order["id"], "order": "ord_" + order["id"],
                        "status": "refunded", "mode": "test", "currency": "USD"},
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
    import re

    from tradingagents.commerce.observability import trace_id

    caplog.set_level(logging.DEBUG, logger="tradingagents.commerce")
    caplog.set_level(logging.INFO, logger="tradingagents.mvp")
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
    for name in ("creem_request", "creem_response", "webhook_request", "resend_request", "resend_response"):
        assert f"event={name} " in output
    assert '"http_status":200' in output and '"elapsed_ms":' in output
    assert '"price":599' in output and '"tax_mode":"inclusive"' in output
    assert '"email":"[redacted]"' in output and '"html":"[redacted]"' in output
    browser_records = [r.getMessage() for r in caplog.records if r.name.startswith("tradingagents.mvp")]
    for name in ("http_request", "purchase_request", "purchase_response", "webhook_response", "http_response"):
        assert any(f"event={name} " in record for record in browser_records)
    # The server creates a fresh correlation ID; neither caller headers nor worker context leak to another request.
    request_trace = re.search(r"event=purchase_request trace_id=([a-f0-9]{32})", "\n".join(browser_records)).group(1)
    assert any(f"event=creem_request trace_id={request_trace}" in record for record in records)
    assert any(f"event=http_response trace_id={request_trace}" in record for record in browser_records)
    assert trace_id.get() == "-"
    caplog.clear()
    c.client.get(f"/api/orders/status/{order['status_token']}")
    assert not [r for r in caplog.records if r.name.startswith(("tradingagents.commerce", "tradingagents.mvp"))]


@pytest.mark.integration
@pytest.mark.parametrize("field,value,failed_check", [
    ("id", None, "checkout_id_present"),
    ("request_id", "wrong_order", "request_id_matches"),
    ("mode", "prod", "mode_matches"),
    ("mode", None, "mode_matches"),
    ("product", "prod_other", "product_id_matches"),
    ("checkout_url", "https://other.example/private-checkout?token=secret", "checkout_url_valid"),
    ("checkout_url", ["malformed-private-url"], "checkout_url_valid"),
])
def test_checkout_validation_logs_exact_failure_and_keeps_unknown_order(commerce, caplog, monkeypatch,
                                                                         field, value, failed_check):
    import json
    import logging

    caplog.set_level(logging.INFO, logger="tradingagents.commerce")
    caplog.set_level(logging.INFO, logger="tradingagents.mvp")
    c = commerce
    original = c.transport.request

    def altered_response(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if method == "POST" and url.endswith("/v1/checkouts"):
            body = {**response.json(), field: value}
            return SimpleNamespace(status_code=200, json=lambda: body)
        return response

    monkeypatch.setattr(c.transport, "request", altered_response)
    headers = {"Idempotency-Key": "logging-uncertain-request-12345", "X-Request-ID": "untrusted-trace-value"}
    response = c.client.post("/api/orders", json={"ticker": "INTC", "language": "zh-CN"}, headers=headers)
    assert response.status_code == 503
    record = next(r.getMessage() for r in caplog.records if "event=checkout_validation_failed " in r.getMessage())
    trace = response.headers["X-Request-ID"]
    assert f"trace_id={trace}" in record
    details = json.loads(record.split(" data=", 1)[1])
    assert details["failed_checks"] == [failed_check]
    assert details["expected"]["mode"] == "test"
    if field == "mode":
        assert details["response"]["mode"] == value
    order = c.store.get_order(details["order_id"])
    assert order["checkout_status"] == "UNKNOWN"
    repeat = c.client.post("/api/orders", json={"ticker": "INTC", "language": "zh-CN"}, headers=headers)
    assert repeat.status_code == 503 and len(c.transport.checkouts) == 1
    assert repeat.headers["X-Request-ID"] != trace
    output = "\n".join(r.getMessage() for r in caplog.records)
    assert "reason=Payment service returned an unexpected checkout." in output
    for private in ("untrusted-trace-value", "private-checkout", "malformed-private-url",
                    headers["Idempotency-Key"], order["status_token"], c.settings.creem_api_key):
        assert private not in output


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["timeout", "http_400", "http_503", "non_json", "json_list"])
def test_creem_failure_logs_request_response_and_timing_without_raw_text(commerce, caplog, monkeypatch, kind):
    import logging

    import requests

    from tradingagents.commerce.creem import ProviderUnavailable

    caplog.set_level(logging.INFO, logger="tradingagents.commerce")
    c = commerce
    private = "private-upstream-message buyer@example.com secret-payload"
    status = 400 if kind == "http_400" else 503 if kind == "http_503" else 200

    def request(*args, **kwargs):
        if kind == "timeout":
            raise requests.ReadTimeout(private)

        def body():
            if kind == "non_json":
                raise ValueError(private)
            if kind == "json_list":
                return [private]
            return {"code": "invalid_product", "message": private, "secret": c.settings.creem_api_key}

        return SimpleNamespace(status_code=status, json=body,
                               headers={"x-request-id": "creem-request-123", "content-type": "application/json"})

    monkeypatch.setattr(c.transport, "request", request)
    with pytest.raises(ProviderUnavailable):
        c.service.creem._request("POST", "/v1/checkouts", json={"product_id": "prod_report",
                                "success_url": "https://merchant.example/success/private-access-token"})
    records = [r.getMessage() for r in caplog.records]
    output = "\n".join(records)
    assert "event=creem_request " in output and '"elapsed_ms":' in output
    if kind == "timeout":
        assert "event=creem_request_failed " in output and '"error_type":"ReadTimeout"' in output
    else:
        assert "event=creem_response " in output and f'"http_status":{status}' in output
        assert '"provider_request_id":"creem-request-123"' in output
    if kind.startswith("http_"):
        assert '"code":"invalid_product"' in output
    if kind == "non_json":
        assert '"body_format":"non_json"' in output
    for sensitive in (private, "buyer@example.com", "secret-payload", "private-access-token", c.settings.creem_api_key):
        assert sensitive not in output


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["success", "wrong_host", "rejected", "timeout", "http_500", "non_json"])
def test_recaptcha_logs_protocol_fields_without_token_or_secret(commerce, caplog, monkeypatch, kind):
    import logging
    from dataclasses import replace

    import requests

    from tradingagents.commerce.captcha import CaptchaRejected, CaptchaUnavailable, verify_recaptcha

    caplog.set_level(logging.INFO, logger="tradingagents.commerce")
    settings = replace(commerce.settings, recaptcha_enabled=True, recaptcha_site_key="public-site-key",
                       recaptcha_secret_key="private-captcha-secret")

    def post(*args, **kwargs):
        if kind == "timeout":
            raise requests.ReadTimeout("private-captcha-token private-captcha-secret")

        def body():
            if kind == "non_json":
                raise ValueError("private-captcha-token")
            return {"success": kind != "rejected", "hostname": "other.example" if kind == "wrong_host" else "testserver",
                    "error-codes": ["timeout-or-duplicate"] if kind == "rejected" else [],
                    "unexpected": "private-captcha-token"}

        return SimpleNamespace(status_code=500 if kind == "http_500" else 200, json=body)

    monkeypatch.setattr("tradingagents.commerce.captcha.requests.post", post)
    if kind == "success":
        verify_recaptcha("private-captcha-token", settings)
    else:
        with pytest.raises((CaptchaRejected, CaptchaUnavailable)):
            verify_recaptcha("private-captcha-token", settings)
    output = "\n".join(r.getMessage() for r in caplog.records)
    assert "event=recaptcha_request " in output and '"elapsed_ms":' in output
    if kind != "timeout":
        assert "event=recaptcha_response " in output and '"expected_hostname":"testserver"' in output
    if kind == "wrong_host":
        assert '"hostname":"other.example"' in output
    if kind == "rejected":
        assert "timeout-or-duplicate" in output
    assert "private-captcha-token" not in output and "private-captcha-secret" not in output


def test_interaction_log_redaction_is_bounded_and_single_line(caplog):
    import json
    import logging

    from tradingagents.commerce.observability import log_event

    logger = logging.getLogger("tradingagents.commerce.observability")
    caplog.set_level(logging.INFO, logger=logger.name)
    log_event(logger, "redaction_test", response={
        "id": "unexpected\nevent=forged", "message": "private-free-text", "Authorization": "Bearer private-key",
        "customer": {"id": "cust_123", "email": "payer@example.com", "name": "Private Person"},
        "checkout_url": "https://user:password@checkout.creem.io/private-session?signature=private-signature",
        "items": [{"secret": "private-nested-key", "price": 1900}] * 100,
    })
    message = caplog.records[-1].getMessage()
    assert "\n" not in message and '"id":"cust_123"' in message and '"price":1900' in message
    for secret in ("event=forged", "private-free-text", "private-key", "payer@example.com", "Private Person",
                   "password", "private-session", "private-signature", "private-nested-key"):
        assert secret not in message
    log_event(logger, "redaction_test", response={f"field_{i}": {f"field_{j}": "private" for j in range(50)}
                                                 for i in range(50)})
    message = caplog.records[-1].getMessage()
    assert len(message) < 8200 and json.loads(message.split(" data=", 1)[1])["truncated"]


@pytest.mark.integration
@pytest.mark.parametrize("kind,expected_status", [
    ("equity", 200), ("etf", 422), ("foreign_exchange", 422), ("empty", 503),
    ("timeout", 503), ("rate_limit", 503), ("invalid_response", 503),
])
def test_market_data_logs_eligibility_and_errors_before_order_creation(commerce, caplog, monkeypatch,
                                                                       kind, expected_status):
    import logging

    from curl_cffi.requests.exceptions import HTTPError, Timeout

    from tradingagents.commerce.profile import validate_us_equity

    caplog.set_level(logging.INFO, logger="tradingagents.commerce")
    c = commerce

    def get_info():
        if kind == "timeout":
            raise Timeout("private-yahoo-crumb private-proxy-password", code=28)
        if kind == "rate_limit":
            raise HTTPError("private-yahoo-cookie", response=SimpleNamespace(status_code=429))
        if kind == "invalid_response":
            return ["private-response"]
        if kind == "empty":
            return {}
        return {"symbol": "SPCX", "quoteType": "ETF" if kind == "etf" else "EQUITY",
                "exchange": "LSE" if kind == "foreign_exchange" else "NMS",
                "unexpected": "private-company-profile"}

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: SimpleNamespace(get_info=get_info))
    c.service.stock_validator = validate_us_equity
    response = c.client.post("/api/orders", json={"ticker": "SPCX", "language": "zh-CN"},
                             headers={"Idempotency-Key": "market-data-query-test-12345"})
    assert response.status_code == expected_status
    records = [r.getMessage() for r in caplog.records]
    output = "\n".join(records)
    trace = response.headers["X-Request-ID"]
    assert f"event=market_data_request trace_id={trace}" in output and '"ticker":"SPCX"' in output
    assert '"operation":"get_instrument_info"' in output and '"elapsed_ms":' in output
    if kind in {"timeout", "rate_limit", "invalid_response"}:
        assert f"event=market_data_request_failed trace_id={trace}" in output
        if kind == "timeout":
            assert '"error_code":28' in output
        elif kind == "rate_limit":
            assert '"http_status":429' in output
        assert response.json()["detail"] == "Market data is temporarily unavailable."
    else:
        assert f"event=market_data_response trace_id={trace}" in output
    if kind == "equity":
        assert '"eligible":true' in output and '"quote_type":"EQUITY"' in output and '"exchange":"NMS"' in output
    else:
        assert not c.transport.checkouts
        assert c.store.get_order("market-data-query-test-12345", by="idempotency_key") is None
    for sensitive in ("private-yahoo-crumb", "private-proxy-password", "private-yahoo-cookie",
                      "private-response", "private-company-profile"):
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
import dotenv
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from tradingagents.commerce.config import CommerceSettings
from tradingagents.mvp.cli import main

dotenv.load_dotenv = lambda **kwargs: None
CommerceSettings.from_env = lambda: None

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


@pytest.fixture
def protected_storefront(commerce, monkeypatch):
    from dataclasses import replace
    from unittest.mock import Mock

    from fastapi.testclient import TestClient

    from tradingagents.commerce import captcha
    from tradingagents.mvp.app import create_app

    settings = replace(commerce.settings, recaptcha_enabled=True,
                       recaptcha_site_key="site-key", recaptcha_secret_key="private-captcha-secret")
    commerce.service.settings = settings
    verification = Mock(return_value=SimpleNamespace(
        status_code=200, json=lambda: {"success": True, "hostname": "testserver"},
    ))
    monkeypatch.setattr(captcha.requests, "post", verification)
    with TestClient(create_app(settings, service=commerce.service, start_workers=False)) as client:
        yield SimpleNamespace(commerce=commerce, settings=settings, client=client, verification=verification)


@pytest.mark.integration
@pytest.mark.parametrize("payload,expected", [
    ({}, 403), ({"recaptcha_token": " "}, 403),
    ({"recaptcha_token": "x" * 4097}, 422), ({"recaptcha_token": None}, 422),
    ({"recaptcha_token": "token", "price": 1}, 422),
])
def test_checkout_cannot_bypass_captcha(protected_storefront, payload, expected):
    c = protected_storefront
    response = c.client.post("/api/orders", json={"ticker": "AAPL", **payload},
                             headers={"Idempotency-Key": "a" * 32})
    assert response.status_code == expected
    c.verification.assert_not_called()
    assert not c.commerce.validated and not c.commerce.transport.requests
    assert c.commerce.store.get_order("a" * 32, by="idempotency_key") is None


@pytest.mark.integration
@pytest.mark.parametrize("result,status", [
    ({"success": False, "error-codes": ["invalid-input-response"]}, 403),
    ({"success": False, "error-codes": ["timeout-or-duplicate"]}, 403),
    ({"success": True, "hostname": "attacker.example"}, 403),
    ({"success": True}, 403),
    ({"success": "true", "hostname": "testserver"}, 503),
    ({"success": False, "error-codes": ["invalid-input-secret"]}, 503),
    ({"success": True, "hostname": "testserver", "error-codes": "bad-shape"}, 503),
    ([], 503),
])
def test_captcha_rejections_have_no_checkout_side_effects(protected_storefront, result, status):
    c = protected_storefront
    c.verification.return_value = SimpleNamespace(status_code=200, json=lambda: result)
    response = c.client.post("/api/orders", json={"ticker": "AAPL", "recaptcha_token": "private-token"},
                             headers={"Idempotency-Key": "a" * 32})
    assert response.status_code == status
    assert "private" not in response.text
    assert not c.commerce.validated and not c.commerce.transport.requests
    assert c.commerce.store.get_order("a" * 32, by="idempotency_key") is None


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["timeout", "invalid_json", "http_error"])
def test_captcha_provider_failure_stops_checkout(protected_storefront, failure):
    from unittest.mock import Mock

    import requests

    c = protected_storefront
    if failure == "timeout":
        c.verification.side_effect = requests.ReadTimeout("private upstream details")
    elif failure == "invalid_json":
        c.verification.return_value = SimpleNamespace(status_code=200, json=Mock(side_effect=ValueError("private")))
    else:
        c.verification.return_value.status_code = 503
    response = c.client.post("/api/orders", json={"ticker": "AAPL", "recaptcha_token": "private-token"},
                             headers={"Idempotency-Key": "a" * 32})
    assert response.status_code == 503
    assert "private" not in response.text
    assert c.verification.call_count == 1
    assert not c.commerce.transport.requests
    assert c.commerce.store.get_order("a" * 32, by="idempotency_key") is None


@pytest.mark.integration
def test_verified_checkout_retries_reuse_order_and_webhooks_need_no_captcha(protected_storefront):
    c = protected_storefront
    responses = []
    for token in ("first-private-token", "fresh-private-token"):
        response = c.client.post("/api/orders", json={"ticker": "AAPL", "recaptcha_token": token},
                                 headers={"Idempotency-Key": "a" * 32})
        assert response.status_code == 200, response.text
        responses.append(response.json())
        c.verification.assert_called_with(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": c.settings.recaptcha_secret_key, "response": token},
            timeout=(3.05, 5), allow_redirects=False,
        )
    assert responses[0] == responses[1]
    assert len(c.commerce.transport.checkouts) == 1
    order = c.commerce.store.get_order("a" * 32, by="idempotency_key")
    assert "private-token" not in str(order)
    assert "recaptcha" not in order["params_json"]
    assert "private-captcha-secret" not in repr(c.settings)
    assert not c.commerce.store.tasks.list_runs()
    # Exercise the protected app with the same signed notification as the payment fixture.
    event_response = c.commerce.post_event(c.commerce.event(order))
    webhook = c.client.post("/api/webhooks/creem", content=event_response.request.content,
                            headers=dict(event_response.request.headers))
    assert webhook.status_code == 200
    assert len(c.commerce.store.tasks.list_runs()) == 1
    assert c.verification.call_count == 2


@pytest.mark.integration
def test_captcha_page_and_csp_are_scoped_to_purchase(protected_storefront):
    c = protected_storefront
    page = c.client.get("/")
    assert 'id="purchase-captcha" data-sitekey="site-key"' in page.text
    assert "private-captcha-secret" not in page.text
    assert "https://www.google.com/recaptcha/" in page.headers["Content-Security-Policy"]
    assert "unsafe-inline" not in page.headers["Content-Security-Policy"]
    policy = c.client.get("/privacy")
    assert "Google reCAPTCHA" in policy.text
    assert "google.com/recaptcha/" not in policy.headers["Content-Security-Policy"]
    assert 'id="purchase-captcha"' not in policy.text


@pytest.mark.integration
@pytest.mark.parametrize("missing,variable", [
    ("product_id", "CREEM_TEST_PRODUCT_ID"),
    ("creem_api_key", "CREEM_TEST_API_KEY"),
    ("webhook_secret", "CREEM_TEST_WEBHOOK_SECRET"),
    ("resend_api_key", "RESEND_API_KEY"),
    ("email_from", "COMMERCE_EMAIL_FROM"),
    ("support_email", "COMMERCE_SUPPORT_EMAIL"),
    ("recaptcha_site_key", "RECAPTCHA_SITE_KEY"),
    ("recaptcha_secret_key", "RECAPTCHA_SECRET_KEY"),
])
@pytest.mark.parametrize("value", ["", " "])
def test_missing_captcha_configuration_closes_sales(protected_storefront, missing, variable, value):
    from dataclasses import replace

    from tradingagents.mvp.app import create_app

    c = protected_storefront
    settings = replace(c.settings, data_dir=c.settings.data_dir / "invalid-start", **{missing: value})
    with pytest.raises(ValueError, match=variable):
        create_app(settings)
    assert not settings.data_dir.exists()
    c.verification.assert_not_called()
    assert not c.commerce.transport.requests


@pytest.mark.integration
def test_configured_rate_limits_isolate_ips_and_checkout_from_reads(commerce):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from tradingagents.mvp.app import create_app

    settings = replace(commerce.settings, checkout_rate_limit=2, read_rate_limit=3)
    app = create_app(settings, service=commerce.service, start_workers=False)
    with TestClient(app) as client, TestClient(app, client=("198.51.100.2", 50000)) as other:
        for _ in range(2):
            assert client.post("/api/orders", json={"ticker": "AAPL"}).status_code == 422
        limited = client.post("/api/orders", json={"ticker": "AAPL"})
        assert limited.status_code == 429
        assert limited.headers["Retry-After"] == "60"
        assert limited.headers["Cache-Control"] == "no-store"
        assert limited.headers["X-Frame-Options"] == "DENY"
        assert other.post("/api/orders", json={"ticker": "AAPL"}).status_code == 422
        for path in ("/api/orders/status/invalid", "/success/invalid", "/report/invalid"):
            assert client.get(path).status_code == 404
        assert client.get("/report/invalid").status_code == 429
        assert other.get("/report/invalid").status_code == 404
        assert client.get("/health/live").status_code == 200
        assert client.post("/api/webhooks/creem", content=b"{}").status_code == 401


def test_rate_limit_window_recovers(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("nh3")
    from tradingagents.mvp import app

    limiter = app.RateLimiter()
    monkeypatch.setattr(app.time, "monotonic", lambda: 100)
    assert limiter.allow(("client", "checkout"), 1)
    assert not limiter.allow(("client", "checkout"), 1)
    monkeypatch.setattr(app.time, "monotonic", lambda: 159.9)
    assert not limiter.allow(("client", "checkout"), 1)
    monkeypatch.setattr(app.time, "monotonic", lambda: 160)
    assert limiter.allow(("client", "checkout"), 1)


def test_security_configuration_from_environment(monkeypatch):
    from tradingagents.commerce.config import CommerceSettings

    monkeypatch.setenv("CREEM_MODE", "test")
    monkeypatch.setenv("COMMERCE_PUBLIC_URL", "http://localhost:8000")
    for name, value in {
        "CREEM_TEST_PRODUCT_ID": "prod_test",
        "CREEM_TEST_API_KEY": "test-key",
        "CREEM_TEST_WEBHOOK_SECRET": "test-secret",
        "RESEND_API_KEY": "email-key",
        "COMMERCE_EMAIL_FROM": "reports@example.com",
        "COMMERCE_SUPPORT_EMAIL": "support@example.com",
        "RECAPTCHA_SITE_KEY": "site",
        "RECAPTCHA_SECRET_KEY": "secret",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("COMMERCE_RECAPTCHA_ENABLED", "COMMERCE_CHECKOUT_RATE_LIMIT", "COMMERCE_READ_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    defaults = CommerceSettings.from_env()
    assert defaults.recaptcha_enabled
    assert (defaults.checkout_rate_limit, defaults.read_rate_limit) == (5, 60)
    monkeypatch.setenv("COMMERCE_CHECKOUT_RATE_LIMIT", "2")
    monkeypatch.setenv("COMMERCE_READ_RATE_LIMIT", "30")
    monkeypatch.setenv("RECAPTCHA_SITE_KEY", " site ")
    monkeypatch.setenv("RECAPTCHA_SECRET_KEY", " secret ")
    configured = CommerceSettings.from_env()
    assert (configured.checkout_rate_limit, configured.read_rate_limit) == (2, 30)
    assert (configured.recaptcha_site_key, configured.recaptcha_secret_key) == ("site", "secret")
    for name, value in (("COMMERCE_CHECKOUT_RATE_LIMIT", "0"), ("COMMERCE_READ_RATE_LIMIT", "-1"),
                        ("COMMERCE_CHECKOUT_RATE_LIMIT", "1.5"), ("COMMERCE_RECAPTCHA_ENABLED", "yes")):
        with monkeypatch.context() as context:
            context.setenv(name, value)
            with pytest.raises(ValueError):
                CommerceSettings.from_env()


@pytest.mark.parametrize("overrides", [
    {"recaptcha_enabled": False},
    {"recaptcha_site_key": "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"},
    {"recaptcha_secret_key": "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe"},
])
def test_production_rejects_captcha_bypass_settings(overrides):
    from tradingagents.commerce.config import CommerceSettings

    settings = CommerceSettings(creem_mode="prod", public_url="https://reports.example.com", **overrides)
    with pytest.raises(ValueError, match="Production requires"):
        settings.validate()


@pytest.fixture
def commerce_environment(monkeypatch, tmp_path):
    # Keep configuration tests independent of the developer's .env and providers.
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: None)
    for name, value in {
        "CREEM_MODE": "test",
        "CREEM_TEST_PRODUCT_ID": "prod_test",
        "CREEM_TEST_API_KEY": "private-test-api-key",
        "CREEM_TEST_WEBHOOK_SECRET": "private-test-webhook-secret",
        "CREEM_PROD_PRODUCT_ID": "prod_live",
        "CREEM_PROD_API_KEY": "private-live-api-key",
        "CREEM_PROD_WEBHOOK_SECRET": "private-live-webhook-secret",
        "COMMERCE_DATA_DIR": str(tmp_path / "storefront"),
        "COMMERCE_PUBLIC_URL": "https://reports.example.com",
        "RESEND_API_KEY": "private-email-key",
        "COMMERCE_EMAIL_FROM": "reports@example.com",
        "COMMERCE_SUPPORT_EMAIL": "support@example.com",
        "COMMERCE_RECAPTCHA_ENABLED": "true",
        "RECAPTCHA_SITE_KEY": "site-key",
        "RECAPTCHA_SECRET_KEY": "private-captcha-secret",
    }.items():
        monkeypatch.setenv(name, value)
    return tmp_path / "storefront"


@pytest.mark.parametrize("mode", ["test", "prod"])
def test_creem_mode_uses_only_selected_configuration(commerce_environment, monkeypatch, mode):
    from tradingagents.commerce.config import CommerceSettings
    from tradingagents.commerce.creem import CreemClient

    monkeypatch.setenv("CREEM_MODE", mode)
    other = "PROD" if mode == "test" else "TEST"
    for suffix in ("PRODUCT_ID", "API_KEY", "WEBHOOK_SECRET"):
        monkeypatch.delenv(f"CREEM_{other}_{suffix}")
        monkeypatch.setenv(f"CREEM_{suffix}", "legacy-must-not-be-used")
    settings = CommerceSettings.from_env()
    expected = "test" if mode == "test" else "live"
    assert settings.creem_mode == mode
    assert settings.product_id == f"prod_{expected}"
    assert settings.creem_api_key == f"private-{expected}-api-key"
    assert settings.webhook_secret == f"private-{expected}-webhook-secret"
    assert settings.resend_api_key == "private-email-key"
    assert settings.data_dir == commerce_environment
    assert settings.recaptcha_enabled
    client = CreemClient(settings)
    assert client.base_url == ("https://test-api.creem.io" if mode == "test" else "https://api.creem.io")
    for suffix in ("PRODUCT_ID", "API_KEY", "WEBHOOK_SECRET"):
        with monkeypatch.context() as context:
            variable = f"CREEM_{mode.upper()}_{suffix}"
            context.delenv(variable)
            context.setenv(f"CREEM_{other}_{suffix}", "other-mode-must-not-be-used")
            with pytest.raises(ValueError, match=variable):
                CommerceSettings.from_env()


@pytest.mark.parametrize("environment_mode,cli_mode,expected", [
    (None, None, "test"),
    ("prod", None, "prod"),
    ("prod", "test", "test"),
    ("test", "prod", "prod"),
])
def test_commerce_cli_selects_creem_mode(commerce_environment, monkeypatch, environment_mode, cli_mode, expected):
    from unittest.mock import Mock

    pytest.importorskip("uvicorn")
    from tradingagents.commerce.config import CommerceSettings
    from tradingagents.mvp.cli import main

    if environment_mode is None:
        monkeypatch.delenv("CREEM_MODE")
    else:
        monkeypatch.setenv("CREEM_MODE", environment_mode)
    server = Mock()
    monkeypatch.setattr("uvicorn.run", server)
    main(["serve"] + (["--mode", cli_mode] if cli_mode else []))
    server.assert_called_once()
    assert server.call_args.args == ("tradingagents.mvp.app:create_app",)
    assert server.call_args.kwargs["factory"] is True
    settings = CommerceSettings.from_env()
    assert settings.creem_mode == expected
    assert settings.product_id == ("prod_test" if expected == "test" else "prod_live")
    assert not commerce_environment.exists()


def test_commerce_cli_rejects_missing_configuration_before_start(commerce_environment, monkeypatch, capsys):
    from unittest.mock import Mock

    pytest.importorskip("uvicorn")
    from tradingagents.mvp.cli import main

    missing = ("CREEM_TEST_WEBHOOK_SECRET", "RECAPTCHA_SITE_KEY", "RECAPTCHA_SECRET_KEY")
    for variable in missing:
        monkeypatch.delenv(variable)
    server = Mock()
    monkeypatch.setattr("uvicorn.run", server)
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--mode", "test"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Missing configuration for Creem test mode" in error
    assert all(variable in error for variable in missing)
    assert "private-" not in error
    assert "Traceback" not in error
    server.assert_not_called()
    assert not commerce_environment.exists()


@pytest.mark.integration
@pytest.mark.parametrize("price,currency,tax_mode,label", [
    (1900, "USD", "exclusive", "19.00"),
    (1234, "EUR", "inclusive", "12.34"),
])
def test_creem_price_drives_page_order_payment_and_refund(commerce, price, currency, tax_mode, label):
    import re
    import uuid

    c = commerce
    c.transport.product.update(price=price, currency=currency, tax_mode=tax_mode)
    home = c.client.get("/")
    assert home.status_code == 200
    assert f'Generate report — {currency} {label}' in home.text
    assert ("Tax included" if tax_mode == "inclusive" else "Tax calculated at checkout") in home.text
    assert "$5.99" not in home.text and c.settings.creem_api_key not in home.text
    quote = re.search(r'data-product-quote="([a-f0-9]{64})"', home.text)[1]
    key = uuid.uuid4().hex
    response = c.client.post("/api/orders", json={"ticker": "AAPL", "product_quote": quote},
                             headers={"Idempotency-Key": key})
    assert response.status_code == 200
    order = c.store.get_order(key, by="idempotency_key")
    assert (order["amount"], order["currency"], order["tax_mode"]) == (price, currency, tax_mode)
    assert c.transport.checkouts[0]["product_id"] == order["product_id"]
    assert "price" not in c.transport.checkouts[0]

    # Creem may expand today's product metadata for yesterday's checkout.
    event = c.event(order)
    c.transport.product.update(price=9900, currency="GBP", tax_mode="exclusive")
    event["object"]["product"].update(price=9900, currency="GBP", tax_mode="exclusive")
    assert c.post_event(event).json()["result"] == "PROCESSED"
    assert len(c.store.tasks.list_runs()) == 1
    completed = c.finish(order)
    assert completed["amount"] == price and completed["currency"] == currency
    refund = {"id": "evt_" + uuid.uuid4().hex, "eventType": "refund.created", "object": {
        "status": "succeeded", "refund_currency": currency,
        "transaction": {"id": "tran_" + order["id"], "order": "ord_" + order["id"],
                        "status": "refunded", "mode": "test", "currency": currency},
    }}
    assert c.post_event(refund).json()["result"] == "PROCESSED"
    assert c.store.get_order(order["id"])["status"] == "REFUNDED"


@pytest.mark.integration
def test_price_change_requires_page_refresh_before_checkout(commerce):
    import re

    c = commerce
    page = c.client.get("/")
    old_quote = re.search(r'data-product-quote="([a-f0-9]{64})"', page.text)[1]
    c.transport.product.update(price=1900, tax_mode="exclusive")
    payload = {"ticker": "AAPL", "product_quote": old_quote}
    response = c.client.post("/api/orders", json=payload, headers={"Idempotency-Key": "a" * 32})
    assert response.status_code == 409 and "Reload" in response.json()["detail"]
    assert not c.transport.checkouts and not c.validated
    assert c.store.get_order("a" * 32, by="idempotency_key") is None
    refreshed = c.client.get("/")
    assert "USD 19.00" in refreshed.text
    new_quote = re.search(r'data-product-quote="([a-f0-9]{64})"', refreshed.text)[1]
    assert old_quote != new_quote
    payload["product_quote"] = new_quote
    assert c.client.post("/api/orders", json=payload, headers={"Idempotency-Key": "b" * 32}).status_code == 200


@pytest.mark.integration
def test_product_cache_expires_and_failed_refresh_never_uses_stale_price(commerce, monkeypatch):
    from tradingagents.commerce import creem

    c = commerce
    clock = [100.0]
    monkeypatch.setattr(creem.time, "monotonic", lambda: clock[0])
    assert c.client.get("/").status_code == 200
    c.transport.product["price"] = 1900
    assert "USD 5.99" in c.client.get("/").text
    assert len(c.transport.requests) == 1
    clock[0] = 160.0
    assert "USD 19.00" in c.client.get("/").text
    assert len(c.transport.requests) == 2
    c.transport.product_unavailable = True
    response = c.client.post("/api/orders", json={"ticker": "AAPL"}, headers={"Idempotency-Key": "a" * 32})
    assert response.status_code == 503
    assert c.store.get_order("a" * 32, by="idempotency_key") is None
    home = c.client.get("/")
    assert home.status_code == 503
    assert "Price temporarily unavailable" in home.text and "USD 19.00" not in home.text
    assert not c.transport.checkouts


@pytest.mark.integration
@pytest.mark.parametrize("field,value", [
    ("price", 0), ("price", -1), ("price", True), ("price", "1900"),
    ("currency", None), ("currency", "<USD>"), ("tax_mode", None), ("tax_mode", []),
    ("billing_type", "recurring"), ("mode", "prod"), ("status", "archived"), ("id", "prod_other"),
])
def test_invalid_creem_product_cannot_create_order(commerce, field, value):
    c = commerce
    c.transport.product[field] = value
    response = c.client.post("/api/orders", json={"ticker": "AAPL"}, headers={"Idempotency-Key": "a" * 32})
    assert response.status_code == 503
    assert c.store.get_order("a" * 32, by="idempotency_key") is None
    assert not c.transport.checkouts and not c.validated


@pytest.mark.integration
@pytest.mark.parametrize("overrides", [
    {"amount_paid": 1900}, {"amount_due": 1900}, {"sub_total": 1},
    {"tax_amount": -1}, {"tax_amount": True}, {"tax_amount": "100"}, {"discount_amount": 1},
])
def test_exclusive_tax_payment_still_requires_exact_total(commerce, overrides):
    c = commerce
    c.transport.product.update(price=1900, tax_mode="exclusive")
    order = c.purchase()
    event = c.event(order)
    event["object"]["order"].update(overrides)
    assert c.post_event(event).json()["result"] == "REJECTED"
    assert not c.store.tasks.list_runs()


@pytest.mark.integration
def test_legacy_price_orders_gain_inclusive_tax_mode_without_repricing(commerce):
    from tradingagents.commerce.store import CommerceStore

    c = commerce
    order = c.purchase()
    with c.store.tasks.transaction() as connection:
        connection.execute("ALTER TABLE trade_orders DROP COLUMN tax_mode")
    migrated = CommerceStore(c.settings)
    restored = migrated.get_order(order["id"])
    assert (restored["amount"], restored["currency"], restored["tax_mode"]) == (599, "USD", "inclusive")
    assert restored["creem_checkout_id"] == order["creem_checkout_id"]
    assert CommerceStore(c.settings).get_order(order["id"]) == restored
    c.transport.product.update(price=1900, tax_mode="exclusive")
    assert c.post_event(c.event(restored)).json()["result"] == "PROCESSED"


@pytest.mark.integration
def test_rate_limit_precedes_captcha_network_call(protected_storefront):
    c = protected_storefront
    c.verification.return_value = SimpleNamespace(status_code=200, json=lambda: {"success": False})
    for _ in range(c.settings.checkout_rate_limit):
        assert c.client.post("/api/orders", json={"ticker": "AAPL", "recaptcha_token": "invalid"}).status_code == 403
    assert c.client.post("/api/orders", json={"ticker": "AAPL", "recaptcha_token": "invalid"}).status_code == 429
    assert c.verification.call_count == c.settings.checkout_rate_limit
    assert not c.commerce.transport.requests


@pytest.mark.integration
def test_captcha_request_size_boundaries(protected_storefront):
    c = protected_storefront
    too_large = c.client.post("/api/orders", content=b" " * 8193, headers={"Content-Type": "application/json"})
    assert too_large.status_code == 413
    c.verification.assert_not_called()
    accepted = c.client.post("/api/orders", json={"ticker": "AAPL", "recaptcha_token": "x" * 4096},
                             headers={"Idempotency-Key": "a" * 32})
    assert accepted.status_code == 200


@pytest.mark.integration
@pytest.mark.parametrize("explicit_config", [False, True])
@pytest.mark.parametrize("ticker,kind", [("AAPL", "equity"), ("GOLD", "equity"), ("SPY", "etf"), ("AAPL", None)])
def test_marketstack_checkout_and_paid_queue_keep_order_configuration(commerce, monkeypatch, ticker, kind, explicit_config):
    import json
    from copy import deepcopy
    from unittest.mock import Mock

    from tradingagents.commerce import service as service_module
    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.service import CommerceService
    from tradingagents.dataflows import marketstack, yahoo
    from tradingagents.dataflows.config import set_config

    c = commerce
    for env_var in ("TRADINGAGENTS_CORE_STOCK_VENDOR", "TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR",
                    "TRADINGAGENTS_INSTRUMENT_VENDOR"):
        monkeypatch.delenv(env_var, raising=False)
    config = deepcopy(c.service.base_config)
    config["data_vendors"].update({"core_stock_apis": "marketstack", "technical_indicators": "local",
                                   "instrument_data": "marketstack"})
    if not explicit_config:
        monkeypatch.setenv("TRADINGAGENTS_CORE_STOCK_VENDOR", "marketstack")
        monkeypatch.setenv("TRADINGAGENTS_INSTRUMENT_VENDOR", "marketstack")
    set_config({"data_vendors": {"core_stock_apis": "yfinance", "instrument_data": "yfinance"}})
    service = CommerceService(c.settings, c.store, creem=c.service.creem, email=c.service.email,
                              base_config=config if explicit_config else None)
    monkeypatch.setenv("MARKETSTACK_API_KEY", "purchase-secret-key")
    monkeypatch.setattr(yahoo.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected Yahoo request")))
    monkeypatch.setattr(marketstack.requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, headers={}, json=lambda: {
            "symbol": ticker, "name": "Company", "item_type": kind,
            "stock_exchange": {"mic": "XNAS", "country_code": "USA"},
        },
    ))
    key = "marketstack-checkout-configuration-12345"
    if kind != "equity":
        with pytest.raises(ProviderUnavailable if kind is None else ValueError):
            service.create_checkout(ticker, "en", key)
        assert not c.transport.checkouts
        assert c.store.get_order(key, by="idempotency_key") is None
    else:
        order = service.create_checkout(ticker, "en", key)
        profile = json.loads(order["params_json"])
        assert profile["config"]["data_vendors"]["core_stock_apis"] == "marketstack"
        assert "purchase-secret-key" not in order["params_json"]
        event = c.event(order)
        assert service.process_webhook(event, json.dumps(event).encode()) == "PROCESSED"
        run = c.store.tasks.list_runs()[0]
        assert run["ticker"] == ticker  # GOLD must not become GC=F while enqueuing.
        assert json.loads(run["request_json"])["ticker"] == ticker
        runner = Mock()
        monkeypatch.setattr(service_module, "AnalysisRunner", runner)
        service.runner_factory(AnalysisRequest.from_dict(json.loads(run["request_json"])),
                               artifact_dir=c.settings.data_dir / run["id"] / "artifacts")
        assert runner.call_args.kwargs["config"]["data_vendors"] == profile["config"]["data_vendors"]
    yahoo.yf.Ticker.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize("override", [None, "", "   ", " yfinance ", "marketstack,alpha_vantage"])
def test_storefront_vendor_defaults_and_environment_do_not_mutate_local_config(commerce, monkeypatch, override):
    from copy import deepcopy

    from tradingagents.commerce import service as service_module
    from tradingagents.dataflows.config import get_config
    from tradingagents.mvp.app import create_app

    expected = {"core_stock_apis": "fmp", "technical_indicators": "local",
                "instrument_data": "fmp", "fundamental_data": "fmp", "news_data": "fmp"}
    for env_var in ("TRADINGAGENTS_CORE_STOCK_VENDOR", "TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR",
                    "TRADINGAGENTS_INSTRUMENT_VENDOR", "TRADINGAGENTS_FUNDAMENTAL_VENDOR",
                    "TRADINGAGENTS_NEWS_VENDOR"):
        monkeypatch.delenv(env_var, raising=False)
    if override is not None:
        monkeypatch.setenv("TRADINGAGENTS_CORE_STOCK_VENDOR", override)
        expected["core_stock_apis"] = override.strip() or "fmp"
    original = deepcopy(service_module.DEFAULT_CONFIG)
    local = _request().build_config()
    global_data = get_config()
    app = create_app(commerce.settings, start_workers=False)
    service = app.state.service
    for category, provider in expected.items():
        assert service.base_config["data_vendors"][category] == provider
        assert local["data_vendors"][category] == "yfinance"
    for category in ("macro_data", "prediction_markets"):
        assert service.base_config["data_vendors"][category] == original["data_vendors"][category]
    service.base_config["data_vendors"]["news_data"] = "changed"
    service.base_config["tool_vendors"]["get_stock_data"] = "changed"
    assert original == service_module.DEFAULT_CONFIG
    assert _request().build_config() == local
    assert get_config() == global_data


@pytest.mark.integration
def test_storefront_vendor_environment_overrides_and_explicit_config_take_precedence(commerce, monkeypatch):
    from copy import deepcopy

    from tradingagents.commerce.service import CommerceService

    for env_var in ("TRADINGAGENTS_CORE_STOCK_VENDOR", "TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR",
                    "TRADINGAGENTS_INSTRUMENT_VENDOR", "TRADINGAGENTS_FUNDAMENTAL_VENDOR",
                    "TRADINGAGENTS_NEWS_VENDOR"):
        monkeypatch.setenv(env_var, "yfinance")
    service = CommerceService(commerce.settings, commerce.store)
    for category in ("core_stock_apis", "technical_indicators", "instrument_data", "fundamental_data", "news_data"):
        assert service.base_config["data_vendors"][category] == "yfinance"
    config = deepcopy(service.base_config)
    config["data_vendors"].update({"core_stock_apis": "marketstack", "technical_indicators": "local",
                                   "instrument_data": "marketstack"})
    config["tool_vendors"]["get_stock_data"] = "alpha_vantage"
    explicit = CommerceService(commerce.settings, commerce.store, base_config=config)
    assert explicit.base_config == config
    explicit.base_config["data_vendors"]["core_stock_apis"] = "changed"
    explicit.base_config["tool_vendors"]["get_stock_data"] = "changed"
    assert config["data_vendors"]["core_stock_apis"] == "marketstack"
    assert config["tool_vendors"]["get_stock_data"] == "alpha_vantage"


@pytest.mark.integration
def test_storefront_default_without_marketstack_key_blocks_checkout_without_yahoo(commerce, monkeypatch):
    from unittest.mock import Mock

    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.service import CommerceService
    from tradingagents.dataflows import marketstack, yahoo

    for env_var in ("TRADINGAGENTS_CORE_STOCK_VENDOR", "TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR",
                    "TRADINGAGENTS_INSTRUMENT_VENDOR", "MARKETSTACK_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("TRADINGAGENTS_INSTRUMENT_VENDOR", "marketstack")
    monkeypatch.setattr(yahoo.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected Yahoo request")))
    monkeypatch.setattr(marketstack.requests, "get", Mock(side_effect=AssertionError("Unexpected HTTP request")))
    service = CommerceService(commerce.settings, commerce.store, creem=commerce.service.creem,
                              email=commerce.service.email)
    key = "missing-marketstack-key-123456789"
    with pytest.raises(ProviderUnavailable):
        service.create_checkout("AAPL", "en", key)
    assert commerce.store.get_order(key, by="idempotency_key") is None
    assert not commerce.transport.checkouts
    yahoo.yf.Ticker.assert_not_called()
    marketstack.requests.get.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize("symbol,kind", [("AAPL", "equity"), ("BRK.B", "equity"),
                                        ("SPY", "etf"), ("INVALID", "empty"), ("MSFT", "unknown")])
def test_fmp_checkout_snapshot_and_paid_queue(commerce, monkeypatch, fmp_http, symbol, kind):
    import json
    from unittest.mock import Mock

    from tradingagents.commerce import service as service_module
    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.service import CommerceService
    from tradingagents.dataflows import yahoo
    from tradingagents.dataflows.config import set_config

    for category in ("CORE_STOCK", "TECHNICAL_INDICATORS", "INSTRUMENT", "FUNDAMENTAL", "NEWS"):
        monkeypatch.delenv(f"TRADINGAGENTS_{category}_VENDOR", raising=False)
    set_config({"data_vendors": {"instrument_data": "yfinance"}})
    yahoo_call = Mock(side_effect=AssertionError("Unexpected Yahoo"))
    monkeypatch.setattr(yahoo.yf, "Ticker", yahoo_call)
    fmp_http.respond = lambda endpoint, params: (200, [] if kind == "empty" else [{
        "symbol": params["symbol"], "companyName": "Company", "exchange": "NYSE",
        "isEtf": True if kind == "etf" else (None if kind == "unknown" else False),
        "isFund": False, "isActivelyTrading": True,
    }])
    c = commerce
    service = CommerceService(c.settings, c.store, creem=c.service.creem, email=c.service.email)
    key = "fmp-checkout-configuration-1234567"
    if kind != "equity":
        with pytest.raises((ProviderUnavailable, ValueError)):
            service.create_checkout(symbol, "en", key)
        assert c.store.get_order(key, by="idempotency_key") is None and not c.transport.checkouts
    else:
        order = service.create_checkout(symbol, "en", key)
        profile = json.loads(order["params_json"])
        vendors = profile["config"]["data_vendors"]
        assert all(vendors[key] == "fmp" for key in ("core_stock_apis", "instrument_data", "fundamental_data", "news_data"))
        assert vendors["technical_indicators"] == "local"
        assert "FMP-TEST-SECRET" not in order["params_json"] and "FMP_API_KEY" not in order["params_json"]
        # Changing deployment configuration after checkout must not rewrite the
        # purchased request when payment arrives or when its runner is built.
        service.base_config["data_vendors"] = dict.fromkeys(vendors, "yfinance")
        event = c.event(order)
        assert service.process_webhook(event, json.dumps(event).encode()) == "PROCESSED"
        run = c.store.tasks.list_runs()[0]
        runner = Mock()
        monkeypatch.setattr(service_module, "AnalysisRunner", runner)
        service.runner_factory(AnalysisRequest.from_dict(json.loads(run["request_json"])),
                               artifact_dir=c.settings.data_dir / run["id"] / "artifacts")
        assert runner.call_args.kwargs["config"]["data_vendors"] == vendors
    yahoo_call.assert_not_called()


@pytest.mark.integration
def test_fmp_default_missing_key_blocks_checkout_before_http(commerce, monkeypatch, fmp_http):
    from unittest.mock import Mock

    from tradingagents.commerce.creem import ProviderUnavailable
    from tradingagents.commerce.service import CommerceService
    from tradingagents.dataflows import yahoo

    for name in ("FMP_API_KEY", "TRADINGAGENTS_INSTRUMENT_VENDOR"):
        monkeypatch.delenv(name, raising=False)
    yahoo_call = Mock(side_effect=AssertionError("Unexpected Yahoo"))
    monkeypatch.setattr(yahoo.yf, "Ticker", yahoo_call)
    service = CommerceService(commerce.settings, commerce.store, creem=commerce.service.creem,
                              email=commerce.service.email)
    key = "missing-fmp-key-1234567890123456"
    with pytest.raises(ProviderUnavailable):
        service.create_checkout("AAPL", "en", key)
    assert not fmp_http.calls and not commerce.transport.checkouts
    assert commerce.store.get_order(key, by="idempotency_key") is None
    yahoo_call.assert_not_called()
