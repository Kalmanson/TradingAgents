"""在同一 SQLite 数据库中保存订单、支付回执、报告正文和邮件交付记录。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import tempfile
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from tradingagents.application.runner import AnalysisRequest
from tradingagents.application.task_store import TaskStore
from tradingagents.commerce.config import CommerceSettings
from tradingagents.commerce.creem import Payment

ORDER_STATUSES = ("PENDING_PAYMENT", "PAID", "RUNNING", "COMPLETED", "FAILED", "REFUNDED")
logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        name = None
        # Persist the rename as well as the bytes on supported local filesystems.
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class CommerceStore:
    def __init__(self, settings: CommerceSettings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.tasks = TaskStore(settings.data_dir)
        with closing(self.tasks._connect()) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS trade_orders (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    language TEXT NOT NULL,
                    analysis_date TEXT,
                    params_json TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    tax_mode TEXT NOT NULL DEFAULT 'inclusive' CHECK(tax_mode IN ('inclusive','exclusive')),
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_PAYMENT'
                        CHECK(status IN ('PENDING_PAYMENT','PAID','RUNNING','COMPLETED','FAILED','REFUNDED')),
                    status_token TEXT NOT NULL UNIQUE,
                    report_token TEXT NOT NULL UNIQUE,
                    checkout_status TEXT NOT NULL DEFAULT 'CREATING',
                    creem_checkout_id TEXT UNIQUE,
                    checkout_url TEXT,
                    creem_order_id TEXT UNIQUE,
                    creem_transaction_id TEXT UNIQUE,
                    creem_customer_id TEXT,
                    customer_email TEXT,
                    trade_run_id TEXT UNIQUE REFERENCES runs(id),
                    report_path TEXT,
                    report_markdown TEXT,
                    report_sha256 TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    paid_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    refunded_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_orders_status ON trade_orders(status, created_at);
                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    order_id TEXT REFERENCES trade_orders(id),
                    remote_order_id TEXT,
                    full_refund INTEGER NOT NULL DEFAULT 0,
                    payload_hash TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    error_summary TEXT,
                    received_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_webhook_remote_order ON webhook_events(remote_order_id);
                CREATE TABLE IF NOT EXISTS email_deliveries (
                    id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES trade_orders(id),
                    template_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    first_attempt_at TEXT,
                    next_attempt_at TEXT NOT NULL,
                    payload_json TEXT,
                    provider_message_id TEXT,
                    error_summary TEXT,
                    sent_at TEXT,
                    UNIQUE(order_id, template_type)
                );
            """)
            # 旧版本全部为含税订单；迁移只补税费方式，保留原价格与支付记录。
            columns = {row[1] for row in connection.execute("PRAGMA table_info(trade_orders)")}
            if "tax_mode" not in columns:
                connection.execute("ALTER TABLE trade_orders ADD COLUMN tax_mode TEXT NOT NULL DEFAULT 'inclusive'")
                connection.commit()

    def get_order(self, value: str, *, by: str = "id") -> dict | None:
        if by not in {"id", "status_token", "report_token", "trade_run_id", "idempotency_key"}:
            raise ValueError("Unsupported order lookup")
        with closing(self.tasks._connect()) as connection:
            row = connection.execute(f"SELECT * FROM trade_orders WHERE {by} = ?", (value,)).fetchone()
            return dict(row) if row else None

    def create_order(self, ticker: str, language: str, profile: dict, key: str, product: dict) -> tuple[dict, bool]:
        fingerprint = hashlib.sha256(json.dumps([ticker, language]).encode()).hexdigest()
        with self.tasks.transaction() as connection:
            row = connection.execute("SELECT * FROM trade_orders WHERE idempotency_key = ?", (key,)).fetchone()
            if row:
                if row["request_hash"] != fingerprint:
                    raise ValueError("This checkout request was already used for different inputs.")
                return dict(row), False
            order_id = uuid.uuid4().hex
            connection.execute("""
                INSERT INTO trade_orders
                    (id,idempotency_key,request_hash,ticker,language,params_json,product_id,amount,
                     currency,tax_mode,mode,status_token,report_token,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (order_id, key, fingerprint, ticker, language, json.dumps(profile),
                  product["id"], product["price"], product["currency"], product["tax_mode"],
                  self.settings.creem_mode, secrets.token_urlsafe(32), secrets.token_urlsafe(32), utc_now()))
            return dict(connection.execute("SELECT * FROM trade_orders WHERE id = ?", (order_id,)).fetchone()), True

    def save_checkout(self, order_id: str, checkout: dict) -> None:
        with self.tasks.transaction() as connection:
            row = connection.execute("SELECT creem_checkout_id FROM trade_orders WHERE id = ?", (order_id,)).fetchone()
            if not row or row[0] not in (None, checkout["id"]):
                raise ValueError("Checkout binding conflict")
            connection.execute("""UPDATE trade_orders SET creem_checkout_id = ?, checkout_url = ?,
                checkout_status = 'READY', error_message = NULL WHERE id = ?""",
                               (checkout["id"], checkout["checkout_url"], order_id))

    def checkout_failed(self, order_id: str, uncertain: bool) -> None:
        with self.tasks.transaction() as connection:
            connection.execute("""UPDATE trade_orders SET checkout_status = ?, error_message = ?
                WHERE id = ? AND status = 'PENDING_PAYMENT' AND checkout_status = 'CREATING'""",
                               ("UNKNOWN" if uncertain else "FAILED", "Checkout could not be created.", order_id))

    def event_result(self, event_id: str, payload_hash: str) -> str | None:
        with closing(self.tasks._connect()) as connection:
            row = connection.execute("SELECT payload_hash, processing_status FROM webhook_events WHERE event_id = ?", (event_id,)).fetchone()
            if row and row["payload_hash"] != payload_hash:
                raise ValueError("Event ID reused with a different payload")
            return row["processing_status"] if row else None

    def record_event(self, event: dict, payload_hash: str, result: str, error: str | None = None) -> None:
        with self.tasks.transaction() as connection:
            connection.execute("""INSERT OR IGNORE INTO webhook_events
                (event_id,event_type,payload_hash,processing_status,error_summary,received_at)
                VALUES (?,?,?,?,?,?)""",
                               (event["id"], event["eventType"], payload_hash, result, error, utc_now()))

    def confirm_payment_and_enqueue(self, order_id: str, payment: Payment, event: dict, payload_hash: str) -> None:
        """原子地确认付款并入队，任何一步失败都不能留下孤立任务。"""
        queued_run_id = None
        with self.tasks.transaction() as connection:
            if connection.execute("SELECT 1 FROM webhook_events WHERE event_id = ?", (event["id"],)).fetchone():
                return
            row = connection.execute("SELECT * FROM trade_orders WHERE id = ?", (order_id,)).fetchone()
            if not row:
                raise ValueError("Order disappeared before payment confirmation")
            if row["creem_checkout_id"] not in (None, payment.checkout_id):
                raise ValueError("Checkout is already bound")
            if row["creem_order_id"] not in (None, payment.order_id):
                raise ValueError("Order is already bound to another payment")
            final_status = row["status"]
            if row["status"] == "PENDING_PAYMENT":
                # 退款通知可能先到，先检查它，避免迟到的付款通知重新触发交付。
                refunded = connection.execute("""SELECT 1 FROM webhook_events
                    WHERE remote_order_id = ? AND full_refund = 1 AND processing_status = 'PROCESSED'""",
                                              (payment.order_id,)).fetchone()
                profile = json.loads(row["params_json"])
                analysis_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
                profile["request"]["analysis_date"] = analysis_date
                run_id = None
                if not refunded:
                    request = AnalysisRequest.from_dict(profile["request"])
                    run_id = self.tasks.create_run(request, source="commerce", connection=connection)
                    queued_run_id = run_id
                final_status = "REFUNDED" if refunded else "PAID"
                connection.execute("""UPDATE trade_orders SET status = ?, creem_checkout_id = ?,
                    creem_order_id = ?, creem_transaction_id = ?, creem_customer_id = ?, customer_email = ?,
                    paid_at = ?, trade_run_id = ?, analysis_date = ?, params_json = ?, error_message = NULL,
                    refunded_at = ? WHERE id = ?""",
                                   ("REFUNDED" if refunded else "PAID", payment.checkout_id, payment.order_id,
                                    payment.transaction_id, payment.customer_id, payment.email, payment.paid_at,
                                    run_id, analysis_date, json.dumps(profile), utc_now() if refunded else None, order_id))
            connection.execute("""INSERT INTO webhook_events
                (event_id,event_type,order_id,remote_order_id,payload_hash,processing_status,received_at)
                VALUES (?,?,?,?,?,'PROCESSED',?)""",
                               (event["id"], event["eventType"], order_id, payment.order_id, payload_hash, utc_now()))
        # 退出事务后才表示真正提交成功；回滚时不能打印“已付款/已入队”。
        logger.info("付款处理记录已保存 event=payment_recorded order_id=%s event_id=%s status=%s",
                    order_id, event["id"], final_status)
        if queued_run_id:
            logger.info("分析任务已入队 event=analysis_queued order_id=%s run_id=%s",
                        order_id, queued_run_id)

    def record_refund(self, event: dict, payload_hash: str, remote_order_id: str, full_refund: bool) -> None:
        """记录退款；全额退款撤销公开访问，但不删除已经保存的报告。"""
        with self.tasks.transaction() as connection:
            if connection.execute("SELECT 1 FROM webhook_events WHERE event_id = ?", (event["id"],)).fetchone():
                return
            row = connection.execute("SELECT id,trade_run_id FROM trade_orders WHERE creem_order_id = ?", (remote_order_id,)).fetchone()
            connection.execute("""INSERT INTO webhook_events
                (event_id,event_type,order_id,remote_order_id,full_refund,payload_hash,processing_status,received_at)
                VALUES (?,?,?,?,?,?,'PROCESSED',?)""",
                               (event["id"], event["eventType"], row["id"] if row else None,
                                remote_order_id, int(full_refund), payload_hash, utc_now()))
            if row and full_refund:
                # 排队任务可直接取消；运行中任务只能请求协作停止，不强杀线程。
                connection.execute("UPDATE trade_orders SET status = 'REFUNDED', refunded_at = ? WHERE id = ?", (utc_now(), row["id"]))
                connection.execute("""UPDATE runs SET status = CASE WHEN status = 'queued' THEN 'cancelled'
                    ELSE 'stop_requested' END WHERE id = ? AND status IN ('queued','running')""", (row["trade_run_id"],))
                connection.execute("UPDATE email_deliveries SET status = 'CANCELLED' WHERE order_id = ? AND status = 'PENDING'", (row["id"],))
        logger.info("退款通知已记录 event=refund_recorded order_id=%s event_id=%s full_refund=%s matched_order=%s",
                    row["id"] if row else "none", event["id"], full_refund, bool(row))

    def reconcile(self) -> None:
        """同步任务终态，报告保存成功后才创建邮件，重启时可继续完成交付。"""
        transitions = []
        with self.tasks.transaction() as connection:
            rows = connection.execute("""SELECT o.*,r.status AS run_status,r.report_path AS run_report_path,
                r.started_at AS run_started_at,r.finished_at AS run_finished_at,r.error_summary AS run_error
                FROM trade_orders o JOIN runs r ON r.id = o.trade_run_id
                WHERE o.status IN ('PAID','RUNNING')
                   OR (o.status = 'REFUNDED' AND o.report_markdown IS NULL AND r.status = 'completed')""").fetchall()
            for row in rows:
                if row["run_status"] == "running":
                    connection.execute("UPDATE trade_orders SET status='RUNNING', started_at=? WHERE id=?", (row["run_started_at"], row["id"]))
                    # 只记录首次状态变化，避免每秒扫描同一运行任务时刷屏。
                    if row["status"] != "RUNNING":
                        transitions.append((logging.INFO, row["id"], row["trade_run_id"],
                                            row["status"], "RUNNING", "running", 0, "none"))
                    continue
                if row["run_status"] == "queued":
                    continue
                status = "FAILED"
                report_bytes = 0
                error_type = "analysis_not_completed"
                error = "The analysis could not be completed. Our team will arrange a refund."
                if row["run_status"] == "completed":
                    try:
                        # 只读取该任务自己的报告文件，不能让数据库路径越过任务目录。
                        source = Path(row["run_report_path"] or "").resolve()
                        source.relative_to((self.tasks.runs_dir / row["trade_run_id"]).resolve())
                        if not source.is_file() or not 0 < source.stat().st_size <= 10_000_000:
                            raise ValueError("Invalid report artifact")
                        markdown = source.read_text(encoding="utf-8")
                        if not markdown.strip():
                            raise ValueError("Empty report")
                        destination = self.settings.data_dir / "orders" / row["id"] / "report.md"
                        report_bytes = len(markdown.encode())
                        checksum = hashlib.sha256(markdown.encode()).hexdigest()
                        # 文件归档与数据库正文都保留，公开阅读不依赖原始分析目录。
                        atomic_write(destination, markdown)
                        metadata = {key: row[key] for key in ("id", "ticker", "language", "analysis_date", "trade_run_id", "params_json")}
                        metadata["report_sha256"] = checksum
                        atomic_write(destination.with_name("order.json"), json.dumps(metadata, ensure_ascii=False, indent=2))
                        connection.execute("""UPDATE trade_orders SET report_path=?, report_markdown=?,
                            report_sha256=? WHERE id=?""", (str(destination), markdown, checksum, row["id"]))
                        status, error = "COMPLETED", None
                        error_type = "none"
                    except (OSError, ValueError, UnicodeError) as exc:
                        error_type = type(exc).__name__
                        error = "The report could not be saved. Our team will arrange a refund."
                if row["status"] == "REFUNDED":
                    if status == "COMPLETED":
                        transitions.append((logging.INFO, row["id"], row["trade_run_id"],
                                            "REFUNDED", "REFUNDED", row["run_status"], report_bytes, error_type))
                    continue  # 归档可用于人工核对，但不能重新开放已退款报告。
                connection.execute("""UPDATE trade_orders SET status=?,error_message=?,started_at=?,completed_at=? WHERE id=?""",
                                   (status, error, row["run_started_at"], row["run_finished_at"] or utc_now(), row["id"]))
                kind = "REPORT_READY" if status == "COMPLETED" else "REPORT_FAILED"
                # 订单终态和待发通知一起提交，避免出现“报告完成但永久漏发邮件”。
                connection.execute("""INSERT OR IGNORE INTO email_deliveries
                    (id,order_id,template_type,next_attempt_at) VALUES (?,?,?,?)""",
                                   (uuid.uuid4().hex, row["id"], kind, utc_now()))
                transitions.append((logging.INFO if status == "COMPLETED" else logging.WARNING,
                                    row["id"], row["trade_run_id"], row["status"], status,
                                    row["run_status"], report_bytes, error_type))
        # 整批提交成功后再输出白名单字段，不记录报告正文、路径或底层异常原文。
        for level, order_id, run_id, previous, status, run_status, size, error_type in transitions:
            logger.log(level, "订单执行状态已保存 event=order_state_saved order_id=%s run_id=%s from_status=%s status=%s run_status=%s report_bytes=%s error_type=%s",
                       order_id, run_id, previous, status, run_status, size, error_type)

    def claim_email(self) -> dict | None:
        """领取一条待发通知，并限制模糊发送结果的自动重试窗口。"""
        delivery = None
        with self.tasks.transaction() as connection:
            row = connection.execute("""SELECT e.* FROM email_deliveries e JOIN trade_orders o ON o.id=e.order_id
                WHERE e.status='PENDING' AND e.next_attempt_at<=? AND o.status IN ('COMPLETED','FAILED')
                ORDER BY e.next_attempt_at LIMIT 1""", (utc_now(),)).fetchone()
            if row is None:
                return None
            # 服务商幂等窗口为 24 小时，提前到 23 小时停止自动重试。
            # 已发送但响应丢失的邮件需人工核对，不能越过窗口再次自动投递。
            if row["first_attempt_at"] and datetime.fromisoformat(row["first_attempt_at"]) < datetime.now(timezone.utc) - timedelta(hours=23):
                connection.execute("UPDATE email_deliveries SET status='FAILED',error_summary='Check provider delivery before retrying: idempotency window elapsed.' WHERE id=?", (row["id"],))
            else:
                connection.execute("""UPDATE email_deliveries SET status='SENDING',attempt_count=attempt_count+1,
                    first_attempt_at=COALESCE(first_attempt_at,?) WHERE id=?""", (utc_now(), row["id"]))
                delivery = dict(connection.execute("SELECT * FROM email_deliveries WHERE id=?", (row["id"],)).fetchone())
        if delivery is None:
            logger.warning("邮件重试窗口已过，请核对服务商记录 event=email_retry_expired order_id=%s delivery_id=%s",
                           row["order_id"], row["id"])
        return delivery

    def save_email_payload(self, delivery_id: str, payload: dict) -> None:
        with self.tasks.transaction() as connection:
            connection.execute("UPDATE email_deliveries SET payload_json=COALESCE(payload_json,?) WHERE id=?", (json.dumps(payload), delivery_id))

    def finish_email(self, delivery: dict, message_id: str | None, error: str | None = None) -> None:
        """保存服务商接受结果或下一次重试时间，不改变订单的报告状态。"""
        attempts = delivery["attempt_count"]
        status = "SENT" if message_id else "FAILED" if attempts >= 4 else "PENDING"
        delay = (30, 120, 600, 600)[min(attempts - 1, 3)]
        next_time = datetime.now(timezone.utc) + timedelta(seconds=delay)
        with self.tasks.transaction() as connection:
            connection.execute("""UPDATE email_deliveries SET status=?,provider_message_id=?,error_summary=?,
                next_attempt_at=?,sent_at=? WHERE id=?""",
                               (status, message_id, error, next_time.isoformat(), utc_now() if message_id else None, delivery["id"]))
        # message_id 只用于数据库核对，不把邮件内容或服务商返回体写入日志。
        logger.log(logging.WARNING if status == "FAILED" else logging.INFO,
                   "邮件交付状态已保存 event=email_delivery_saved order_id=%s delivery_id=%s status=%s attempt=%s next_attempt_at=%s",
                   delivery["order_id"], delivery["id"], status, attempts,
                   next_time.isoformat() if status == "PENDING" else "none")

    def recover_deliveries(self) -> None:
        """启动时恢复模糊中间态，必须由持有运行锁的实例调用。"""
        with self.tasks.transaction() as connection:
            emails = connection.execute("UPDATE email_deliveries SET status='PENDING' WHERE status='SENDING'").rowcount
            checkouts = connection.execute("UPDATE trade_orders SET checkout_status='UNKNOWN' WHERE checkout_status='CREATING'").rowcount
        logger.info("启动恢复已完成 event=delivery_recovered pending_emails=%s uncertain_checkouts=%s",
                    emails, checkouts)

    def capacity_available(self) -> bool:
        with closing(self.tasks._connect()) as connection:
            count = connection.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running','stop_requested')").fetchone()[0]
            stalled_before = (datetime.now(timezone.utc) - timedelta(seconds=self.settings.stalled_after_seconds)).isoformat()
            stalled = connection.execute("SELECT 1 FROM runs WHERE status IN ('running','stop_requested') AND started_at < ? LIMIT 1", (stalled_before,)).fetchone()
            return count < self.settings.max_pending_runs and not stalled
