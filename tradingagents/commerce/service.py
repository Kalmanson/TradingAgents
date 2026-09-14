"""付费报告业务编排：验证付款、隔离分析配置，并协调可靠交付。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from copy import deepcopy

from tradingagents.application.runner import AnalysisRunner
from tradingagents.application.task_manager import AnalysisTaskManager
from tradingagents.commerce.config import CommerceSettings
from tradingagents.commerce.creem import (
    CheckoutUncertain,
    CreemClient,
    PaymentRejected,
    ProviderUnavailable,
)
from tradingagents.commerce.email import EmailClient
from tradingagents.commerce.profile import build_profile, validate_inputs, validate_us_equity
from tradingagents.commerce.store import CommerceStore
from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class CommerceService:
    def __init__(self, settings: CommerceSettings, store: CommerceStore, *, creem=None,
                 email=None, stock_validator=validate_us_equity, base_config=None):
        self.settings = settings
        self.store = store
        self.creem = creem or CreemClient(settings)
        self.email = email or EmailClient(settings)
        self.stock_validator = stock_validator
        self.base_config = deepcopy(DEFAULT_CONFIG if base_config is None else base_config)

    def create_checkout(self, ticker: str, language: str, idempotency_key: str) -> dict:
        """先保存订单再创建收银台；相同请求重试时复用原订单。"""
        started = time.monotonic()
        if not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", idempotency_key):
            raise ValueError("A valid Idempotency-Key header is required.")
        ticker = validate_inputs(ticker, language)
        if not self.settings.sales_enabled or self.settings.missing_configuration():
            logger.debug("购买入口未开放 event=checkout_unavailable reason=configuration_or_sales_disabled")
            raise ProviderUnavailable("Report purchases are not available yet.")
        existing = self.store.get_order(idempotency_key, by="idempotency_key")
        if existing:
            if existing["ticker"] != ticker or existing["language"] != language:
                raise ValueError("This checkout request was already used for different inputs.")
            if existing["checkout_url"]:
                logger.info("复用已有收银台 event=checkout_reused order_id=%s", existing["id"])
                return existing
            logger.warning("收银台结果待核对 event=checkout_unconfirmed order_id=%s checkout_status=%s",
                           existing["id"], existing["checkout_status"])
            raise ProviderUnavailable("This checkout could not be confirmed. Please contact support before trying again.")
        if not self.store.capacity_available():
            logger.info("暂缓新购买 event=checkout_unavailable reason=capacity_or_stalled_run")
            raise ProviderUnavailable("We are at capacity. Please return shortly.")
        self.stock_validator(ticker)
        profile = build_profile(ticker, language, self.base_config)
        order, created = self.store.create_order(ticker, language, profile, idempotency_key)
        if not created:
            # 并发请求可能同时通过首次查询，数据库唯一约束决定谁负责创建收银台。
            if order["checkout_url"]:
                return order
            raise ProviderUnavailable("Checkout is being prepared. Please wait before retrying.")
        logger.info("订单已创建 event=order_created order_id=%s ticker=%s language=%s",
                    order["id"], ticker, language)
        try:
            checkout = self.creem.create_checkout(order)
        except CheckoutUncertain:
            # POST 超时不代表远端失败，标记 UNKNOWN 后交由人工核对，不能盲目重发。
            self.store.checkout_failed(order["id"], uncertain=True)
            logger.warning("收银台创建结果不确定 event=checkout_uncertain order_id=%s", order["id"])
            raise
        except (ProviderUnavailable, PaymentRejected) as exc:
            self.store.checkout_failed(order["id"], uncertain=False)
            logger.warning("收银台创建失败 event=checkout_failed order_id=%s error_type=%s",
                           order["id"], type(exc).__name__)
            raise
        self.store.save_checkout(order["id"], checkout)
        # 不输出 checkout_url，它可能携带会话凭证；订单号足以关联后续处理。
        logger.info("收银台已就绪 event=checkout_ready order_id=%s elapsed_ms=%.0f",
                    order["id"], (time.monotonic() - started) * 1000)
        return self.store.get_order(order["id"])

    def process_webhook(self, event: dict, raw: bytes) -> str:
        """只处理入口已验签的通知，不能从浏览器成功跳转直接调用此方法。"""
        # event_id 校验后才允许进入日志，原始报文与客户对象都不直接记录。
        if (not isinstance(event, dict) or not isinstance(event.get("id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", event["id"])
                or not isinstance(event.get("eventType"), str) or len(event["eventType"]) > 80
                or not isinstance(event.get("object"), dict)):
            raise ValueError("Malformed webhook event")
        payload_hash = hashlib.sha256(raw).hexdigest()
        previous = self.store.event_result(event["id"], payload_hash)
        if previous:
            logger.debug("跳过重复支付通知 event=webhook_duplicate event_id=%s result=%s",
                         event["id"], previous)
            return previous
        try:
            if event["eventType"] == "checkout.completed":
                reference = event["object"].get("request_id")
                if not isinstance(reference, str):
                    raise PaymentRejected("Missing merchant order reference.")
                order = self.store.get_order(reference)
                if order is None:
                    # 本地订单可能等待恢复；不写成功回执，让平台后续重试。
                    raise ProviderUnavailable("Merchant order not found. Retry after reconciliation.")
                if order["mode"] != self.settings.creem_mode:
                    raise PaymentRejected("Payment environment mismatch.")
                payment = self.creem.validate_payment(event["object"], order)
                # 付款、事件回执和任务入队共用事务；成功日志由存储层在提交后输出。
                self.store.confirm_payment_and_enqueue(order["id"], payment, event, payload_hash)
            elif event["eventType"] == "refund.created":
                details = self.creem.refund_details(event["object"])
                self.store.record_refund(event, payload_hash, **details)
            else:
                self.store.record_event(event, payload_hash, "IGNORED")
                logger.debug("忽略非交付通知 event=webhook_ignored event_id=%s", event["id"])
                return "IGNORED"
        except ProviderUnavailable as exc:
            logger.warning("支付通知等待重试 event=webhook_retry event_id=%s error_type=%s",
                           event["id"], type(exc).__name__)
            raise
        except (PaymentRejected, sqlite3.IntegrityError) as exc:
            reason = str(exc) if isinstance(exc, PaymentRejected) else "Payment uniqueness constraint failed."
            self.store.record_event(event, payload_hash, "REJECTED", reason)
            # reason 是本项目生成的固定校验原因，不是支付平台的原始响应。
            logger.warning("支付通知未通过校验 event=webhook_rejected event_id=%s reason=%s", event["id"], reason)
            return "REJECTED"
        return "PROCESSED"

    def runner_factory(self, request, *, artifact_dir):
        """从已付款订单构造运行器，避免混用本地工作台或其他订单的配置。"""
        order = self.store.get_order(artifact_dir.parent.name, by="trade_run_id")
        if not order or order["status"] not in {"PAID", "RUNNING"}:
            raise ValueError("No paid order owns this analysis run")
        profile = json.loads(order["params_json"])
        config = deepcopy(self.base_config)
        config.update(profile["config"])
        # 分析图仍存在进程级配置，因此只使用一个分析线程。
        # 每个订单独立保存记忆、缓存和日志，避免跨订单串用分析数据。
        root = self.settings.data_dir / "orders" / order["id"] / "analysis"
        config.update({"results_dir": str(root / "logs"), "data_cache_dir": str(root / "cache"),
                       "memory_log_path": str(root / "memory.md")})
        logger.info("开始准备订单分析 event=analysis_starting order_id=%s run_id=%s ticker=%s language=%s",
                    order["id"], order["trade_run_id"], order["ticker"], order["language"])
        return AnalysisRunner(request, config=config, artifact_dir=artifact_dir)

    def deliver_once(self) -> bool:
        """领取一封通知并尝试发送；空队列直接返回，不在每秒轮询时打印日志。"""
        delivery = self.store.claim_email()
        if not delivery:
            return False
        order = self.store.get_order(delivery["order_id"])
        if order["status"] == "REFUNDED":
            logger.info("停止已退款订单的邮件交付 event=email_skipped order_id=%s delivery_id=%s",
                        order["id"], delivery["id"])
            self.store.finish_email({**delivery, "attempt_count": 4}, None, "Order was refunded.")
            return True
        # 首次发送后冻结邮件内容；重试保持正文、收件人和幂等键一致。
        # 即使服务商已接受但响应丢失，也不会因重试生成第二封通知。
        payload = json.loads(delivery["payload_json"]) if delivery["payload_json"] else self.email.payload(order, delivery["template_type"])
        if not delivery["payload_json"]:
            self.store.save_email_payload(delivery["id"], payload)
        logger.info("开始发送通知邮件 event=email_sending order_id=%s delivery_id=%s attempt=%s",
                    order["id"], delivery["id"], delivery["attempt_count"])
        try:
            message_id = self.email.send(delivery, payload)
        except ProviderUnavailable as exc:
            logger.warning("邮件发送未确认 event=email_send_failed order_id=%s delivery_id=%s error_type=%s",
                           order["id"], delivery["id"], type(exc).__name__)
            self.store.finish_email(delivery, None, str(exc))
        else:
            self.store.finish_email(delivery, message_id)
        return True


class CommerceRuntime:
    """用文件锁保护单实例，避免重复服务把仍在执行的任务误判成中断。"""

    def __init__(self, service: CommerceService, *, runner_factory=None):
        self.service = service
        self.runner_factory = runner_factory or service.runner_factory
        self.manager = None
        self.lock_file = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self) -> None:
        import fcntl

        # 必须先取得所有权，再恢复任务和待发邮件；顺序反过来会干扰其他实例。
        self.lock_file = (self.service.settings.data_dir / "worker.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.error("数据目录已有运行实例 event=runtime_lock_conflict")
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError("Another commerce worker owns this directory. Use one server process.") from None
        try:
            self.service.store.recover_deliveries()
            self.manager = AnalysisTaskManager(self.service.store.tasks, runner_factory=self.runner_factory)
            self.service.store.reconcile()
            self.thread = threading.Thread(target=self._loop, name="report-delivery", daemon=True)
            self.thread.start()
            logger.info("付费报告服务已启动 event=runtime_started mode=%s sales_enabled=%s missing_configuration=%s",
                        self.service.settings.creem_mode, self.service.settings.sales_enabled,
                        ",".join(self.service.settings.missing_configuration()) or "none")
        except BaseException:
            self.stop()
            raise

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            stage = "report_reconciliation"
            try:
                self.service.store.reconcile()
                stage = "email_delivery"
                if self.service.settings.resend_api_key and self.service.settings.email_from:
                    self.service.deliver_once()
            except Exception as exc:
                # 第三方异常可能含邮箱、正文或凭证，不打印原异常及 traceback。
                logger.error("交付协调器处理失败 event=coordinator_failed stage=%s error_type=%s",
                             stage, type(exc).__name__)
            self.stop_event.wait(1)

    def stop(self) -> None:
        logger.info("正在停止付费报告服务 event=runtime_stopping")
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)
        if self.manager:
            self.manager.shutdown(timeout=5)
        # 后台线程未退出时不能提前释放锁；进程退出后由系统释放。
        if ((self.manager and self.manager._worker.is_alive())
                or (self.thread and self.thread.is_alive())):
            logger.warning("退出时后台任务尚未结束，下次启动将恢复中断状态 event=runtime_stop_pending")
            return
        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None
        logger.info("付费报告服务已停止 event=runtime_stopped")

    def healthy(self) -> bool:
        return bool(self.manager and self.manager._worker.is_alive()
                    and self.thread and self.thread.is_alive() and not self.stop_event.is_set())
