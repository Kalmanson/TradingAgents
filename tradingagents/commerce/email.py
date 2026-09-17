"""只发送订单通知和报告链接；交付状态与重试记录由 SQLite 持久保存。"""

from __future__ import annotations

import html
import logging
import time
import uuid

import requests

from tradingagents.commerce.config import LANGUAGES, CommerceSettings
from tradingagents.commerce.creem import ProviderUnavailable
from tradingagents.commerce.observability import log_event

logger = logging.getLogger(__name__)


class EmailClient:
    def __init__(self, settings: CommerceSettings, transport=None):
        self.settings = settings
        self.transport = transport or requests

    def payload(self, order: dict, kind: str) -> dict:
        ready = kind == "REPORT_READY"
        ticker = order["ticker"]
        title = f"Your {ticker} AI Investment Committee Report is ready" if ready else f"An update on your {ticker} report"
        url = (f"{self.settings.public_url}/report/{order['report_token']}" if ready
               else f"{self.settings.public_url}/success/{order['status_token']}")
        description = ("Your analysis is complete. Open your private report using the link below."
                       if ready else "We could not complete your analysis. Our team will review the order and arrange a refund. You do not need to pay again.")
        language = LANGUAGES[order["language"]][1]
        disclaimer = "AI-generated research for information and education. Not personalized investment advice."
        support = self.settings.support_email
        text = f"{description}\n\nTicker: {ticker}\nReport language: {language}\n{url}\n\n{disclaimer}\nSupport: {support}"
        escaped_url = html.escape(url, quote=True)
        body = (f"<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p>"
                f"<p>Ticker: <strong>{html.escape(ticker)}</strong><br>Report language: {html.escape(language)}</p>"
                f'<p><a href="{escaped_url}">{"View report" if ready else "View order status"}</a></p>'
                f"<p>{html.escape(disclaimer)}</p><p>Support: {html.escape(support)}</p>")
        return {"from": self.settings.email_from, "to": [order["customer_email"]],
                "subject": title, "text": text, "html": body, "reply_to": support}

    def send(self, delivery: dict, payload: dict) -> str:
        """用固定的交付 ID 作为幂等键，网络失败由上层按同一内容重试。"""
        if not self.settings.resend_api_key:
            raise ProviderUnavailable("Resend is not configured.")
        started = time.monotonic()
        context = {"provider": "resend", "exchange_id": uuid.uuid4().hex, "method": "POST", "path": "/emails",
                   "delivery_id": delivery["id"], "order_id": delivery.get("order_id")}
        log_event(logger, "resend_request", **context, request=payload)
        try:
            response = self.transport.request(
                "POST", "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self.settings.resend_api_key}",
                         "Idempotency-Key": f"report/{delivery['id']}"},
                json=payload, timeout=(3.05, 12), allow_redirects=False,
            )
            try:
                result = response.json()
                body_format = "json"
            except ValueError:
                result, body_format = None, "non_json"
            headers = getattr(response, "headers", {})
            log_event(logger, "resend_response", **context,
                      level=logging.INFO if 200 <= response.status_code < 300 else logging.WARNING,
                      http_status=response.status_code, elapsed_ms=round((time.monotonic() - started) * 1000),
                      provider_request_id=headers.get("x-request-id"), body_format=body_format, response=result)
            if not 200 <= response.status_code < 300:
                logger.warning("Resend 拒绝邮件请求 event=resend_http_failed delivery_id=%s http_status=%s",
                               delivery["id"], response.status_code)
                raise ProviderUnavailable(f"Email service returned HTTP {response.status_code}.")
            message_id = result.get("id") if isinstance(result, dict) else None
            if not isinstance(message_id, str) or not message_id:
                raise ValueError
            return message_id
        except (requests.RequestException, ValueError) as exc:
            # 原异常可能带有请求内容；日志只保留已知交付 ID 和异常类型。
            log_event(logger, "resend_response_unconfirmed", level=logging.WARNING, **context,
                      elapsed_ms=round((time.monotonic() - started) * 1000), error_type=type(exc).__name__)
            raise ProviderUnavailable("Email response unavailable; retry with the same idempotency key.") from None
