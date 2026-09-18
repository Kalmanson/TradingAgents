"""Bounded, structured interaction logs without credentials or private content."""

import json
import logging
import re
from contextvars import ContextVar
from urllib.parse import urlsplit

trace_id: ContextVar[str] = ContextVar("commerce_trace_id", default="-")

_PRIVATE_FIELDS = {
    "authorization", "x-api-key", "creem-signature", "secret", "token", "recaptcha_token",
    "idempotency-key", "idempotency_key", "status_token", "report_token", "email", "customer_email",
    "from", "to", "reply_to", "subject", "text", "html", "name", "address", "ip_address",
}
_PUBLIC_FIELDS = {
    "id", "request_id", "order_id", "product_id", "checkout_id", "transaction_id", "customer_id",
    "event_id", "delivery_id", "exchange_id", "provider_request_id", "product", "customer", "order",
    "transaction", "mode", "status", "billing_type", "currency", "refund_currency", "tax_mode", "type",
    "eventType", "ticker", "language", "code", "error-codes", "error_type", "body_format", "method",
    "provider", "result", "failed_checks", "hostname", "expected_hostname", "content_type",
    "operation", "symbol", "quoteType", "quote_type", "exchange", "country",
}


def safe_payload(value, *, key: str = "", depth: int = 0):
    """Keep response shape and protocol fields; omit free text and bound all nesting."""
    if key.lower() in _PRIVATE_FIELDS:
        return "[redacted]"
    if depth >= 6:
        return "[truncated]"
    if isinstance(value, dict):
        result = {}
        for index, (field, item) in enumerate(value.items()):
            if index >= 40:
                result["_truncated"] = True
                break
            if isinstance(field, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", field):
                result[field] = safe_payload(item, key=field, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [safe_payload(item, key=key, depth=depth + 1) for item in value[:10]]
        return items + (["[truncated]"] if len(value) > 10 else [])
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if key.lower().endswith("url"):
            try:
                url = urlsplit(value)
                host = url.hostname or ""
                return {"scheme": url.scheme[:16], "host": host[:160], "path": "[redacted]"}
            except ValueError:
                return "[invalid URL]"
        if key in _PUBLIC_FIELDS and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", value):
            return value
        if key == "path" and re.fullmatch(r"/(?:api/|v1/|recaptcha/)[A-Za-z0-9_/-]{1,80}|/emails", value):
            return value
    return "[redacted]"


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields) -> None:
    """One JSON line per interaction, correlated across async tasks and worker threads."""
    if not logger.isEnabledFor(level):
        return
    payload = json.dumps(safe_payload(fields), ensure_ascii=True, separators=(",", ":"))
    if len(payload) > 8000:
        payload = json.dumps({"truncated": True, "preview": payload[:3000]}, separators=(",", ":"))
    logger.log(level, "event=%s trace_id=%s data=%s", event, trace_id.get(), payload)
