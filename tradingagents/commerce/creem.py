"""Creem 接入边界：浏览器返回值不可信，交付必须依据验签后的付款信息。"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

from tradingagents.commerce.config import CommerceSettings
from tradingagents.commerce.observability import log_event

logger = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    """A transient dependency error, safe to show without upstream payloads."""


class CheckoutUncertain(ProviderUnavailable):
    """Creation may have succeeded remotely; never blindly create another checkout."""


class PaymentRejected(ValueError):
    """A signed event failed the merchant's purchase contract."""


class PriceChanged(ValueError):
    """The visitor must review the current price before creating a checkout."""


@dataclass(frozen=True)
class Payment:
    checkout_id: str
    order_id: str
    transaction_id: str | None
    customer_id: str
    email: str
    paid_at: str


def verify_signature(raw: bytes, signature: str, secret: str) -> bool:
    """对原始字节计算 HMAC，并用恒定时间比较减少签名比较的时间差异。"""
    if not secret or not re.fullmatch(r"[a-fA-F0-9]{64}", signature):
        return False
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.lower())


def entity_id(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("id")
    return value if isinstance(value, str) and 0 < len(value) <= 128 else None


def is_number(value) -> bool:
    """官方 schema 中金额与数量字段均为 number 类型，int 与 float 都合法；bool 不算。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CreemClient:
    def __init__(self, settings: CommerceSettings, transport=None):
        self.settings = settings
        self.transport = transport or requests
        self.base_url = (
            "https://api.creem.io" if settings.creem_mode == "prod"
            else "https://test-api.creem.io"
        )
        self._product = None
        self._product_expires_at = 0.0
        self._product_lock = threading.Lock()

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """统一限制超时，并记录脱敏后的请求、响应和耗时。"""
        started = time.monotonic()
        context = {"provider": "creem", "exchange_id": uuid.uuid4().hex, "method": method,
                   "path": path, "mode": self.settings.creem_mode}
        log_event(logger, "creem_request", **context,
                  request={"params": kwargs.get("params"), "json": kwargs.get("json")})
        try:
            response = self.transport.request(
                method, self.base_url + path,
                headers={"x-api-key": self.settings.creem_api_key},
                timeout=(3.05, 12), allow_redirects=False, **kwargs,
            )
        except requests.RequestException as exc:
            log_event(logger, "creem_request_failed", level=logging.WARNING, **context,
                      elapsed_ms=round((time.monotonic() - started) * 1000), error_type=type(exc).__name__)
            # 创建操作可能已经被远端接受，异常类型提醒上层不要盲目重复 POST。
            error = CheckoutUncertain if method == "POST" else ProviderUnavailable
            raise error("Payment service temporarily unavailable.") from None
        try:
            data = response.json()
            body_format = "json"
        except ValueError:
            data, body_format = None, "non_json"
        headers = getattr(response, "headers", {})
        log_event(logger, "creem_response", level=logging.INFO if 200 <= response.status_code < 300 else logging.WARNING,
                  **context, http_status=response.status_code, elapsed_ms=round((time.monotonic() - started) * 1000),
                  provider_request_id=headers.get("x-request-id"), content_type=headers.get("content-type"),
                  body_format=body_format, response=data)
        if response.status_code >= 500 or response.status_code in {408, 429}:
            logger.warning("Creem 暂时不可用 event=creem_http_failed method=%s http_status=%s",
                           method, response.status_code)
            error = CheckoutUncertain if method == "POST" else ProviderUnavailable
            raise error("Payment service temporarily unavailable.")
        if not 200 <= response.status_code < 300:
            logger.warning("Creem 拒绝请求 event=creem_http_failed method=%s http_status=%s",
                           method, response.status_code)
            raise ProviderUnavailable(f"Payment service returned HTTP {response.status_code}.")
        if not isinstance(data, dict):
            logger.warning("Creem 响应格式无效 event=creem_response_invalid method=%s", method)
            error = CheckoutUncertain if method == "POST" else ProviderUnavailable
            raise error("Payment service returned an invalid response.") from None
        return data

    def get_product(self, *, refresh: bool = False) -> dict:
        """首页短暂缓存公开商品信息；新下单总是重新从商户 API 读取。"""
        with self._product_lock:
            if not refresh and self._product and time.monotonic() < self._product_expires_at:
                return dict(self._product)
            try:
                raw = self._request("GET", "/v1/products", params={"product_id": self.settings.product_id})
                currency = raw.get("currency")
                if (entity_id(raw) != self.settings.product_id
                        or not is_number(raw.get("price")) or raw["price"] <= 0
                        or not isinstance(currency, str) or not re.fullmatch(r"[A-Za-z]{3}", currency)
                        or raw.get("billing_type") != "onetime"
                        or raw.get("tax_mode") not in ("inclusive", "exclusive")
                        or raw.get("mode", self.settings.creem_mode) != self.settings.creem_mode
                        or raw.get("status", "active") != "active"):
                    raise PaymentRejected("Configure an active one-time Creem product with a valid price and tax mode.")
                product = {"id": raw["id"], "price": raw["price"], "currency": currency.upper(),
                           "billing_type": "onetime", "tax_mode": raw["tax_mode"], "mode": self.settings.creem_mode}
                # 仅用于发现页面展示后价格变化；价格本身始终来自服务端查询。
                product["quote"] = hashlib.sha256(json.dumps(product, sort_keys=True).encode()).hexdigest()
            except (ProviderUnavailable, PaymentRejected):
                self._product = None
                self._product_expires_at = 0.0
                raise
            self._product = product
            self._product_expires_at = time.monotonic() + 60
            return dict(product)

    def create_checkout(self, order: dict, product: dict) -> dict:
        self._validate_product(product, order)
        result = self._request("POST", "/v1/checkouts", json={
            "product_id": order["product_id"],
            "request_id": order["id"],
            "units": 1,
            "success_url": f"{self.settings.public_url}/success/{order['status_token']}",
            "metadata": {"order_id": order["id"]},
        })
        try:
            url = urlsplit(result.get("checkout_url", ""))
            host = url.hostname or ""
            valid_url = (url.scheme == "https" and (host == "creem.io" or host.endswith(".creem.io"))
                         and not url.username and not url.password)
        except (TypeError, ValueError, AttributeError):
            valid_url = False
        # 按官方 API 规范，创建收银台的响应不回显 request_id（仅在 webhook 对象与
        # GET /v1/checkouts 的查询结果中返回），此处只校验响应确实存在的字段。
        checks = {
            "checkout_id_present": bool(entity_id(result)),
            "mode_matches": result.get("mode") == self.settings.creem_mode,
            "product_id_matches": entity_id(result.get("product")) == order["product_id"],
            "checkout_url_valid": bool(valid_url),
        }
        if not all(checks.values()):
            log_event(logger, "checkout_validation_failed", level=logging.WARNING, order_id=order["id"],
                      failed_checks=[name for name, passed in checks.items() if not passed],
                      expected={"product_id": order["product_id"],
                                "mode": self.settings.creem_mode}, response=result)
            raise CheckoutUncertain("Payment service returned an unexpected checkout.")
        return result

    def _validate_product(self, product: dict, order: dict) -> None:
        if (entity_id(product) != order["product_id"]
                or not is_number(product.get("price")) or product["price"] != order["amount"]
                or product.get("currency", "").upper() != order["currency"]
                or product.get("billing_type") != "onetime"
                or product.get("tax_mode") != order["tax_mode"]
                or product.get("mode", self.settings.creem_mode) != self.settings.creem_mode):
            raise PaymentRejected("Product must match the order's saved price, currency and tax mode.")

    def validate_payment(self, checkout: dict, order: dict) -> Payment:
        """验证付款与本地订单的一致性，再提取可信的交付邮箱。"""
        checkout_id = entity_id(checkout)
        if not checkout_id or checkout.get("request_id") != order["id"]:
            raise PaymentRejected("Checkout reference mismatch.")
        if order["creem_checkout_id"] and checkout_id != order["creem_checkout_id"]:
            raise PaymentRejected("Checkout ID mismatch.")
        # 通知有时只携带对象 ID，必须用商户 API 补齐，不能相信浏览器补充的数据。
        if (not isinstance(checkout.get("order"), dict)
                or not isinstance(checkout.get("product"), dict)
                or not isinstance(checkout.get("customer"), dict)
                or "status" not in checkout or "mode" not in checkout):
            fresh = self._request("GET", "/v1/checkouts", params={"checkout_id": checkout_id})
            if entity_id(fresh) != checkout_id or fresh.get("request_id") != order["id"]:
                raise PaymentRejected("Retrieved checkout mismatch.")
            checkout = fresh
        product = checkout.get("product")
        if isinstance(product, str):
            product = self._request("GET", "/v1/products", params={"product_id": product})
        customer = checkout.get("customer")
        if isinstance(customer, str):
            customer = self._request("GET", "/v1/customers", params={"customer_id": customer})
        remote_order = checkout.get("order")
        if not all(isinstance(obj, dict) for obj in (product, customer, remote_order)):
            raise ProviderUnavailable("Incomplete payment information; retry the webhook.")
        # 商品可能在订单创建后调价；按订单快照与实际付款核验，不读取当前售价作为标准。
        if (entity_id(product) != order["product_id"]
                or product.get("mode", self.settings.creem_mode) != self.settings.creem_mode):
            raise PaymentRejected("Payment product does not match this order.")
        paid = remote_order.get("amount_paid", remote_order.get("amount"))
        due = remote_order.get("amount_due", remote_order.get("amount"))
        tax = remote_order.get("tax_amount", 0)
        if not is_number(tax) or tax < 0 or order["tax_mode"] not in {"inclusive", "exclusive"}:
            raise PaymentRejected("Invalid payment tax information.")
        expected_total = order["amount"] + tax if order["tax_mode"] == "exclusive" else order["amount"]
        expected_subtotal = expected_total - tax
        if (expected_subtotal < 0
                or ("sub_total" in remote_order and (not is_number(remote_order["sub_total"])
                                                     or remote_order["sub_total"] != expected_subtotal))):
            raise PaymentRejected("Payment subtotal does not match the order's saved price.")
        if (checkout.get("status") != "completed"
                or checkout.get("mode") != self.settings.creem_mode
                or not is_number(checkout.get("units", 1)) or checkout.get("units", 1) != 1
                or remote_order.get("status") != "paid"
                or remote_order.get("type") != "onetime"
                or remote_order.get("mode", self.settings.creem_mode) != self.settings.creem_mode
                or entity_id(remote_order.get("product")) != order["product_id"]
                or remote_order.get("currency", "").upper() != order["currency"]
                or not is_number(paid) or paid != expected_total
                or not is_number(due) or due != expected_total
                or remote_order.get("discount_amount", 0) not in (None, 0)
                or checkout.get("subscription")):
            raise PaymentRejected("Payment does not match this purchase.")
        customer_id = entity_id(customer)
        remote_id = entity_id(remote_order)
        email = customer.get("email", "")
        if (not customer_id or not remote_id
                or entity_id(remote_order.get("customer")) != customer_id
                or not isinstance(email, str) or len(email) > 254
                or not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", email)):
            raise PaymentRejected("Missing or inconsistent payment customer.")
        stamp = remote_order.get("created_at")
        try:
            paid_at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if paid_at.tzinfo is None:
                raise ValueError
        except (ValueError, AttributeError):
            raise PaymentRejected("Missing payment timestamp.") from None
        return Payment(checkout_id, remote_id, entity_id(remote_order.get("transaction")),
                       customer_id, email, paid_at.astimezone(timezone.utc).isoformat())

    def refund_details(self, refund: dict) -> dict:
        transaction = refund.get("transaction")
        transaction_id = entity_id(transaction)
        if not transaction_id:
            raise PaymentRejected("Missing refund transaction.")
        if (not isinstance(transaction, dict) or not entity_id(transaction.get("order"))
                or not transaction.get("currency")):
            transaction = self._request("GET", "/v1/transactions", params={"transaction_id": transaction_id})
        if (entity_id(transaction) != transaction_id or not entity_id(transaction.get("order"))
                or transaction.get("mode", self.settings.creem_mode) != self.settings.creem_mode
                or refund.get("status") != "succeeded"
                or not isinstance(refund.get("refund_currency"), str)
                or not re.fullmatch(r"[A-Z]{3}", refund["refund_currency"])
                or transaction.get("currency") != refund["refund_currency"]):
            raise PaymentRejected("Unconfirmed or inconsistent refund.")
        return {
            "remote_order_id": entity_id(transaction["order"]),
            "full_refund": transaction.get("status") == "refunded",
        }
