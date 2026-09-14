"""Creem 接入边界：浏览器返回值不可信，交付必须依据验签后的付款信息。"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

from tradingagents.commerce.config import CommerceSettings

logger = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    """A transient dependency error, safe to show without upstream payloads."""


class CheckoutUncertain(ProviderUnavailable):
    """Creation may have succeeded remotely; never blindly create another checkout."""


class PaymentRejected(ValueError):
    """A signed event failed the merchant's purchase contract."""


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


class CreemClient:
    def __init__(self, settings: CommerceSettings, transport=None):
        self.settings = settings
        self.transport = transport or requests
        self.base_url = (
            "https://api.creem.io" if settings.creem_mode == "prod"
            else "https://test-api.creem.io"
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """统一限制超时；只记录 HTTP 状态及异常类型，不记录请求/响应内容。"""
        try:
            response = self.transport.request(
                method, self.base_url + path,
                headers={"x-api-key": self.settings.creem_api_key},
                timeout=(3.05, 12), allow_redirects=False, **kwargs,
            )
        except requests.RequestException as exc:
            logger.warning("Creem 请求失败 event=creem_request_failed method=%s error_type=%s",
                           method, type(exc).__name__)
            # 创建操作可能已经被远端接受，异常类型提醒上层不要盲目重复 POST。
            error = CheckoutUncertain if method == "POST" else ProviderUnavailable
            raise error("Payment service temporarily unavailable.") from None
        if response.status_code >= 500 or response.status_code in {408, 429}:
            logger.warning("Creem 暂时不可用 event=creem_http_failed method=%s http_status=%s",
                           method, response.status_code)
            error = CheckoutUncertain if method == "POST" else ProviderUnavailable
            raise error("Payment service temporarily unavailable.")
        if not 200 <= response.status_code < 300:
            logger.warning("Creem 拒绝请求 event=creem_http_failed method=%s http_status=%s",
                           method, response.status_code)
            raise ProviderUnavailable(f"Payment service returned HTTP {response.status_code}.")
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError
            return data
        except ValueError:
            logger.warning("Creem 响应格式无效 event=creem_response_invalid method=%s", method)
            error = CheckoutUncertain if method == "POST" else ProviderUnavailable
            raise error("Payment service returned an invalid response.") from None

    def create_checkout(self, order: dict) -> dict:
        product = self._request("GET", f"/v1/products/{order['product_id']}")
        self._validate_product(product, order)
        result = self._request("POST", "/v1/checkouts", json={
            "product_id": order["product_id"],
            "request_id": order["id"],
            "units": 1,
            "success_url": f"{self.settings.public_url}/success/{order['status_token']}",
            "metadata": {"order_id": order["id"]},
        })
        url = urlsplit(result.get("checkout_url", ""))
        host = url.hostname or ""
        if (not entity_id(result) or result.get("request_id") != order["id"]
                or result.get("mode") != self.settings.creem_mode
                or entity_id(result.get("product")) != order["product_id"]
                or url.scheme != "https" or not (host == "creem.io" or host.endswith(".creem.io"))
                or url.username or url.password):
            raise CheckoutUncertain("Payment service returned an unexpected checkout.")
        return result

    def _validate_product(self, product: dict, order: dict) -> None:
        if (entity_id(product) != order["product_id"]
                or type(product.get("price")) is not int or product["price"] != order["amount"]
                or product.get("currency", "").upper() != order["currency"]
                or product.get("billing_type") != "onetime"
                or product.get("tax_mode") != "inclusive"
                or product.get("mode", self.settings.creem_mode) != self.settings.creem_mode):
            raise PaymentRejected("Product must match the configured one-time, tax-inclusive price.")

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
            product = self._request("GET", f"/v1/products/{product}")
        customer = checkout.get("customer")
        if isinstance(customer, str):
            customer = self._request("GET", "/v1/customers", params={"customer_id": customer})
        remote_order = checkout.get("order")
        if not all(isinstance(obj, dict) for obj in (product, customer, remote_order)):
            raise ProviderUnavailable("Incomplete payment information; retry the webhook.")
        self._validate_product(product, order)
        paid = remote_order.get("amount_paid", remote_order.get("amount"))
        due = remote_order.get("amount_due", remote_order.get("amount"))
        # 当前商品为含税定价，应付和实付都必须等于 599 美分。
        # sub_total 可能是税前小计，不应拿它与含税售价比较。
        if (checkout.get("status") != "completed"
                or checkout.get("mode") != self.settings.creem_mode
                or type(checkout.get("units", 1)) is not int or checkout.get("units", 1) != 1
                or remote_order.get("status") != "paid"
                or remote_order.get("type") != "onetime"
                or remote_order.get("mode", self.settings.creem_mode) != self.settings.creem_mode
                or entity_id(remote_order.get("product")) != order["product_id"]
                or remote_order.get("currency", "").upper() != order["currency"]
                or type(paid) is not int or paid != order["amount"]
                or type(due) is not int or due != order["amount"]
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
        if not isinstance(transaction, dict) or not entity_id(transaction.get("order")):
            transaction = self._request("GET", "/v1/transactions", params={"transaction_id": transaction_id})
        if (entity_id(transaction) != transaction_id or not entity_id(transaction.get("order"))
                or transaction.get("mode", self.settings.creem_mode) != self.settings.creem_mode
                or refund.get("status") != "succeeded"
                or refund.get("refund_currency") != self.settings.currency):
            raise PaymentRejected("Unconfirmed or inconsistent refund.")
        return {
            "remote_order_id": entity_id(transaction["order"]),
            "full_refund": transaction.get("status") == "refunded",
        }
