"""付费网站页面与接口；只有经过验签的服务端支付通知能够触发交付。"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, closing
from decimal import Decimal
from pathlib import Path

import nh3
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from tradingagents.commerce.captcha import CaptchaRejected, CaptchaUnavailable, verify_recaptcha
from tradingagents.commerce.config import LANGUAGES, TOKEN_PATTERN, CommerceSettings
from tradingagents.commerce.creem import (
    PaymentRejected,
    PriceChanged,
    ProviderUnavailable,
    verify_signature,
)
from tradingagents.commerce.observability import log_event, trace_id
from tradingagents.commerce.service import CommerceRuntime, CommerceService
from tradingagents.commerce.store import CommerceStore

logger = logging.getLogger(__name__)


class PurchaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=10)
    language: str = Field(default="en", min_length=2, max_length=8)
    recaptcha_token: str = Field(default="", max_length=4096, repr=False)
    product_quote: str = Field(default="", max_length=64)


class RateLimiter:
    def __init__(self):
        self.buckets = OrderedDict()
        self.lock = threading.Lock()

    def allow(self, key: tuple, limit: int) -> bool:
        now = time.monotonic()
        with self.lock:
            bucket = self.buckets.setdefault(key, deque())
            self.buckets.move_to_end(key)
            while bucket and bucket[0] <= now - 60:
                bucket.popleft()
            if len(self.buckets) > 4096:
                self.buckets.popitem(last=False)
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


def create_app(settings: CommerceSettings | None = None, *, service: CommerceService | None = None,
               start_workers: bool = True, runner_factory=None) -> FastAPI:
    settings = settings or CommerceSettings.from_env()
    settings.validate()
    service = service or CommerceService(settings, CommerceStore(settings))
    runtime = CommerceRuntime(service, runner_factory=runner_factory)

    @asynccontextmanager
    async def lifespan(app):
        if start_workers:
            await run_in_threadpool(runtime.start)
        try:
            yield
        finally:
            if start_workers:
                await run_in_threadpool(runtime.stop)

    app = FastAPI(title="AI Investment Committee", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service
    app.state.runtime = runtime
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
    limiter = RateLimiter()
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.middleware("http")
    async def boundaries(request: Request, call_next):
        path = request.url.path
        client = request.client.host if request.client else "unknown"
        limit = settings.checkout_rate_limit if path == "/api/orders" else settings.read_rate_limit
        category = "checkout" if path == "/api/orders" else "read"
        response = None
        if path.startswith(("/api/orders", "/report/", "/success/")) and not limiter.allow((client, category), limit):
            response = JSONResponse({"detail": "Too many requests. Please wait 60 seconds before trying again."}, status_code=429, headers={"Retry-After": "60"})
        elif path == "/api/orders" and request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != settings.public_url:
                response = JSONResponse({"detail": "Origin not allowed."}, status_code=403)
            elif request.headers.get("content-type", "").split(";")[0] != "application/json":
                response = JSONResponse({"detail": "Expected application/json."}, status_code=415)
        if response is None:
            response = await call_next(request)
        script_src = "'self'"
        frame_src = "'none'"
        connect_src = "'self'"
        if path == "/" and settings.recaptcha_enabled:
            script_src += " https://www.google.com/recaptcha/ https://www.gstatic.com/recaptcha/"
            frame_src = "https://www.google.com/recaptcha/ https://recaptcha.google.com/recaptcha/"
            connect_src += " https://www.google.com/recaptcha/"
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": f"default-src 'self'; script-src {script_src}; style-src 'self'; img-src 'self'; connect-src {connect_src}; frame-src {frame_src}; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        })
        if path.startswith(("/api", "/report", "/success")):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Robots-Tag"] = "noindex, nofollow"
        if settings.public_url.startswith("https://"):
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.middleware("http")
    async def trace_interactions(request: Request, call_next):
        # Generate locally; do not trust/log a caller-supplied tracing header.
        token = trace_id.set(uuid.uuid4().hex)
        started = time.monotonic()
        tracked = request.url.path in {"/api/orders", "/api/webhooks/creem"}
        context = {"method": request.method, "path": request.url.path}
        try:
            if tracked:
                log_event(logger, "http_request", **context)
            response = await call_next(request)
            response.headers["X-Request-ID"] = trace_id.get()
            if tracked:
                log_event(logger, "http_response", **context,
                          level=logging.INFO if response.status_code < 400 else logging.WARNING,
                          http_status=response.status_code, elapsed_ms=round((time.monotonic() - started) * 1000))
            return response
        except Exception as exc:
            if tracked:
                log_event(logger, "http_request_failed", level=logging.ERROR, **context,
                          elapsed_ms=round((time.monotonic() - started) * 1000), error_type=type(exc).__name__)
            raise
        finally:
            trace_id.reset(token)

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        available = service.store.capacity_available() and (not start_workers or runtime.healthy())
        product = None
        try:
            product = service.creem.get_product()
            product["display_price"] = f"{Decimal(product['price']) / 100:,.2f}"
        except (ProviderUnavailable, PaymentRejected) as exc:
            logger.warning("商品价格暂不可用 event=product_unavailable error_type=%s", type(exc).__name__)
            available = False
        return templates.TemplateResponse(request=request, name="index.html", context={
            "languages": LANGUAGES, "available": available, "support_email": settings.support_email,
            "product": product,
            "recaptcha_site_key": settings.recaptcha_site_key if settings.recaptcha_enabled else "",
        }, status_code=200 if product else 503)

    @app.post("/api/orders")
    async def purchase(request: Request, idempotency_key: str = Header(default="")):
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > 8192:
                raise HTTPException(413, "Request too large.")
        try:
            inputs = PurchaseInput.model_validate_json(chunks)
        except ValueError:
            raise HTTPException(422, "Provide a valid ticker, report language and verification token.") from None
        log_event(logger, "purchase_request", request=inputs.model_dump())
        if start_workers and not runtime.healthy():
            raise HTTPException(503, "Report generation is temporarily unavailable.")
        if settings.recaptcha_enabled:
            try:
                await run_in_threadpool(verify_recaptcha, inputs.recaptcha_token, settings)
            except CaptchaRejected as exc:
                raise HTTPException(403, str(exc)) from None
            except CaptchaUnavailable as exc:
                raise HTTPException(503, str(exc)) from None
        try:
            order = await run_in_threadpool(service.create_checkout, inputs.ticker, inputs.language,
                                           idempotency_key, inputs.product_quote)
        except PriceChanged as exc:
            raise HTTPException(409, str(exc)) from None
        except PaymentRejected:
            raise HTTPException(503, "Checkout is not configured correctly. Please contact support.") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        except ProviderUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        response = {"checkoutUrl": order["checkout_url"], "statusUrl": f"/success/{order['status_token']}"}
        log_event(logger, "purchase_response", order_id=order["id"], response=response)
        return response

    @app.post("/api/webhooks/creem")
    async def webhook(request: Request):
        # 先限制请求体大小并验签，再解析 JSON；解析/重新序列化会改变签名输入。
        if not settings.webhook_secret:
            logger.warning("支付通知入口尚未配置 event=webhook_unconfigured")
            raise HTTPException(503, "Webhook is not configured.")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65_536:
                raise HTTPException(413, "Webhook too large.")
        raw = bytes(body)
        if not verify_signature(raw, request.headers.get("creem-signature", ""), settings.webhook_secret):
            # 此时所有报文内容均不可信，不记录其中的 ID、邮箱或签名。
            logger.warning("支付通知验签失败 event=webhook_signature_invalid")
            raise HTTPException(401, "Invalid webhook signature.")
        try:
            event = json.loads(raw)
            result = await run_in_threadpool(service.process_webhook, event, raw)
        except ValueError:
            logger.warning("支付通知格式无效 event=webhook_malformed")
            raise HTTPException(400, "Malformed webhook event.") from None
        except (ProviderUnavailable, sqlite3.OperationalError) as exc:
            logger.warning("支付确认暂未完成 event=webhook_unavailable error_type=%s", type(exc).__name__)
            raise HTTPException(503, "Payment confirmation is pending. Please retry.") from None
        response = {"received": True, "result": result}
        log_event(logger, "webhook_response", response=response)
        return response

    def public_status(order: dict) -> dict:
        return {"status": order["status"], "ticker": order["ticker"], "language": order["language"],
                "analysisDate": order["analysis_date"],
                "reportUrl": f"/report/{order['report_token']}" if order["status"] == "COMPLETED" else None,
                "message": order["error_message"] if order["status"] == "FAILED" else None}

    @app.get("/api/orders/status/{token}")
    def status(token: str):
        if not TOKEN_PATTERN.fullmatch(token):
            raise HTTPException(404, "Order not found.")
        order = service.store.get_order(token, by="status_token")
        if order is None:
            raise HTTPException(404, "Order not found.")
        return public_status(order)

    @app.get("/success/{token}", response_class=HTMLResponse)
    def success(request: Request, token: str):
        if not TOKEN_PATTERN.fullmatch(token):
            raise HTTPException(404, "Order not found.")
        order = service.store.get_order(token, by="status_token")
        if order is None:
            raise HTTPException(404, "Order not found.")
        # Strip Creem's redirect query; none of it is payment evidence.
        if request.url.query:
            return RedirectResponse(f"/success/{token}", status_code=303)
        return templates.TemplateResponse(request=request, name="status.html", context={
            "order": public_status(order), "status_token": token, "support_email": settings.support_email,
        })

    @app.get("/report/{token}", response_class=HTMLResponse)
    def report(request: Request, token: str):
        if not TOKEN_PATTERN.fullmatch(token):
            raise HTTPException(404, "Report not found.")
        order = service.store.get_order(token, by="report_token")
        if order is None:
            raise HTTPException(404, "Report not found.")
        if order["status"] != "COMPLETED":
            return templates.TemplateResponse(request=request, name="status.html", context={
                "order": public_status(order), "status_token": order["status_token"],
                "support_email": settings.support_email,
            }, status_code=410 if order["status"] == "REFUNDED" else 200)
        if not order["report_markdown"]:
            raise HTTPException(503, "Report is temporarily unavailable. Please contact support.")
        rendered = nh3.clean(markdown.render(order["report_markdown"]),
                             url_schemes={"https", "http"}, link_rel="noopener noreferrer")
        return templates.TemplateResponse(request=request, name="report.html", context={
            "order": public_status(order), "language_label": LANGUAGES[order["language"]][1],
            "report_html": rendered, "support_email": settings.support_email,
        })

    @app.get("/health/live")
    def live():
        return {"alive": True}

    @app.get("/health/ready")
    def ready():
        available = False
        try:
            with closing(service.store.tasks._connect()) as connection:
                connection.execute("SELECT 1 FROM trade_orders LIMIT 1").fetchone()
            healthy = not start_workers or runtime.healthy()
            available = service.store.capacity_available()
        except sqlite3.Error:
            healthy = False
        return JSONResponse({"ready": healthy, "acceptingOrders": healthy and available},
                            status_code=200 if healthy else 503)

    @app.get("/terms", response_class=HTMLResponse)
    @app.get("/privacy", response_class=HTMLResponse)
    def policy(request: Request):
        return templates.TemplateResponse(request=request, name="policy.html", context={
            "privacy": request.url.path == "/privacy", "support_email": settings.support_email,
        })

    return app
