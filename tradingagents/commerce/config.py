"""服务端产品配置；导入本模块不读取密钥，启动时才从环境变量构造配置。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

# 页面提交语言代码，订单保存该代码；元组依次为模型提示词语言与页面显示名称。
LANGUAGES = {
    "en": ("English", "English"),
    "zh-CN": ("Chinese", "简体中文"),
    "ja": ("Japanese", "日本語"),
    "ko": ("Korean", "한국어"),
    "hi": ("Hindi", "हिन्दी"),
    "es": ("Spanish", "Español"),
    "pt": ("Portuguese", "Português"),
    "fr": ("French", "Français"),
    "de": ("German", "Deutsch"),
    "ar": ("Arabic", "العربية"),
    "ru": ("Russian", "Русский"),
}
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


@dataclass(frozen=True)
class CommerceSettings:
    data_dir: Path = field(default_factory=lambda: Path.home() / ".tradingagents" / "commerce")
    public_url: str = "http://localhost:8000"
    creem_mode: str = "test"
    product_id: str = ""
    amount: int = 599
    currency: str = "USD"
    creem_api_key: str = field(default="", repr=False)
    webhook_secret: str = field(default="", repr=False)
    resend_api_key: str = field(default="", repr=False)
    email_from: str = ""
    support_email: str = ""
    sales_enabled: bool = False
    max_pending_runs: int = 3
    stalled_after_seconds: int = 1800
    checkout_rate_limit: int = 5
    read_rate_limit: int = 60
    recaptcha_enabled: bool = True
    recaptcha_site_key: str = ""
    recaptcha_secret_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> CommerceSettings:
        flag = os.getenv("COMMERCE_SALES_ENABLED", "false").lower()
        if flag not in {"true", "false", "1", "0"}:
            raise ValueError("COMMERCE_SALES_ENABLED must be true or false")
        captcha_flag = os.getenv("COMMERCE_RECAPTCHA_ENABLED", "true").lower()
        if captcha_flag not in {"true", "false", "1", "0"}:
            raise ValueError("COMMERCE_RECAPTCHA_ENABLED must be true or false")
        settings = cls(
            data_dir=Path(os.getenv("COMMERCE_DATA_DIR", str(Path.home() / ".tradingagents/commerce"))).expanduser().resolve(),
            public_url=os.getenv("COMMERCE_PUBLIC_URL", "http://localhost:8000").rstrip("/"),
            creem_mode=os.getenv("CREEM_MODE", "test"),
            product_id=os.getenv("CREEM_PRODUCT_ID", ""),
            creem_api_key=os.getenv("CREEM_API_KEY", ""),
            webhook_secret=os.getenv("CREEM_WEBHOOK_SECRET", ""),
            resend_api_key=os.getenv("RESEND_API_KEY", ""),
            email_from=os.getenv("COMMERCE_EMAIL_FROM", ""),
            support_email=os.getenv("COMMERCE_SUPPORT_EMAIL", ""),
            sales_enabled=flag in {"true", "1"},
            max_pending_runs=int(os.getenv("COMMERCE_MAX_PENDING_RUNS", "3")),
            stalled_after_seconds=int(os.getenv("COMMERCE_STALLED_AFTER_SECONDS", "1800")),
            checkout_rate_limit=int(os.getenv("COMMERCE_CHECKOUT_RATE_LIMIT", "5")),
            read_rate_limit=int(os.getenv("COMMERCE_READ_RATE_LIMIT", "60")),
            recaptcha_enabled=captcha_flag in {"true", "1"},
            recaptcha_site_key=os.getenv("RECAPTCHA_SITE_KEY", "").strip(),
            recaptcha_secret_key=os.getenv("RECAPTCHA_SECRET_KEY", "").strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        url = urlsplit(self.public_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.path
                or url.query or url.fragment or url.username or url.password):
            raise ValueError("COMMERCE_PUBLIC_URL must be an origin, e.g. https://reports.example.com")
        if self.creem_mode not in {"test", "prod"}:
            raise ValueError("CREEM_MODE must be test or prod")
        if self.creem_mode == "prod" and url.scheme != "https":
            raise ValueError("Production requires an HTTPS public URL")
        if self.amount != 599 or self.currency != "USD":
            raise ValueError("This product is fixed at USD 5.99, tax inclusive")
        if self.max_pending_runs < 1 or self.stalled_after_seconds < 60:
            raise ValueError("Invalid queue capacity or stalled-run threshold")
        if self.checkout_rate_limit < 1 or self.read_rate_limit < 1:
            raise ValueError("Request rate limits must be positive integers")
        if self.creem_mode == "prod":
            if not self.recaptcha_enabled:
                raise ValueError("Production requires reCAPTCHA verification")
            # Google's public v2 test keys always pass; never accept them in production.
            if (self.recaptcha_site_key == "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"
                    or self.recaptcha_secret_key == "6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe"):
                raise ValueError("Production requires real reCAPTCHA keys")
        if any("\n" in x or "\r" in x for x in (self.email_from, self.support_email)):
            raise ValueError("Email settings must not contain newlines")

    def missing_configuration(self) -> list[str]:
        fields = {
            "CREEM_PRODUCT_ID": self.product_id,
            "CREEM_API_KEY": self.creem_api_key,
            "CREEM_WEBHOOK_SECRET": self.webhook_secret,
            "RESEND_API_KEY": self.resend_api_key,
            "COMMERCE_EMAIL_FROM": self.email_from,
            "COMMERCE_SUPPORT_EMAIL": self.support_email,
        }
        if self.recaptcha_enabled:
            fields.update({
                "RECAPTCHA_SITE_KEY": self.recaptcha_site_key,
                "RECAPTCHA_SECRET_KEY": self.recaptcha_secret_key,
            })
        return [name for name, value in fields.items() if not value]
