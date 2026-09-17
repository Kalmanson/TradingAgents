"""Verify one-use reCAPTCHA v2 tokens before creating a checkout."""

import logging
import time
from urllib.parse import urlsplit

import requests

from tradingagents.commerce.config import CommerceSettings
from tradingagents.commerce.observability import log_event

logger = logging.getLogger(__name__)


class CaptchaRejected(Exception):
    """The visitor must complete a fresh challenge."""


class CaptchaUnavailable(Exception):
    """Verification cannot be completed; checkout must stay closed."""


def verify_recaptcha(token: str, settings: CommerceSettings) -> None:
    if not settings.recaptcha_site_key or not settings.recaptcha_secret_key:
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.")
    if not token.strip():
        raise CaptchaRejected("Please complete the human verification before continuing.")
    started = time.monotonic()
    context = {"provider": "google_recaptcha", "method": "POST", "path": "/recaptcha/api/siteverify"}
    log_event(logger, "recaptcha_request", **context,
              request={"secret": settings.recaptcha_secret_key, "response": token})
    try:
        # Do not retry automatically: Google tokens expire after two minutes and
        # can only be verified once. Log only the sanitized protocol fields.
        response = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": settings.recaptcha_secret_key, "response": token},
            timeout=(3.05, 5), allow_redirects=False,
        )
    except requests.RequestException as exc:
        log_event(logger, "recaptcha_request_failed", level=logging.WARNING, **context,
                  elapsed_ms=round((time.monotonic() - started) * 1000), error_type=type(exc).__name__)
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.") from None
    try:
        result = response.json()
        body_format = "json"
    except ValueError:
        result, body_format = None, "non_json"
    log_event(logger, "recaptcha_response", **context,
              level=logging.INFO if response.status_code == 200 else logging.WARNING,
              http_status=response.status_code, elapsed_ms=round((time.monotonic() - started) * 1000),
              body_format=body_format, response=result, expected_hostname=urlsplit(settings.public_url).hostname)
    if response.status_code != 200:
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.")
    if not isinstance(result, dict) or type(result.get("success")) is not bool:
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.")
    errors = result.get("error-codes", [])
    if not isinstance(errors, list) or any(code in errors for code in (
        "missing-input-secret", "invalid-input-secret", "bad-request",
    )):
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.")
    hostname = result.get("hostname")
    if (not result["success"] or errors or not isinstance(hostname, str)
            or hostname.lower() != urlsplit(settings.public_url).hostname):
        raise CaptchaRejected("Human verification failed or expired. Please complete it again.")
