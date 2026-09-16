"""Verify one-use reCAPTCHA v2 tokens before creating a checkout."""

from urllib.parse import urlsplit

import requests

from tradingagents.commerce.config import CommerceSettings


class CaptchaRejected(Exception):
    """The visitor must complete a fresh challenge."""


class CaptchaUnavailable(Exception):
    """Verification cannot be completed; checkout must stay closed."""


def verify_recaptcha(token: str, settings: CommerceSettings) -> None:
    if not settings.recaptcha_site_key or not settings.recaptcha_secret_key:
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.")
    if not token.strip():
        raise CaptchaRejected("Please complete the human verification before continuing.")
    try:
        # Do not retry automatically: Google tokens expire after two minutes and
        # can only be verified once. Never log the token, secret or response body.
        response = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": settings.recaptcha_secret_key, "response": token},
            timeout=(3.05, 5), allow_redirects=False,
        )
        if response.status_code != 200:
            raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.")
        result = response.json()
    except (requests.RequestException, ValueError):
        raise CaptchaUnavailable("Human verification is temporarily unavailable. Please try again later.") from None
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
