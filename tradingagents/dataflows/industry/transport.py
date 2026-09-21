"""Bounded public-document transport, with isolated caches and SEC fair access."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)
_SEC_LOCK = threading.Lock()
_SEC_LAST_REQUEST = 0.0
MAX_BYTES = 32 * 1024 * 1024


class SourceUnavailable(ValueError):
    """A bounded, credential-free explanation of missing evidence."""


class DocumentClient:
    def __init__(self, config: dict):
        self.root = Path(config["data_cache_dir"]) / "industry"
        self.blocked: set[str] = set()
        self.requests = 0
        self.cache_hits = 0

    def fetch(self, url: str, *, source: str, ttl: int | None = 21600) -> dict:
        """Return bytes and retrieval metadata; never cache errors or forward identity."""
        global _SEC_LAST_REQUEST
        url = urldefrag(url)[0]
        key = hashlib.sha256(url.encode()).hexdigest()
        metadata_path, body_path = self.root / f"{key}.json", self.root / f"{key}.bin"
        try:
            metadata = json.loads(metadata_path.read_text())
            if (ttl is None or time.time() - metadata["cached_at"] < ttl) and body_path.exists():
                body = body_path.read_bytes()
                if hashlib.sha256(body).hexdigest() == metadata["sha256"]:
                    self.cache_hits += 1
                    return {**metadata, "body": body, "cache_hit": True}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if source in self.blocked:
            raise SourceUnavailable(f"{source}: access blocked for this research run")

        headers = {"User-Agent": "TradingAgents industry research", "Accept-Encoding": "gzip, deflate"}
        if source == "sec":
            user_agent = os.getenv("SEC_USER_AGENT", "").strip()
            if not user_agent:
                contact = os.getenv("COMMERCE_SUPPORT_EMAIL", "").strip()
                if contact:
                    user_agent = f"TradingAgents research {contact}"
            if (not user_agent.isascii() or "\r" in user_agent or "\n" in user_agent
                    or not re.search(r"\S+@\S+\.\S+", user_agent)):
                raise SourceUnavailable("SEC_USER_AGENT requires an application name and contact email")
            headers["User-Agent"] = user_agent

        current_url = url
        with requests.Session() as session:
            for redirect in range(4):
                parsed = urlsplit(current_url)
                if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                        or parsed.password or parsed.port not in (None, 443)):
                    raise SourceUnavailable("Only public HTTPS document URLs are supported")
                if source == "sec" and parsed.hostname not in {"www.sec.gov", "data.sec.gov"}:
                    raise SourceUnavailable("SEC redirects to other hosts are not followed")
                try:
                    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
                    # Local DNS proxies (including Clash fake-IP mode) synthesize
                    # 198.18.0.0/15 addresses for public hostnames. Permit that
                    # mapping, while still rejecting LAN, loopback and link-local
                    # destinations. A literal benchmark address is never accepted.
                    synthetic = ipaddress.ip_network("198.18.0.0/15")
                    literal = re.fullmatch(r"[0-9.]+", parsed.hostname) is not None
                    if not addresses or any(
                        not ipaddress.ip_address(row[4][0]).is_global
                        and (literal or ipaddress.ip_address(row[4][0]) not in synthetic)
                        for row in addresses
                    ):
                        raise SourceUnavailable("Non-public document address rejected")
                except OSError:
                    raise SourceUnavailable(f"{source}: DNS unavailable") from None
                response = None
                for attempt in range(2):
                    if source == "sec":
                        with _SEC_LOCK:
                            delay = 1.0 - (time.monotonic() - _SEC_LAST_REQUEST)
                            if delay > 0:
                                time.sleep(delay)
                            _SEC_LAST_REQUEST = time.monotonic()
                    started = time.monotonic()
                    try:
                        self.requests += 1
                        response = session.get(current_url, headers=headers, timeout=(5, 30),
                                               allow_redirects=False, stream=True)
                        status = response.status_code
                        logger.info("industry_request source=%s status=%s elapsed_ms=%d", source, status,
                                    round((time.monotonic() - started) * 1000))
                        if status in {403, 429}:
                            self.blocked.add(source)
                            response.close()
                            raise SourceUnavailable(f"{source}: HTTP {status}; stopped for this run")
                        if status >= 500 and attempt == 0:
                            response.close()
                            time.sleep(1)
                            continue
                        break
                    except (requests.Timeout, requests.ConnectionError):
                        if attempt:
                            raise SourceUnavailable(f"{source}: network timeout or connection failure") from None
                        time.sleep(1)
                    except requests.RequestException:
                        raise SourceUnavailable(f"{source}: HTTP transport failure") from None
                if response is None:
                    raise SourceUnavailable(f"{source}: no HTTP response")
                with response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        if redirect == 3 or not response.headers.get("Location"):
                            raise SourceUnavailable(f"{source}: redirect limit exceeded")
                        current_url = urljoin(current_url, response.headers["Location"])
                        continue
                    if response.status_code != 200:
                        raise SourceUnavailable(f"{source}: HTTP {response.status_code}")
                    body = bytearray()
                    try:
                        for chunk in response.iter_content(65536):
                            body.extend(chunk)
                            if len(body) > MAX_BYTES:
                                raise SourceUnavailable(f"{source}: document exceeds 32 MB")
                    except requests.RequestException:
                        raise SourceUnavailable(f"{source}: incomplete document download") from None
                    metadata = {
                        "url": url, "content_type": response.headers.get("Content-Type", ""),
                        "retrieved_at": datetime.now(timezone.utc).isoformat(), "cached_at": time.time(),
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                self.root.mkdir(parents=True, exist_ok=True)
                # Atomic writes prevent interrupted downloads from poisoning future reads.
                for path, value in ((body_path, bytes(body)), (metadata_path, json.dumps(metadata).encode())):
                    temp = path.with_suffix(f".{uuid4().hex}.tmp")
                    try:
                        temp.write_bytes(value)
                        temp.replace(path)
                    finally:
                        temp.unlink(missing_ok=True)
                return {**metadata, "body": bytes(body), "cache_hit": False}
        raise SourceUnavailable(f"{source}: document unavailable")


def evidence(source: str, url: str, data, *, retrieved_at: str | None = None,
             published_at: str | None = None, period=None, notes: str = "") -> dict:
    """A shared provenance envelope for numeric series and document excerpts."""
    return {"source": source, "url": url, "status": "available", "published_at": published_at,
            "period": period, "retrieved_at": retrieved_at or datetime.now(timezone.utc).isoformat(),
            "data": data, "notes": notes}
