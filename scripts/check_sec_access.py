"""Check public EDGAR access without an API key.

Configure SEC_USER_AGENT="TradingAgents contact@your-domain.com" in .env,
using an actual contact address you want sent to SEC. Then run:

    python scripts/check_sec_access.py --cik 1045810 --output-dir /tmp/sec-check

Alternatively, --use-support-contact explicitly uses COMMERCE_SUPPORT_EMAIL
as the contact address. It does not send email or change configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


def check_sec_access() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cik", default="1045810", help="Company CIK, up to 10 digits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-support-contact", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{1,10}", args.cik) or int(args.cik) == 0:
        parser.error("CIK must be a positive number of at most 10 digits")

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if args.use_support_contact:
        contact = os.getenv("COMMERCE_SUPPORT_EMAIL", "").strip()
        if not re.fullmatch(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", contact):
            parser.error("COMMERCE_SUPPORT_EMAIL must contain a contact email")
        user_agent = f"TradingAgents research {contact}"
    if (not user_agent.isascii() or "\r" in user_agent or "\n" in user_agent
            or not re.search(r"\S+@\S+\.\S+", user_agent)):
        parser.error("Set SEC_USER_AGENT to an application name and actual contact email")

    cik = str(int(args.cik)).zfill(10)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_to_make = [
        ("submissions", f"https://data.sec.gov/submissions/CIK{cik}.json", "json"),
        ("companyfacts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", "json"),
    ]
    results = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    })
    try:
        for name, url, kind in requests_to_make:
            if results:
                time.sleep(1)
            result = {"name": name, "url": url,
                      "checked_at": datetime.now(timezone.utc).isoformat()}
            results.append(result)
            try:
                # Do not forward the declared contact to another host via redirects.
                response = session.get(url, timeout=(10, 30), allow_redirects=False)
                result.update(status=response.status_code, bytes=len(response.content),
                              content_type=response.headers.get("Content-Type", ""))
                if response.status_code != 200:
                    result["result"] = "access_denied" if response.status_code == 403 else (
                        "rate_limited" if response.status_code == 429 else "http_error"
                    )
                    print(json.dumps(result), flush=True)
                    print("Stopped. Check declared contact, SEC fair-access policy and network; "
                          "no automatic retries or identity changes were attempted.")
                    return 1
                if kind == "json":
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise ValueError("Expected an EDGAR JSON object")
                    if name == "submissions":
                        if str(payload.get("cik", "")).zfill(10) != cik:
                            raise ValueError("Response CIK does not match the requested company")
                        recent = payload["filings"]["recent"]
                        result.update(company=payload.get("name"), filings=len(recent["form"]))
                        for i, form in enumerate(recent["form"]):
                            if form not in {"10-K", "10-Q", "20-F", "40-F"}:
                                continue
                            accession = recent["accessionNumber"][i]
                            document = recent["primaryDocument"][i]
                            if (not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession)
                                    or not re.fullmatch(r"[A-Za-z0-9_.-]+", document)):
                                raise ValueError("Unexpected filing path in SEC response")
                            archive_url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                                           f"{accession.replace('-', '')}/{document}")
                            requests_to_make.append(("filing", archive_url, "html"))
                            result.update(filing_form=form, filing_date=recent["filingDate"][i])
                            break
                        else:
                            raise ValueError("No recent annual or quarterly report in the response")
                    else:
                        if not payload.get("facts"):
                            raise ValueError("Company facts are missing")
                        result.update(company=payload.get("entityName"),
                                      taxonomies=list(payload["facts"]))
                    (output_dir / f"{name}.json").write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
                else:
                    if "html" not in result["content_type"].lower():
                        raise ValueError("Expected an HTML filing")
                    if len(response.content) < 10_000:
                        raise ValueError("Filing response is unexpectedly small; inspect it manually")
                    (output_dir / "filing.html").write_bytes(response.content)
                result["result"] = "ok"
                print(json.dumps(result), flush=True)
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                # Avoid printing request headers or exception objects carrying identity metadata.
                result.update(result="request_or_validation_failed", error_type=type(exc).__name__)
                print(json.dumps(result), flush=True)
                return 1
        print("Verified SEC submissions, company facts and the latest filing document.")
        return 0
    finally:
        session.close()
        (output_dir / "access-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    raise SystemExit(check_sec_access())
