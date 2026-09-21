"""Exercise the production industry adapters without invoking a paid language model.

Example: python scripts/check_industry_access.py --symbols NVDA XOM CAT MSFT --output-dir /tmp/industry-check
Use --date YYYY-MM-DD to verify historical exclusions. Outputs contain public evidence, never API keys or User-Agent contact addresses.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "XOM", "CAT", "MSFT"])
    parser.add_argument("--date", default=datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--related", nargs="*", default=[])
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.industry import IndustryResearch
    from tradingagents.default_config import DEFAULT_CONFIG

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {**DEFAULT_CONFIG, "data_cache_dir": str(output / "cache")}
    set_config(config)
    summaries = []
    for symbol in args.symbols:
        started = time.monotonic()
        research = IndustryResearch(symbol, args.date, config)
        context = research.get_industry_context()
        indicators = research.get_industry_indicators()
        related = research.get_related_company_evidence(args.related) if args.related else {}
        payload = {"context": context, "indicators": indicators, "related": related}
        (output / f"{research.ticker}-{args.date}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        summary = {"ticker": research.ticker, "date": args.date,
                   "primary_evidence": context["primary_evidence_available"],
                   "families": indicators["families"],
                   "evidence_sources": sorted({row["source"] for row in context["evidence"] + indicators["evidence"]}),
                   "disclosures": sum(isinstance(row["data"], dict) and "excerpts" in row["data"] for row in context["evidence"]),
                   "indicator_datasets": len(indicators["evidence"]),
                   "failures": context["failures"] + indicators["failures"],
                   "related_checked": related.get("checked_symbols", []),
                   "related_rejected": related.get("rejected", []),
                   "public_document_requests": research.client.requests, "cache_hits": research.client.cache_hits,
                   "elapsed_seconds": round(time.monotonic() - started, 2)}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(row["primary_evidence"] for row in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
