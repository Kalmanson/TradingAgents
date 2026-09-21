"""Industry evidence service. Sources are combined explicitly, never silently substituted."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from tradingagents.dataflows import fmp
from tradingagents.dataflows.errors import VendorError

from .documents import official_company_documents, sec_disclosures
from .indicators import collect_indicators, select_industry
from .transport import DocumentClient, SourceUnavailable, evidence

logger = logging.getLogger(__name__)
INDUSTRY_SOURCES = ("sec", "fmp", "fred", "company_ir", "census", "eia", "wsts")


def source_failure(source: str, exc: Exception, *, dataset: str) -> dict:
    """Expose actionable local errors without forwarding arbitrary upstream payloads."""
    safe_fmp_errors = {
        "FMP_API_KEY is not configured": "missing_api_key",
        "Install the project's pinned fmpsdk dependency": "missing_dependency",
        "FMP authentication failed (HTTP 401)": "authentication_failed",
        "FMP endpoint is not included in the current subscription (HTTP 402)": "subscription_required",
        "FMP access denied (HTTP 403); check endpoint permissions or access restrictions": "access_denied",
        "FMP authentication or plan permission denied": "authentication_or_permission_denied",
        "FMP request quota exhausted": "quota_exhausted",
        "FMP request rate limited": "rate_limited",
        "FMP temporarily unavailable or request timed out": "temporarily_unavailable",
        "FMP transport or response parsing failure": "transport_or_parse_error",
        "FMP rejected the request or returned an invalid response": "invalid_response",
    }
    message = str(exc)
    code = safe_fmp_errors.get(message) if source == "fmp" and isinstance(exc, VendorError) else None
    reason = message if code or isinstance(exc, SourceUnavailable) else type(exc).__name__
    return {"source": source, "dataset": dataset, "status": "unavailable",
            "code": code or "source_unavailable", "reason": reason}


class UnsupportedIndustryAsset(SourceUnavailable):
    """A positively identified fund must not trigger company-disclosure collection."""


class IndustryResearch:
    """One research run, one primary ticker, at most three non-recursive related issuers."""

    def __init__(self, ticker: str, curr_date: str, config: dict):
        self.ticker = fmp.normalize_symbol(ticker)
        self.cutoff = date.fromisoformat(curr_date).isoformat()
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if self.cutoff > datetime.now().date().isoformat():
            raise ValueError("Industry analysis date cannot be in the future")
        self.live = self.cutoff >= today
        configured = config.get("industry_sources", INDUSTRY_SOURCES)
        if not isinstance(configured, (list, tuple)) or set(configured) - set(INDUSTRY_SOURCES):
            raise ValueError("industry_sources must contain only supported source names")
        self.sources = set(configured)
        limit = config.get("industry_max_related_companies", 3)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 3:
            raise ValueError("industry_max_related_companies must be an integer from 0 to 3")
        self.max_related = limit
        self.client = DocumentClient(config)
        self.profiles: dict[str, dict] = {}
        self.sec_identities: dict[str, dict] = {}
        self.related_used: set[str] = set()
        self.related_results: dict[str, dict] = {}
        self.context: dict | None = None
        self.indicators: dict | None = None
        self.fmp_unavailable_endpoints: set[tuple[str, str]] = set()

    def _fmp_rows(self, group: str, method: str, **params) -> list[dict]:
        if "fmp" not in self.sources or (group, method) in self.fmp_unavailable_endpoints:
            raise SourceUnavailable("FMP dataset disabled or unavailable for this research run")
        try:
            return fmp._request(group, method, **params)
        except Exception:
            # Entitlements differ by endpoint. A denied segmentation request must
            # not disable profile, geographic revenue or peer discovery.
            self.fmp_unavailable_endpoints.add((group, method))
            raise

    def _profile(self, symbol: str) -> dict:
        if symbol not in self.profiles:
            rows = self._fmp_rows("company", "profile", symbol=symbol)
            matches = [row for row in rows if str(row.get("symbol", "")).upper() == symbol]
            if len(matches) != 1:
                raise SourceUnavailable("FMP returned no unique company profile")
            self.profiles[symbol] = matches[0]
        row = self.profiles[symbol]
        if row.get("isEtf") is True or row.get("isFund") is True:
            raise UnsupportedIndustryAsset("Industry research does not cover ETFs or funds")
        return self.profiles[symbol]

    def _sec_identity(self, symbol: str) -> dict:
        if "sec" not in self.sources:
            raise SourceUnavailable("SEC identity source is disabled")
        if symbol not in self.sec_identities:
            mapping = self.client.fetch("https://www.sec.gov/files/company_tickers.json", source="sec", ttl=3600)
            rows = json.loads(mapping["body"])
            matches = [row for row in rows.values()
                       if str(row.get("ticker", "")).upper().replace(".", "-") == symbol.replace(".", "-")]
            if len(matches) != 1:
                raise SourceUnavailable("No unique SEC ticker-to-CIK mapping")
            self.sec_identities[symbol] = {"symbol": symbol, "cik": str(matches[0]["cik_str"]),
                                           "companyName": matches[0].get("title", "")}
        return self.sec_identities[symbol]

    def _company(self, symbol: str, *, related: bool = False) -> dict:
        records, failures, profile, sec = [], [], {}, {}
        if "fmp" in self.sources:
            try:
                profile = self._profile(symbol)
            except UnsupportedIndustryAsset:
                raise
            except Exception as exc:
                failures.append(source_failure("fmp", exc, dataset="profile"))
        if "sec" in self.sources:
            try:
                cik = profile.get("cik")
                if not cik:
                    cik = self._sec_identity(symbol)["cik"]
                sec = sec_disclosures(self.client, str(cik), self.cutoff, include_facts=not related,
                                      expected_symbol=symbol)
                records.extend(sec["evidence"])
                failures.extend(sec["failures"])
            except Exception as exc:
                failures.append({"source": "sec", "status": "unavailable", "reason": str(exc) if isinstance(exc, SourceUnavailable) else type(exc).__name__})
        if self.live and profile:
            fields = {key: profile.get(key) for key in ("symbol", "companyName", "sector", "industry", "description", "website", "cik")}
            records.append(evidence("fmp", f"https://financialmodelingprep.com/stable/profile?symbol={symbol}", fields,
                                    notes="Current company profile, not historical classification or relationship evidence."))
        if "fmp" in self.sources and not related:
            for method, endpoint in (("revenue_product_segmentation", "revenue-product-segmentation"),
                                     ("revenue_geographic_segmentation", "revenue-geographic-segmentation")):
                try:
                    if not self.live:
                        raise SourceUnavailable("FMP segmentation snapshots have no verified historical vintage")
                    rows = self._fmp_rows("statements", method, symbol=symbol)
                    rows = [row for row in rows if row.get("symbol") == symbol and row.get("date", "9999") <= self.cutoff]
                    rows.sort(key=lambda row: row["date"], reverse=True)
                    if not rows:
                        raise SourceUnavailable("No dated revenue segmentation")
                    records.append(evidence("fmp", f"https://financialmodelingprep.com/stable/{endpoint}?symbol={symbol}", rows[:3],
                                            period=[row.get("date") for row in rows[:3]],
                                            notes="Fiscal period end is not filing date. Provider segment definitions and currency retained; categories may overlap."))
                except Exception as exc:
                    failures.append(source_failure("fmp", exc, dataset=endpoint))
        if "company_ir" in self.sources:
            try:
                records.extend(official_company_documents(self.client, symbol,
                               profile.get("website") or sec.get("website"), sec.get("investor_website"),
                               self.cutoff, live=self.live))
            except Exception as exc:
                failures.append({"source": "company_ir", "status": "unavailable", "reason": str(exc) if isinstance(exc, SourceUnavailable) else type(exc).__name__})
        classification = " / ".join(str(profile.get(key) or "") for key in ("sector", "industry")) if self.live else ""
        if not classification.replace("/", "").strip():
            # SIC is only a routing hint, confirmed against dated filing text.
            # It must never establish a historical business fact by itself.
            business = " ".join(entry["text"] for item in records if item["source"] == "sec"
                                and isinstance(item["data"], dict)
                                for excerpts in item["data"].get("excerpts", {}).values() for entry in excerpts)
            sic_description = str(sec.get("sic_description") or "")
            confirmed = set(select_industry(sic_description)) & set(select_industry(business))
            classification = sic_description if confirmed else business[:18000]
        primary = any(isinstance(row["data"], dict) and row["data"].get("excerpts") for row in records)
        return {"ticker": symbol, "company": profile.get("companyName") or sec.get("issuer"),
                "classification": classification, "sic": sec.get("sic"), "cik": sec.get("cik"),
                "classification_note": "Routing aid; historical business conclusions must cite dated issuer disclosures.",
                "primary_evidence_available": primary, "evidence": records, "failures": failures}

    def get_industry_context(self) -> dict:
        if self.context is None:
            self.context = self._company(self.ticker)
            peers = []
            if "fmp" in self.sources and self.live:
                try:
                    rows = self._fmp_rows("company", "stock_peers", symbol=self.ticker)
                    peers = [{"symbol": row.get("symbol"), "company": row.get("companyName")}
                             for row in rows[:12] if row.get("symbol") != self.ticker]
                except Exception as exc:
                    self.context["failures"].append(source_failure("fmp", exc, dataset="peers"))
            self.context.update(candidate_peers=peers, peer_source_url=f"https://financialmodelingprep.com/stable/stock-peers?symbol={self.ticker}",
                                peer_note="Current provider candidates, not verified competitors, customers or suppliers.",
                                analysis_date=self.cutoff, max_related_companies=self.max_related)
        return self.context

    def get_industry_indicators(self) -> dict:
        if self.indicators is None:
            context = self.get_industry_context()
            self.indicators = collect_indicators(self.client, context["classification"], self.cutoff, self.sources, live=self.live)
            self.indicators.update(ticker=self.ticker, analysis_date=self.cutoff)
        return self.indicators

    def get_related_company_evidence(self, related_symbols: list[str]) -> dict:
        if not isinstance(related_symbols, list) or len(related_symbols) > self.max_related:
            raise SourceUnavailable(f"Request at most {self.max_related} related companies in total")
        context = self.get_industry_context()
        disclosure = " ".join(entry["text"] for item in context["evidence"] if isinstance(item["data"], dict)
                              for excerpts in item["data"].get("excerpts", {}).values() for entry in excerpts)
        peers = {row["symbol"] for row in context["candidate_peers"]}
        results, rejected = [], []
        for raw in dict.fromkeys(related_symbols):
            try:
                symbol = fmp.normalize_symbol(raw)
                if symbol == self.ticker:
                    raise SourceUnavailable("Related company must differ from the primary ticker")
                if symbol in self.related_results:
                    results.append(self.related_results[symbol])
                    continue
                if symbol not in self.related_used and len(self.related_used) >= self.max_related:
                    raise SourceUnavailable("Per-run related-company limit reached")
                # Count attempts too, to bound lookup cost and prevent repeated speculative requests.
                self.related_used.add(symbol)
                try:
                    profile = self._profile(symbol)
                except UnsupportedIndustryAsset:
                    raise
                except Exception:
                    profile = self._sec_identity(symbol)
                words = re.findall(r"[a-z]+", str(profile.get("companyName", "")).lower())
                words = [word for word in words if word not in {"inc", "corporation", "corp", "limited", "ltd", "company", "co", "plc", "the", "group", "holdings"}]
                mentioned = bool(words) and " ".join(words[:3]) in re.sub(r"[^a-z]+", " ", disclosure.lower())
                # Some disclosures name a company by its trading abbreviation.
                # Require an explicit relationship term nearby, not a bare match.
                if not mentioned and len(symbol) >= 3:
                    for match in re.finditer(r"\b" + re.escape(symbol) + r"\b", disclosure):
                        nearby = disclosure[max(0, match.start() - 200):match.end() + 200]
                        if re.search(r"competit|suppl|customer|manufactur|such as", nearby, re.I):
                            mentioned = True
                            break
                if symbol not in peers and not mentioned:
                    raise SourceUnavailable("No primary-disclosure mention or provider peer-candidate basis")
                company = self._company(symbol, related=True)
                company["selection_basis"] = "named in primary disclosure" if mentioned else "unverified provider peer candidate"
                company["relationship_note"] = "Verify the relationship from cited disclosures; candidate status never establishes a supply contract. One hop only."
                self.related_results[symbol] = company
                results.append(company)
            except Exception as exc:
                rejected.append({"symbol": raw, "reason": str(exc) if isinstance(exc, SourceUnavailable) else type(exc).__name__})
        return {"ticker": self.ticker, "analysis_date": self.cutoff, "companies": results,
                "checked_symbols": sorted(self.related_used), "rejected": rejected}

    def render(self, data: dict) -> str:
        """Return valid bounded JSON, keeping provenance even when long excerpts are shortened."""
        output = {**data, "blocked_sources": sorted(self.client.blocked),
                  "unavailable_fmp_endpoints": sorted(self.fmp_unavailable_endpoints),
                  "transport_stats": {"requests": self.client.requests, "cache_hits": self.client.cache_hits}}
        # Copy before compacting so relationship checks still use the fuller cached evidence.
        output = json.loads(json.dumps(output, ensure_ascii=False, default=str))
        for cap in (2000, 1000, 500, 250):
            rendered = json.dumps(output, ensure_ascii=False)
            if len(rendered) <= 65000:
                return rendered
            pending = [output]
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (list, dict)):
                            pending.append(value)
                        elif isinstance(value, str) and key not in {"url", "source", "locator"} and len(value) > cap:
                            item[key] = value[:cap] + " [excerpt shortened]"
                elif isinstance(item, list):
                    pending.extend(value for value in item if isinstance(value, (list, dict)))
        return json.dumps(output, ensure_ascii=False)
