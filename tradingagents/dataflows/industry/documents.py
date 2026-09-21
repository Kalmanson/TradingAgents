"""Extract bounded, locatable disclosure evidence from SEC and official IR pages."""

from __future__ import annotations

import io
import json
import re
from datetime import date, timedelta
from urllib.parse import urldefrag, urljoin, urlsplit

from parsel import Selector

from .transport import DocumentClient, SourceUnavailable, evidence

TOPICS = {
    "business": r"segment|data center|end.market|business overview|our business|principal product",
    "supply_chain": r"suppl(?:ier|y)|manufactur|foundr|raw material|capacity|procurement",
    "customers": r"customer|concentration|distributor|backlog|demand",
    "competition": r"competit|pricing|substitut|barrier|market share",
    "cycle_risks": r"inventor|capital expend|cyclic|risk factor|export|tariff|outlook",
}
FOCUS_TERMS = {
    "business": r"reportable.{0,20}segment|we (?:operate|design|develop|manufacture)|our (?:business|products|segments)|revenue.{0,20}(?:segment|geograph)",
    "supply_chain": r"foundr|manufacturing (?:partners|capacity)|suppl(?:ier|y).{0,25}(?:relationship|concentrat|depend)|single.source|sole.source|third.party|we (?:purchase|utilize|engage)|contract manufactur",
    "customers": r"customer.{0,30}(?:account|percent|concentrat|depend)|end.market|backlog|purchase commitment|direct customer|distributor",
    "competition": r"competitors? (?:include|are)|we compete|competitive (?:advantage|position|factor)|barrier|market share|pricing power|\[Section heading: Competition",
    "cycle_risks": r"inventor.{0,30}(?:growth|declin|excess|increas)|capital expend|export control|demand.{0,30}(?:growth|declin|increas)|tariff|cyclic|supply.{0,15}constraint",
}
ANNUAL_FORMS = {"10-K", "20-F", "40-F"}
IR_SEEDS = {
    "NVDA": "https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx",
    "MSFT": "https://www.microsoft.com/en-us/Investor/earnings/FY-2026-Q4/press-release-webcast",
}


def extract_document(document: dict, *, budget: int = 18000) -> dict:
    """Select topic-balanced paragraphs/tables, with source anchors or PDF page numbers."""
    body = document["body"]
    sections = []
    if body.startswith(b"%PDF"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted:
            raise SourceUnavailable("Encrypted PDF cannot be used as disclosure evidence")
        for number, page in enumerate(reader.pages[:250], 1):
            text = page.extract_text() or ""
            for index, paragraph in enumerate(re.split(r"\n\s*\n|(?<=\.)\n", text)):
                if len(paragraph.strip()) >= 70:
                    sections.append((f"page {number}, paragraph {index + 1}", paragraph))
    else:
        html = body.decode("utf-8", errors="replace")
        selector = Selector(text=html)
        for node in selector.xpath("//script|//style|//nav|//header|//footer|//*[local-name()='header']"):
            node.drop()
        # Financial tables remain whole; inline XBRL spans are preserved as text.
        nodes = selector.xpath("//h1|//h2|//h3|//table[not(ancestor::table)]|//p[not(ancestor::table)]|"
                               "//div[not(ancestor::table) and not(descendant::div) and not(descendant::p)]")
        heading = ""
        for index, node in enumerate(nodes):
            text = re.sub(r"\s+", " ", " ".join(node.xpath(".//text()").getall())).strip()
            styled_heading = bool(node.xpath('.//*[contains(@style,"font-weight:700") or contains(@style,"font-weight:bold")]'))
            if (3 <= len(text) < 70 and not text.startswith("•") and (
                styled_heading or re.search(r"competit|manufactur|supply|customers|business|risk|segment", text, re.I)
            )):
                heading = text
            elif heading and len(text) >= 70:
                text = f"[Section heading: {heading}] {text}"
            anchor = node.attrib.get("id")
            sections.append((f"#{anchor}" if anchor else f"block {index + 1}", text))
        if not sections:
            sections = [("document text", " ".join(selector.xpath("//body//text()").getall()))]
    cleaned = []
    seen = set()
    for locator, text in sections:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 70 and text not in seen:
            cleaned.append((locator, text))
            seen.add(text)
    result = {}
    allowance = budget // len(TOPICS)
    for topic, pattern in TOPICS.items():
        selected, remaining = [], allowance
        # The first keyword hits are often cover-page boilerplate. Rank actual
        # relationship/operating paragraphs ahead of generic mentions.
        candidates = sorted(cleaned, key=lambda row: (
            min(len(re.findall(FOCUS_TERMS[topic], row[1], re.I)), 3) * 10
            + (40 if re.match(r"\[Section heading: [^\]]*(?:" + pattern + r")", row[1], re.I) else 0)
            + min(len(re.findall(pattern, row[1], re.I)), 8)
            - (20 if re.search(r"forward.looking statements|safe harbor", row[1], re.I) else 0)
            - (5 if len(row[1]) > 10000 else 0)
        ), reverse=True)
        for locator, text in candidates:
            if not re.search(pattern, text, re.I):
                continue
            excerpt = text[:min(remaining, 2400)]
            if len(excerpt) < 70:
                break
            selected.append({"locator": locator, "text": excerpt, "truncated": len(excerpt) < len(text)})
            remaining -= len(excerpt)
            if remaining < 70:
                break
        result[topic] = selected
    if not any(result.values()):
        raise SourceUnavailable("No relevant disclosure paragraphs found")
    return result


def sec_disclosures(client: DocumentClient, cik: str, cutoff: str, *, include_facts=True,
                    expected_symbol: str | None = None) -> dict:
    """Fetch only reports already filed at cutoff, including older submissions shards."""
    if not re.fullmatch(r"[0-9]{1,10}", str(cik)) or int(cik) == 0:
        raise SourceUnavailable("SEC CIK is missing or invalid")
    cik = str(int(cik)).zfill(10)
    index_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    doc = client.fetch(index_url, source="sec", ttl=3600)
    index = json.loads(doc["body"])
    if str(index.get("cik", "")).zfill(10) != cik:
        raise SourceUnavailable("SEC returned a different issuer CIK")
    tickers = {str(ticker).upper().replace(".", "-") for ticker in index.get("tickers", [])}
    if expected_symbol and tickers and expected_symbol.upper().replace(".", "-") not in tickers:
        raise SourceUnavailable("SEC issuer tickers do not match the requested stock")
    rows = []
    files = index.get("filings", {}).get("files", [])
    batches = [index.get("filings", {}).get("recent", {})]
    for shard in sorted(files, key=lambda row: row.get("filingTo", ""), reverse=True):
        if shard.get("filingFrom", "9999") > cutoff:
            continue
        # Recent submissions usually suffice. Follow older ranges only if needed.
        recent_annual = any(form in ANNUAL_FORMS and stamp <= cutoff for batch in batches
                            for form, stamp in zip(batch.get("form", []), batch.get("filingDate", []), strict=False))
        if recent_annual:
            break
        if len(batches) >= 6:
            break
        name = shard.get("name", "")
        if not re.fullmatch(r"CIK[0-9]+-submissions-[0-9]+\.json", name):
            continue
        older = client.fetch(f"https://data.sec.gov/submissions/{name}", source="sec", ttl=3600)
        batches.append(json.loads(older["body"]))
    for batch in batches:
        for i, form in enumerate(batch.get("form", [])):
            stamp = batch["filingDate"][i]
            if stamp > cutoff or form not in ANNUAL_FORMS | {"10-Q"}:
                continue
            rows.append({key: batch.get(key, [""] * len(batch["form"]))[i]
                         for key in ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument")})
    rows.sort(key=lambda row: (row["filingDate"], row["accessionNumber"]), reverse=True)
    annual = next((row for row in rows if row["form"] in ANNUAL_FORMS), None)
    quarterly = next((row for row in rows if row["form"] == "10-Q"
                      and (annual is None or row["reportDate"] > annual["reportDate"])), None)
    records, failures = [], []
    for row in (annual, quarterly):
        if not row:
            continue
        accession, primary = row["accessionNumber"], row["primaryDocument"]
        if (not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", primary)):
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{primary}"
        try:
            report = client.fetch(url, source="sec", ttl=None)
            extracted = extract_document(report)
            records.append(evidence("sec", url, {"form": row["form"], "excerpts": extracted},
                                    retrieved_at=report["retrieved_at"], published_at=row["filingDate"],
                                    period=row["reportDate"], notes="Issuer disclosure; excerpts are not the entire filing."))
        except (SourceUnavailable, ValueError, KeyError) as exc:
            failures.append({"source": "sec", "url": url, "status": "unavailable", "reason": type(exc).__name__})
    if include_facts and records and "sec" not in client.blocked:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            facts_doc = client.fetch(url, source="sec", ttl=3600)
            facts = json.loads(facts_doc["body"])
            if str(facts.get("cik", "")).zfill(10) != cik:
                raise SourceUnavailable("Companyfacts CIK mismatch")
            metrics = []
            earliest_period = (date.fromisoformat(cutoff) - timedelta(days=1096)).isoformat()
            wanted = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "InventoryNet",
                      "CostOfRevenue", "CostOfGoodsAndServicesSold", "GrossProfit",
                      "PaymentsToAcquirePropertyPlantAndEquipment"}
            for taxonomy in ("us-gaap", "ifrs-full"):
                for name, fact in facts.get("facts", {}).get(taxonomy, {}).items():
                    if name not in wanted:
                        continue
                    for unit, values in fact.get("units", {}).items():
                        by_period = {}
                        for value in values:
                            if (not value.get("filed") or value["filed"] > cutoff
                                    or not value.get("end") or not earliest_period <= value["end"] <= cutoff):
                                continue
                            period_key = (value.get("start"), value["end"])
                            if value["filed"] >= by_period.get(period_key, {}).get("filed", ""):
                                by_period[period_key] = value
                        recent = sorted(by_period.values(), key=lambda row: (row["end"], row["filed"]), reverse=True)[:8]
                        if recent:
                            metrics.append({"metric": name, "taxonomy": taxonomy, "unit": unit, "observations": recent})
            if metrics:
                records.append(evidence("sec", url, metrics, retrieved_at=facts_doc["retrieved_at"],
                                        period=f"filed <= {cutoff}",
                                        notes="Entity-wide facts; units and period lengths retained. Do not sum annual and quarterly values."))
        except (SourceUnavailable, ValueError, KeyError) as exc:
            failures.append({"source": "sec", "url": url, "status": "unavailable", "reason": type(exc).__name__})
    return {"issuer": index.get("name"), "cik": cik, "sic": index.get("sic"),
            "sic_description": index.get("sicDescription"), "website": index.get("website"),
            "investor_website": index.get("investorWebsite"), "evidence": records, "failures": failures}


def official_company_documents(client: DocumentClient, symbol: str, website: str | None,
                               investor_website: str | None, cutoff: str, *, live: bool) -> list[dict]:
    """Bounded official-link discovery. Current webpages are excluded from historical runs."""
    if not live:
        raise SourceUnavailable("Current IR pages have no verified historical version; use dated SEC filings")
    seeds = [url for url in (investor_website, IR_SEEDS.get(symbol), website) if url]
    if symbol == "MSFT":
        seeds.insert(0, f"https://www.microsoft.com/investor/reports/ar{str(date.fromisoformat(cutoff).year)[-2:]}/index.html")
        seeds.insert(1, f"https://www.microsoft.com/investor/reports/ar{str(date.fromisoformat(cutoff).year - 1)[-2:]}/index.html")
    queue = [(url, 0, "") for url in dict.fromkeys(seeds) if url.startswith("https://")]
    visited, result = set(), []
    while queue and len(visited) < 7 and len(result) < 2:
        url, depth, link_text = queue.pop(0)
        url = urldefrag(url)[0]
        if url in visited:
            continue
        visited.add(url)
        try:
            doc = client.fetch(url, source="company_ir")
            is_pdf = doc["body"].startswith(b"%PDF")
            selector = Selector(text=doc["body"].decode("utf-8", errors="replace")) if not is_pdf else None
            title = (selector.xpath("string(//title)").get() or "") if selector else link_text
            qualifies = bool(re.search(r"annual.report|financial.results|quarter.*results|earnings.release", f"{title} {url} {link_text}", re.I))
            if qualifies:
                published = None
                if selector:
                    raw_date = selector.css('meta[property="article:published_time"]::attr(content)').get()
                    raw_date = raw_date or selector.css('meta[name="date"]::attr(content), time::attr(datetime)').get()
                    if raw_date and re.match(r"\d{4}-\d{2}-\d{2}", raw_date):
                        published = raw_date[:10]
                if published and published > cutoff:
                    continue
                try:
                    excerpts = extract_document(doc, budget=10000)
                    result.append(evidence("company_ir", url, {"title": title, "excerpts": excerpts},
                                           retrieved_at=doc["retrieved_at"], published_at=published,
                                           notes="Official current webpage/PDF; publication date unknown unless supplied. Not a historical vintage."))
                except (ValueError, SourceUnavailable):
                    pass
            if selector is not None and depth < 2:
                links = []
                for anchor in selector.css("a[href]"):
                    href = urldefrag(urljoin(url, anchor.attrib["href"]))[0]
                    label = " ".join(anchor.xpath(".//text()").getall())
                    if re.search(r"annual.report|financial.results|earnings|investor|\.pdf", f"{label} {href}", re.I):
                        host = urlsplit(href).hostname or ""
                        # Cross-host resources must have been linked from an official page.
                        if href.startswith("https://") and not re.search(r"login|signin|subscribe", href, re.I):
                            score = 0 if re.search(r"annual.report|financial.results|\.pdf", f"{label} {href}", re.I) else 1
                            links.append((score, href, label, host))
                links.sort(key=lambda row: row[0])
                queue = [(href, depth + 1, label) for _, href, label, _ in links[:6]] + queue
        except (SourceUnavailable, ValueError, OSError):
            continue
    if not result:
        raise SourceUnavailable("Official IR documents not discovered or not parseable within the request budget")
    return result
