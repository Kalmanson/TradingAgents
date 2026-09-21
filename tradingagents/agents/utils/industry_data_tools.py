"""Per-graph industry tools: state-bound primary ticker, bounded related-company budget."""

from __future__ import annotations

import json
import threading
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from tradingagents.dataflows.industry import IndustryResearch
from tradingagents.dataflows.industry.transport import SourceUnavailable


def build_industry_tools(config: dict) -> list:
    research = None
    run_key = None
    lock = threading.RLock()

    def execute(operation: str, ticker: str, curr_date: str, state: dict, related_symbols=None) -> str:
        nonlocal research, run_key
        with lock:
            try:
                if state.get("asset_type", "stock") != "stock":
                    raise SourceUnavailable("Industry analyst supports stocks only; ETFs and crypto are excluded")
                if (ticker != state["company_of_interest"] or curr_date != state["trade_date"]):
                    raise SourceUnavailable("Use the exact primary ticker and analysis date from the run state")
                key = (state.get("industry_run_id"), ticker, curr_date)
                if research is None or key != run_key:
                    research = IndustryResearch(ticker, curr_date, config)
                    run_key = key
                # Restore budgets and access denials when a new process resumes the tool loop.
                for message in state.get("messages", []):
                    if not isinstance(message, ToolMessage) or message.name not in {
                        "get_industry_context", "get_industry_indicators", "get_related_company_evidence",
                    }:
                        continue
                    try:
                        prior = json.loads(message.content)
                        research.related_used.update(prior.get("checked_symbols", []))
                        research.client.blocked.update(prior.get("blocked_sources", []))
                        research.fmp_unavailable_endpoints.update(
                            tuple(endpoint) for endpoint in prior.get("unavailable_fmp_endpoints", [])
                        )
                    except (ValueError, TypeError, AttributeError):
                        continue
                if operation == "get_related_company_evidence":
                    result = research.get_related_company_evidence(related_symbols)
                else:
                    result = getattr(research, operation)()
                return research.render(result)
            except Exception as exc:
                return json.dumps({"status": "unavailable", "primary_evidence_available": False,
                                   "reason": str(exc) if isinstance(exc, SourceUnavailable) else type(exc).__name__})

    @tool
    def get_industry_context(ticker: str, curr_date: str, state: Annotated[dict, InjectedState]) -> str:
        """Get the primary stock's SEC/official disclosures, revenue exposures and candidate peers, as of YYYY-MM-DD."""
        return execute("get_industry_context", ticker, curr_date, state)

    @tool
    def get_industry_indicators(ticker: str, curr_date: str, state: Annotated[dict, InjectedState]) -> str:
        """Get explicitly mapped FRED/Census/EIA/WSTS statistics for the primary stock's industry; no generic proxy guessing."""
        return execute("get_industry_indicators", ticker, curr_date, state)

    @tool
    def get_related_company_evidence(ticker: str, related_symbols: list[str], curr_date: str,
                                     state: Annotated[dict, InjectedState]) -> str:
        """Cross-check up to three named suppliers/customers or candidate peers, one hop only. ticker remains the PRIMARY stock."""
        return execute("get_related_company_evidence", ticker, curr_date, state, related_symbols)

    return [get_industry_context, get_industry_indicators, get_related_company_evidence]
