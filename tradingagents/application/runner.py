"""Reusable streaming analysis runner for terminal and web frontends."""

from __future__ import annotations

import ast
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.application.stats import StatsCallbackHandler
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    applicable_analysts,
    build_analyst_execution_plan,
    sync_analyst_tracker_from_chunk,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree

ANALYST_ORDER = ("market", "social", "news", "fundamentals", "industry")
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "industry": "Industry and Supply Chain Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
    "industry": "industry_report",
}
FIXED_AGENTS = (
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Neutral Analyst",
    "Conservative Analyst",
    "Portfolio Manager",
)
CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")
_TICKER_RE = re.compile(r"^[A-Za-z0-9._^=-]{1,32}$")

logger = logging.getLogger(__name__)


def normalize_ticker(value: str, config: dict | None = None) -> str:
    """Normalize a user ticker through the data layer's canonical resolver."""
    from tradingagents.dataflows.symbol_utils import normalize_input_symbol

    return normalize_input_symbol(value.strip(), config)


def infer_asset_type(ticker: str) -> str:
    """Return the graph asset type for a normalized ticker."""
    return "crypto" if normalize_ticker(ticker).upper().endswith(CRYPTO_SUFFIXES) else "stock"


@dataclass(frozen=True)
class AnalysisRequest:
    ticker: str
    analysis_date: str
    analysts: tuple[str, ...]
    research_depth: int
    llm_provider: str
    quick_think_llm: str
    deep_think_llm: str
    asset_type: str = "stock"
    output_language: str = "Chinese"
    backend_url: str | None = None
    google_thinking_level: str | None = None
    openai_reasoning_effort: str | None = None
    anthropic_effort: str | None = None
    checkpoint_enabled: bool = True

    def validate(self) -> None:
        raw_ticker = self.ticker.strip()
        if not _TICKER_RE.fullmatch(raw_ticker):
            raise ValueError("Ticker 只能包含字母、数字以及 . _ - ^ =，且长度不能超过 32。")
        try:
            selected_date = date.fromisoformat(self.analysis_date)
        except ValueError as exc:
            raise ValueError("分析日期必须使用 YYYY-MM-DD 格式。") from exc
        if selected_date > date.today():
            raise ValueError("分析日期不能晚于今天。")
        if self.asset_type not in {"stock", "etf", "crypto"}:
            raise ValueError("资产类型必须是 stock、etf 或 crypto。")
        if not self.analysts:
            raise ValueError("至少选择一位分析师。")
        unknown = set(self.analysts) - set(ANALYST_ORDER)
        if unknown:
            raise ValueError(f"未知分析师：{', '.join(sorted(unknown))}")
        if self.asset_type == "crypto" and "fundamentals" in self.analysts:
            raise ValueError("加密资产暂不支持基本面分析师。")
        if self.asset_type != "stock" and "industry" in self.analysts:
            raise ValueError("行业与产业链分析师仅支持普通股票。")
        if self.research_depth not in {1, 3, 5}:
            raise ValueError("研究深度必须是 1、3 或 5。")
        if not self.llm_provider.strip():
            raise ValueError("请选择 LLM Provider。")
        if not self.quick_think_llm.strip() or not self.deep_think_llm.strip():
            raise ValueError("快速模型和深度模型都不能为空。")
        if self.llm_provider.lower() == "openai_compatible" and not self.backend_url:
            raise ValueError("OpenAI Compatible Provider 必须配置 Backend URL。")
        if self.backend_url and not re.match(r"^https?://", self.backend_url):
            raise ValueError("Backend URL 必须以 http:// 或 https:// 开头。")

    def normalized(self, config: dict | None = None) -> AnalysisRequest:
        ticker = normalize_ticker(self.ticker, config)
        asset_type = "crypto" if ticker.upper().endswith(CRYPTO_SUFFIXES) else (
            "etf" if self.asset_type == "etf" else "stock"
        )
        analysts = tuple(key for key in ANALYST_ORDER if key in self.analysts)
        analysts = applicable_analysts(analysts, asset_type)
        return AnalysisRequest(
            **{
                **asdict(self),
                "ticker": ticker,
                "asset_type": asset_type,
                "analysts": analysts,
                "llm_provider": self.llm_provider.lower(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["analysts"] = list(self.analysts)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisRequest:
        values = dict(data)
        values["analysts"] = tuple(values.get("analysts", ()))
        return cls(**values)

    def build_config(self, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = deepcopy(DEFAULT_CONFIG if base_config is None else base_config)
        config.update(
            {
                "max_debate_rounds": self.research_depth,
                "max_risk_discuss_rounds": self.research_depth,
                "quick_think_llm": self.quick_think_llm,
                "deep_think_llm": self.deep_think_llm,
                "backend_url": self.backend_url,
                "llm_provider": self.llm_provider.lower(),
                "google_thinking_level": self.google_thinking_level,
                "openai_reasoning_effort": self.openai_reasoning_effort,
                "anthropic_effort": self.anthropic_effort,
                "output_language": self.output_language,
                "checkpoint_enabled": self.checkpoint_enabled,
            }
        )
        return config


@dataclass
class AnalysisResult:
    final_state: dict[str, Any]
    signal: str
    report_path: Path | None
    stats: dict[str, Any]


@dataclass
class AnalysisEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )
    result: AnalysisResult | None = field(default=None, repr=False)
    error: BaseException | None = field(default=None, repr=False)

    def persisted_payload(self) -> dict[str, Any]:
        if self.result is None:
            return self.payload
        return {
            **self.payload,
            "signal": self.result.signal,
            "report_path": str(self.result.report_path) if self.result.report_path else None,
            "stats": self.result.stats,
        }


class CancellationToken:
    """Cooperative cancellation checked between LangGraph chunks."""

    def __init__(self, checker: Callable[[], bool] | None = None) -> None:
        self._event = threading.Event()
        self._checker = checker

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set() or bool(self._checker and self._checker())


def _extract_content(content: Any) -> str | None:
    def is_empty(value: Any) -> bool:
        if value is None or value == "":
            return True
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return True
            try:
                return not bool(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                return False
        return not bool(value)

    if is_empty(content):
        return None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if not is_empty(text) else None
    if isinstance(content, list):
        parts = [
            item.get("text", "").strip()
            if isinstance(item, dict) and item.get("type") == "text"
            else item.strip() if isinstance(item, str) else ""
            for item in content
        ]
        result = " ".join(part for part in parts if part and not is_empty(part))
        return result or None
    return str(content).strip() or None


def _message_kind(message: Any) -> tuple[str, str | None]:
    content = _extract_content(getattr(message, "content", None))
    if isinstance(message, HumanMessage):
        return ("Control" if content == "Continue" else "User", content)
    if isinstance(message, ToolMessage):
        return "Data", content
    if isinstance(message, AIMessage):
        return "Agent", content
    return "System", content


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for key, value in os.environ.items():
        if ("KEY" in key or "TOKEN" in key or "SECRET" in key) and value:
            message = message.replace(value, "***")
    return message[:2000]


class _ProgressTracker:
    def __init__(self, analysts: tuple[str, ...]) -> None:
        self.analysts = analysts
        self.agent_status = {
            **{ANALYST_AGENT_NAMES[key]: "pending" for key in analysts},
            **dict.fromkeys(FIXED_AGENTS, "pending"),
        }
        self.reports: dict[str, str] = {}
        self.processed_message_ids: set[str] = set()
        self.plan = build_analyst_execution_plan(analysts)
        self.wall_time = AnalystWallTimeTracker(self.plan)
        self.agent_status[ANALYST_AGENT_NAMES[analysts[0]]] = "in_progress"
        self.wall_time.mark_started(analysts[0])

    def _set_report(self, key: str, content: str, events: list[AnalysisEvent]) -> None:
        if content and self.reports.get(key) != content:
            self.reports[key] = content
            events.append(AnalysisEvent("report", {"section": key, "content": content}))

    def _update_analysts(self, chunk: dict[str, Any], events: list[AnalysisEvent]) -> None:
        sync_analyst_tracker_from_chunk(self.wall_time, chunk)
        active_found = False
        for key in self.analysts:
            report_key = ANALYST_REPORT_MAP[key]
            if chunk.get(report_key):
                self._set_report(report_key, chunk[report_key], events)
            if self.reports.get(report_key):
                self.agent_status[ANALYST_AGENT_NAMES[key]] = "completed"
            elif not active_found:
                self.agent_status[ANALYST_AGENT_NAMES[key]] = "in_progress"
                active_found = True
            else:
                self.agent_status[ANALYST_AGENT_NAMES[key]] = "pending"
        if not active_found and self.analysts:
            self.agent_status["Bull Researcher"] = "in_progress"

    def _update_research(self, chunk: dict[str, Any], events: list[AnalysisEvent]) -> None:
        debate = chunk.get("investment_debate_state") or {}
        bull = str(debate.get("bull_history", "")).strip()
        bear = str(debate.get("bear_history", "")).strip()
        judge = str(debate.get("judge_decision", "")).strip()
        if bull or bear:
            for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                self.agent_status[agent] = "in_progress"
        parts = []
        if bull:
            parts.append(f"### Bull Researcher Analysis\n{bull}")
        if bear:
            parts.append(f"### Bear Researcher Analysis\n{bear}")
        if judge:
            parts.append(f"### Research Manager Decision\n{judge}")
            for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                self.agent_status[agent] = "completed"
            self.agent_status["Trader"] = "in_progress"
        if parts:
            self._set_report("investment_plan", "\n\n".join(parts), events)

    def _update_trading_and_risk(self, chunk: dict[str, Any], events: list[AnalysisEvent]) -> None:
        trader_plan = chunk.get("trader_investment_plan")
        if trader_plan:
            self._set_report("trader_investment_plan", str(trader_plan), events)
            self.agent_status["Trader"] = "completed"
            self.agent_status["Aggressive Analyst"] = "in_progress"

        risk = chunk.get("risk_debate_state") or {}
        mappings = (
            ("aggressive_history", "Aggressive Analyst", "Aggressive Analyst Analysis"),
            ("conservative_history", "Conservative Analyst", "Conservative Analyst Analysis"),
            ("neutral_history", "Neutral Analyst", "Neutral Analyst Analysis"),
        )
        parts = []
        for field_name, agent, title in mappings:
            content = str(risk.get(field_name, "")).strip()
            if content:
                self.agent_status[agent] = "in_progress"
                parts.append(f"### {title}\n{content}")
        judge = str(risk.get("judge_decision", "")).strip()
        if judge:
            parts.append(f"### Portfolio Manager Decision\n{judge}")
            for agent in (
                "Aggressive Analyst",
                "Conservative Analyst",
                "Neutral Analyst",
                "Portfolio Manager",
            ):
                self.agent_status[agent] = "completed"
        if parts:
            self._set_report("final_trade_decision", "\n\n".join(parts), events)

    def process(self, chunk: dict[str, Any]) -> list[AnalysisEvent]:
        events: list[AnalysisEvent] = []
        for message in chunk.get("messages", []):
            message_id = getattr(message, "id", None)
            if message_id is not None:
                if message_id in self.processed_message_ids:
                    continue
                self.processed_message_ids.add(message_id)
            kind, content = _message_kind(message)
            if content:
                events.append(AnalysisEvent("message", {"kind": kind, "content": content}))
            for tool_call in getattr(message, "tool_calls", []) or []:
                if isinstance(tool_call, dict):
                    name, args = tool_call.get("name", "unknown"), tool_call.get("args", {})
                else:
                    name, args = tool_call.name, tool_call.args
                events.append(AnalysisEvent("tool_call", {"name": name, "args": args}))

        self._update_analysts(chunk, events)
        self._update_research(chunk, events)
        self._update_trading_and_risk(chunk, events)
        events.append(
            AnalysisEvent(
                "progress",
                {
                    "stage": self.stage(),
                    "agents": dict(self.agent_status),
                    "reports_completed": sum(bool(value) for value in self.reports.values()),
                    "reports_total": len(self.analysts) + 3,
                },
            )
        )
        return events

    def stage(self) -> str:
        if self.agent_status.get("Portfolio Manager") == "completed":
            return "completed"
        if any(
            self.agent_status.get(agent) == "in_progress"
            for agent in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager")
        ):
            return "risk"
        if self.agent_status.get("Trader") == "in_progress":
            return "trading"
        if any(
            self.agent_status.get(agent) == "in_progress"
            for agent in ("Bull Researcher", "Bear Researcher", "Research Manager")
        ):
            return "research"
        return "analysts"


class AnalysisRunner:
    """Execute one request and emit frontend-neutral progress events."""

    def __init__(
        self,
        request: AnalysisRequest,
        *,
        config: dict[str, Any] | None = None,
        artifact_dir: str | Path | None = None,
        graph_factory: Callable[..., TradingAgentsGraph] = TradingAgentsGraph,
    ) -> None:
        normalized = request.normalized(DEFAULT_CONFIG if config is None else config)
        normalized.validate()
        self.request = normalized
        self.config = normalized.build_config(config)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.graph_factory = graph_factory
        self.stats_handler = StatsCallbackHandler()

    def stream(self, cancel_token: CancellationToken | None = None) -> Iterator[AnalysisEvent]:
        token = cancel_token or CancellationToken()
        request = self.request
        started_at = time.monotonic()
        tracker = _ProgressTracker(request.analysts)
        graph: TradingAgentsGraph | None = None

        yield AnalysisEvent(
            "started",
            {
                "ticker": request.ticker,
                "analysis_date": request.analysis_date,
                "asset_type": request.asset_type,
            },
        )
        logger.info(
            "分析开始 event=analysis_started ticker=%s date=%s asset_type=%s provider=%s "
            "quick_llm=%s deep_llm=%s analysts=%s",
            request.ticker, request.analysis_date, request.asset_type, request.llm_provider,
            request.quick_think_llm, request.deep_think_llm, ",".join(request.analysts),
        )
        try:
            graph = self.graph_factory(
                request.analysts,
                config=self.config,
                debug=False,
                callbacks=[self.stats_handler],
            )
            graph.ticker = request.ticker
            graph._resolve_pending_entries(request.ticker)
            past_context = graph.memory_log.get_past_context(
                request.ticker, as_of=graph._memory_as_of(request.analysis_date)
            )
            instrument_context = graph.resolve_instrument_context(
                request.ticker, request.asset_type
            )
            effective_asset = getattr(graph, "resolved_asset_type", request.asset_type)
            effective_analysts = tuple(getattr(graph, "selected_analysts", request.analysts))
            tracker = _ProgressTracker(effective_analysts)
            yield AnalysisEvent("analysts_resolved", {
                "analysts": list(effective_analysts), "asset_type": effective_asset,
            })
            initial_state = graph.propagator.create_initial_state(
                request.ticker,
                request.analysis_date,
                asset_type=getattr(graph, "resolved_asset_type", request.asset_type),
                past_context=past_context,
                instrument_context=instrument_context,
            )
            graph_args = graph.propagator.get_graph_args(callbacks=[self.stats_handler])
            checkpoint_tid = graph.begin_checkpoint(
                request.ticker, request.analysis_date, effective_asset
            )
            if checkpoint_tid is not None:
                graph_args.setdefault("config", {}).setdefault("configurable", {})[
                    "thread_id"
                ] = checkpoint_tid

            final_state: dict[str, Any] = {}
            for chunk in graph.graph.stream(graph.checkpoint_input(initial_state), **graph_args):
                final_state.update(chunk)
                yield from tracker.process(chunk)
                yield AnalysisEvent("stats", self.stats_handler.get_stats())
                if token.is_cancelled():
                    logger.info(
                        "用户取消分析 event=analysis_stopped ticker=%s", request.ticker,
                    )
                    yield AnalysisEvent(
                        "stopped",
                        {"message": "停止请求已生效，可稍后从 checkpoint 恢复。"},
                    )
                    return

            if not final_state.get("final_trade_decision"):
                raise RuntimeError("分析已结束，但没有生成最终交易决策。")

            graph.curr_state = final_state
            graph._log_state(request.analysis_date, final_state)
            graph.memory_log.store_decision(
                ticker=request.ticker,
                trade_date=request.analysis_date,
                final_trade_decision=final_state["final_trade_decision"],
            )
            signal = graph.process_signal(final_state["final_trade_decision"])
            report_path = None
            if self.artifact_dir is not None:
                report_path = write_report_tree(final_state, request.ticker, self.artifact_dir)
            graph.clear_checkpoint_on_success(
                request.ticker, request.analysis_date, effective_asset
            )
            stats = {
                **self.stats_handler.get_stats(),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "analyst_wall_time_summary": tracker.wall_time.format_summary(),
            }
            result = AnalysisResult(final_state, signal, report_path, stats)
            logger.info(
                "分析完成 event=analysis_completed ticker=%s signal=%s elapsed_seconds=%.1f",
                request.ticker, signal, stats["elapsed_seconds"],
            )
            yield AnalysisEvent(
                "completed",
                {"message": "分析完成。"},
                result=result,
            )
        except Exception as exc:
            logger.error(
                "分析失败 event=analysis_failed ticker=%s error_type=%s error=%s",
                request.ticker, type(exc).__name__, _safe_error(exc),
            )
            yield AnalysisEvent(
                "failed",
                {"error": _safe_error(exc)},
                error=exc,
            )
        finally:
            if graph is not None:
                graph.end_checkpoint()
