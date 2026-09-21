"""Chinese local dashboard for queued TradingAgents analysis runs."""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from tradingagents.application.runner import AnalysisRequest, infer_asset_type, normalize_ticker
from tradingagents.application.task_manager import AnalysisTaskManager
from tradingagents.application.task_store import (
    ACTIVE_STATUSES,
    RESUMABLE_STATUSES,
    RUN_STATUSES,
    TaskStore,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_model_options

STATUS_LABELS = {
    "queued": "排队中",
    "running": "运行中",
    "stop_requested": "正在停止",
    "stopped": "已停止",
    "completed": "已完成",
    "failed": "失败",
    "interrupted": "已中断",
    "cancelled": "已取消",
}
AGENT_STATUS_LABELS = {
    "pending": "等待中",
    "in_progress": "进行中",
    "completed": "已完成",
}
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google Gemini",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "qwen": "Qwen 国际",
    "qwen-cn": "Qwen 中国",
    "glm": "GLM 国际",
    "glm-cn": "GLM 中国",
    "minimax": "MiniMax 国际",
    "minimax-cn": "MiniMax 中国",
    "openrouter": "OpenRouter",
    "azure": "Azure OpenAI",
    "bedrock": "Amazon Bedrock",
    "ollama": "Ollama",
    "openai_compatible": "OpenAI Compatible",
    "mistral": "Mistral",
    "kimi": "Kimi",
    "groq": "Groq",
    "nvidia": "NVIDIA NIM",
}
ANALYST_LABELS = {
    "market": "市场分析师",
    "social": "情绪分析师",
    "news": "新闻分析师",
    "fundamentals": "基本面分析师",
    "industry": "行业与产业链分析师",
}
REPORT_LABELS = {
    "market_report": "市场分析",
    "sentiment_report": "市场情绪",
    "news_report": "新闻分析",
    "fundamentals_report": "基本面分析",
    "industry_report": "行业与产业链分析",
    "investment_plan": "多空研究",
    "trader_investment_plan": "交易计划",
    "final_trade_decision": "风险与最终决策",
}
STAGE_LABELS = {
    "analysts": "分析师",
    "research": "多空研究",
    "trading": "交易员",
    "risk": "风险团队",
    "completed": "投资组合经理",
}

st.set_page_config(page_title="TradingAgents 工作台", page_icon="📈", layout="wide")


@st.cache_resource
def get_manager() -> AnalysisTaskManager:
    store = TaskStore()
    return AnalysisTaskManager(store, legacy_reports_dir=Path.cwd() / "reports")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openrouter_models() -> list[tuple[str, str]]:
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json().get("data", [])
        models.sort(key=lambda item: item.get("created") or 0, reverse=True)
        choices = [
            (item.get("name") or item["id"], item["id"])
            for item in models[:10]
            if item.get("id")
        ]
        return choices
    except (requests.RequestException, ValueError, KeyError):
        return []


def model_options(provider: str, mode: str) -> list[tuple[str, str]]:
    if provider == "openrouter":
        return [*fetch_openrouter_models(), ("自定义 Model ID", "custom")]
    try:
        return get_model_options(provider, mode)
    except KeyError:
        return [("自定义 Model ID", "custom")]


def select_model(provider: str, mode: str, label: str, default: str) -> str:
    options = model_options(provider, mode)
    values = [value for _, value in options]
    labels = {value: display for display, value in options}
    if default and default not in values:
        options = [(default, default), *options]
        values.insert(0, default)
        labels[default] = default
    index = values.index(default) if default in values else 0
    choice = st.selectbox(
        label,
        values,
        index=index,
        format_func=lambda value: labels.get(value, value),
        key=f"{provider}-{mode}-choice",
    )
    if choice == "custom":
        return st.text_input(
            f"{label} · 自定义 ID",
            key=f"{provider}-{mode}-custom",
            placeholder="输入服务商提供的 model ID",
        ).strip()
    return choice


def api_key_status(provider: str) -> tuple[str, str]:
    env_name = get_api_key_env(provider)
    if provider == "bedrock":
        return "info", "Bedrock 使用 AWS credential chain，运行时验证。"
    if provider in {"ollama", "openai_compatible"}:
        suffix = f"（可选：{env_name}）" if env_name else ""
        return "success", f"该 Provider 可无密钥运行{suffix}。"
    if env_name and os.environ.get(env_name):
        return "success", f"已检测到 {env_name}。"
    return "error", f"未检测到 {env_name or '对应的 API Key'}，请在 .env 中配置。"


def parse_request(row: dict[str, Any]) -> AnalysisRequest | None:
    if not row.get("request_json"):
        return None
    return AnalysisRequest.from_dict(json.loads(row["request_json"]))


def format_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def zip_report(report_path: Path) -> bytes:
    root = report_path.parent if report_path.is_file() else report_path
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(root))
    return buffer.getvalue()


def latest_event_payload(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    for event in reversed(events):
        if event["event_type"] == event_type:
            return event["payload"]
    return {}


def render_phase(stage: str) -> None:
    stages = ["analysts", "research", "trading", "risk", "completed"]
    current = stages.index(stage) if stage in stages else 0
    columns = st.columns(len(stages))
    for index, stage_name in enumerate(stages):
        icon = "✅" if index < current or stage == "completed" else "🔵" if index == current else "⚪"
        columns[index].markdown(f"{icon} **{STAGE_LABELS[stage_name]}**")


def render_run_detail(manager: AnalysisTaskManager, row: dict[str, Any]) -> None:
    events = manager.store.list_events(row["id"])
    progress = latest_event_payload(events, "progress")
    stats = latest_event_payload(events, "stats")
    request = parse_request(row)

    title_cols = st.columns([4, 1])
    title_cols[0].subheader(f"{row['ticker']} · {row.get('analysis_date') or '日期未知'}")
    title_cols[1].metric("状态", STATUS_LABELS.get(row["status"], row["status"]))
    if row.get("error_summary"):
        st.error(row["error_summary"])
    if row["status"] in {"queued", "running"}:
        button_text = "取消排队" if row["status"] == "queued" else "请求停止"
        if st.button(button_text, key=f"stop-{row['id']}"):
            manager.stop(row["id"])
            st.rerun()
    elif row["status"] == "stop_requested":
        st.warning("正在等待当前 LLM 或工具调用结束，然后停止任务。")

    render_phase(progress.get("stage", "completed" if row["status"] == "completed" else "analysts"))
    metric_cols = st.columns(5)
    metric_cols[0].metric("LLM 调用", stats.get("llm_calls", 0))
    metric_cols[1].metric("工具调用", stats.get("tool_calls", 0))
    metric_cols[2].metric("输入 Token", stats.get("tokens_in", 0))
    metric_cols[3].metric("输出 Token", stats.get("tokens_out", 0))
    metric_cols[4].metric("最终信号", row.get("signal") or "—")

    if request:
        with st.expander("运行参数", expanded=False):
            st.json(
                {
                    "ticker": request.ticker,
                    "analysis_date": request.analysis_date,
                    "analysts": list(request.analysts),
                    "research_depth": request.research_depth,
                    "provider": request.llm_provider,
                    "quick_model": request.quick_think_llm,
                    "deep_model": request.deep_think_llm,
                    "output_language": request.output_language,
                    "checkpoint": request.checkpoint_enabled,
                }
            )

    agents = progress.get("agents", {})
    reports: dict[str, str] = {}
    messages = []
    tools = []
    for event in events:
        if event["event_type"] == "report":
            reports[event["payload"]["section"]] = event["payload"]["content"]
        elif event["event_type"] == "message":
            messages.append(event["payload"])
        elif event["event_type"] == "tool_call":
            tools.append(event["payload"])

    tab_progress, tab_reports, tab_logs = st.tabs(["Agent 进度", "分析报告", "运行日志"])
    with tab_progress:
        if agents:
            st.dataframe(
                [
                    {"Agent": name, "状态": AGENT_STATUS_LABELS.get(status, status)}
                    for name, status in agents.items()
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("任务尚未产生 Agent 状态。")
    with tab_reports:
        if reports:
            report_tabs = st.tabs([REPORT_LABELS.get(key, key) for key in reports])
            for tab, (_, content) in zip(report_tabs, reports.items(), strict=True):
                with tab:
                    st.markdown(content)
        elif row.get("report_path") and Path(row["report_path"]).exists():
            st.markdown(Path(row["report_path"]).read_text(encoding="utf-8"))
        else:
            st.info("报告尚未生成。")
    with tab_logs:
        with st.expander(f"Agent 消息（{len(messages)}）", expanded=False):
            for message in messages[-100:]:
                st.markdown(f"**{message.get('kind', '消息')}** · {message.get('content', '')}")
        with st.expander(f"工具调用（{len(tools)}）", expanded=False):
            for tool in tools[-100:]:
                st.code(f"{tool.get('name')}({tool.get('args')})")


def render_new_analysis(manager: AnalysisTaskManager) -> None:
    st.header("新建分析")
    st.caption("任务会进入本机队列；同一时间只执行一个分析。")
    with st.form("new-analysis"):
        first, second = st.columns(2)
        ticker_input = first.text_input("Ticker", value="SPY", placeholder="例如 SPY、0700.HK、BTC-USD")
        analysis_date = second.date_input("分析日期", value=date.today(), max_value=date.today())

        try:
            normalized_ticker = normalize_ticker(ticker_input or "SPY")
            asset_type = infer_asset_type(normalized_ticker)
        except Exception:
            normalized_ticker = ticker_input.strip().upper()
            asset_type = "stock"
        st.caption(f"规范化标的：{normalized_ticker or '—'} · 资产类型：{asset_type}")

        analyst_options = ["market", "social", "news"]
        if asset_type != "crypto":
            analyst_options.extend(["fundamentals", "industry"])
        st.caption("行业与产业链分析仅用于个股；运行时识别到 ETF 会自动跳过。")
        analysts = st.multiselect(
            "分析师团队",
            analyst_options,
            default=analyst_options,
            format_func=lambda key: ANALYST_LABELS[key],
        )
        depth_label = st.radio(
            "研究深度",
            ["快速（1 轮）", "标准（3 轮）", "深度（5 轮）"],
            index=0,
            horizontal=True,
        )
        depth = {"快速（1 轮）": 1, "标准（3 轮）": 3, "深度（5 轮）": 5}[depth_label]
        output_label = st.selectbox("报告语言", ["简体中文", "English"])
        output_language = "Chinese" if output_label == "简体中文" else "English"

        providers = sorted(set(MODEL_OPTIONS) | {"openrouter", "azure"})
        default_provider = str(DEFAULT_CONFIG.get("llm_provider", "openai")).lower()
        provider_index = providers.index(default_provider) if default_provider in providers else 0
        provider = st.selectbox(
            "LLM Provider",
            providers,
            index=provider_index,
            format_func=lambda value: PROVIDER_LABELS.get(value, value),
        )
        status_level, status_text = api_key_status(provider)
        getattr(st, status_level)(status_text)

        quick_default = (
            str(DEFAULT_CONFIG.get("quick_think_llm", ""))
            if provider == default_provider
            else ""
        )
        deep_default = (
            str(DEFAULT_CONFIG.get("deep_think_llm", ""))
            if provider == default_provider
            else ""
        )
        quick_model = select_model(provider, "quick", "快速模型", quick_default)
        deep_model = select_model(provider, "deep", "深度模型", deep_default)

        backend_url = (
            DEFAULT_CONFIG.get("backend_url") or "" if provider == default_provider else ""
        )
        google_level = None
        openai_effort = None
        anthropic_effort = None
        with st.expander("高级设置", expanded=False):
            backend_url = st.text_input(
                "Backend URL",
                value=backend_url,
                placeholder="通常留空，使用 Provider 默认地址",
                key=f"backend-url-{provider}",
            ).strip()
            if provider == "google":
                google_level = st.selectbox(
                    "Gemini Thinking Level", [None, "minimal", "low", "medium", "high"]
                )
            elif provider == "openai":
                openai_effort = st.selectbox(
                    "OpenAI Reasoning Effort", [None, "low", "medium", "high"]
                )
            elif provider == "anthropic":
                anthropic_effort = st.selectbox(
                    "Anthropic Effort", [None, "low", "medium", "high"]
                )
            checkpoint_enabled = st.checkbox("启用断点恢复", value=True)

        submitted = st.form_submit_button("加入分析队列", type="primary", use_container_width=True)

    if submitted:
        if status_level == "error":
            st.error(status_text)
            return
        try:
            request = AnalysisRequest(
                ticker=normalized_ticker,
                analysis_date=analysis_date.isoformat(),
                analysts=tuple(analysts),
                research_depth=depth,
                llm_provider=provider,
                quick_think_llm=quick_model,
                deep_think_llm=deep_model,
                asset_type=asset_type,
                output_language=output_language,
                backend_url=backend_url or None,
                google_thinking_level=google_level,
                openai_reasoning_effort=openai_effort,
                anthropic_effort=anthropic_effort,
                checkpoint_enabled=checkpoint_enabled,
            )
            run_id = manager.submit(request)
            st.session_state["selected_run_id"] = run_id
            st.success(f"任务 {run_id} 已加入队列。")
        except ValueError as exc:
            st.error(str(exc))


def render_run_center(manager: AnalysisTaskManager) -> None:
    st.header("运行中心")

    @st.fragment(run_every=1)
    def live_runs() -> None:
        rows = manager.store.list_runs(statuses=ACTIVE_STATUSES)
        if not rows:
            st.info("当前没有运行中或排队中的任务。")
            return
        running = [row for row in rows if row["status"] in {"running", "stop_requested"}]
        queued = sorted(
            [row for row in rows if row["status"] == "queued"],
            key=lambda item: item["created_at"],
        )
        if running:
            st.subheader("当前任务")
            render_run_detail(manager, running[0])
        if queued:
            st.subheader("等待队列")
            for position, row in enumerate(queued, start=1):
                columns = st.columns([1, 3, 2, 1])
                columns[0].write(f"#{position}")
                columns[1].write(f"**{row['ticker']}** · {row.get('analysis_date') or '—'}")
                columns[2].write(format_time(row["created_at"]))
                if columns[3].button("取消", key=f"cancel-{row['id']}"):
                    manager.stop(row["id"])
                    st.rerun(scope="fragment")

    live_runs()


def render_history(manager: AnalysisTaskManager) -> None:
    st.header("历史工作台")
    filter_cols = st.columns(4)
    ticker = filter_cols[0].text_input("筛选 Ticker").strip()
    analysis_date_filter = filter_cols[1].text_input(
        "分析日期", placeholder="YYYY-MM-DD"
    ).strip()
    status_options = filter_cols[2].multiselect(
        "筛选状态",
        list(RUN_STATUSES),
        format_func=lambda value: STATUS_LABELS[value],
    )
    time_range = filter_cols[3].selectbox(
        "创建时间", ["全部", "最近 7 天", "最近 30 天"]
    )
    rows = manager.store.list_runs(
        statuses=tuple(status_options) if status_options else None,
        ticker=ticker or None,
    )
    if time_range != "全部":
        days = 7 if time_range == "最近 7 天" else 30
        threshold = datetime.now().astimezone() - timedelta(days=days)
        rows = [
            row
            for row in rows
            if datetime.fromisoformat(row["created_at"]).astimezone() >= threshold
        ]
    if analysis_date_filter:
        rows = [row for row in rows if row.get("analysis_date") == analysis_date_filter]
    if not rows:
        st.info("没有符合条件的历史记录。")
        return

    labels = {
        row["id"]: (
            f"{row['ticker']} · {row.get('analysis_date') or '日期未知'} · "
            f"{STATUS_LABELS.get(row['status'], row['status'])} · {format_time(row['created_at'])}"
        )
        for row in rows
    }
    default_id = st.session_state.get("selected_run_id")
    ids = list(labels)
    index = ids.index(default_id) if default_id in ids else 0
    run_id = st.selectbox("选择记录", ids, index=index, format_func=lambda value: labels[value])
    row = manager.store.get_run(run_id)
    if row is None:
        return
    st.session_state["selected_run_id"] = run_id
    render_run_detail(manager, row)

    action_cols = st.columns(4)
    if row["source"] != "legacy" and action_cols[0].button("复制参数重新运行"):
        new_id = manager.rerun(run_id)
        st.session_state["selected_run_id"] = new_id
        st.success(f"新任务 {new_id} 已加入队列。")
    if row["status"] in RESUMABLE_STATUSES and action_cols[1].button("从断点恢复"):
        try:
            new_id = manager.resume(run_id)
            st.session_state["selected_run_id"] = new_id
            st.success(f"恢复任务 {new_id} 已加入队列。")
        except ValueError as exc:
            st.error(str(exc))

    report_value = row.get("report_path")
    if report_value:
        report_path = Path(report_value)
        if report_path.exists() and report_path.is_file():
            action_cols[2].download_button(
                "下载完整 Markdown",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="text/markdown",
            )
            action_cols[3].download_button(
                "下载报告 ZIP",
                data=zip_report(report_path),
                file_name=f"{row['ticker']}_{row['id']}.zip",
                mime="application/zip",
            )


def main() -> None:
    manager = get_manager()
    st.sidebar.title("📈 TradingAgents")
    page = st.sidebar.radio("导航", ["新建分析", "运行中心", "历史工作台"])
    st.sidebar.caption("本地单用户工作台 · API 密钥仅从环境读取")
    if page == "新建分析":
        render_new_analysis(manager)
    elif page == "运行中心":
        render_run_center(manager)
    else:
        render_history(manager)


if __name__ == "__main__":
    main()
