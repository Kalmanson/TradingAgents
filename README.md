<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>
<br>
<div align="center">
  <a href="https://github.com/TauricResearch" target="_blank"><img alt="TradingAgents #1 Repository of the Day" src="https://trendshift.io/api/badge/repositories/16192" width="250" height="55"/></a>
</div>
<br>
<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

## News
- [2026-08] **TradingAgents v0.4.0** released with look-ahead / point-in-time fixes across FRED macro, social sentiment, and the decision-log memory; clearer decision signals; working CLI checkpoint resume; Trader price grounding; and the GPT-5.6 and GLM-5.3 models. See [CHANGELOG.md](CHANGELOG.md) for the full list.
- [2026-07] **TradingAgents v0.3.1** released with correctness and stability fixes: Alpha Vantage look-ahead filtering, graph-router crash-safety, graph-shape-aware checkpoint resume, working crypto sentiment sources, a configurable LLM retry budget, Bedrock API-key auth, and Claude Sonnet 5 / Fable 5 support.
- [2026-06] **TradingAgents v0.3.0** released with a verified data-access contract, an expanded provider registry (NVIDIA, Kimi, Groq, Mistral, Bedrock, and any OpenAI-compatible endpoint), FRED and Polymarket data vendors, a current-generation model catalog, and a CI gate.
- [2026-05] **TradingAgents v0.2.5** released with the grounded Sentiment Analyst, GPT-5.5 etc. model coverage, Qwen/GLM/MiniMax dual-region support, `TRADINGAGENTS_*` env-var configurability with API-key auto-detection, remote Ollama support, non-US alpha benchmarks, and ticker path-traversal hardening.
- [2026-04] **TradingAgents v0.2.4** released with structured-output agents (Research Manager, Trader, Portfolio Manager), LangGraph checkpoint resume, persistent decision log, DeepSeek/Qwen/GLM/Azure provider support, Docker, and a Windows UTF-8 encoding fix.
- [2026-03] **TradingAgents v0.2.3** released with multi-language support, GPT-5.4 family models, unified model catalog, backtesting date fidelity, and proxy support.
- [2026-03] **TradingAgents v0.2.2** released with GPT-5.4/Gemini 3.1/Claude 4.6 model coverage, five-tier rating scale, OpenAI Responses API, Anthropic effort control, and cross-platform stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles.

### Analyst Team

- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Industry and Supply Chain Analyst: Explains industry cycles, upstream/downstream transmission, bargaining power, company exposures and falsification conditions using dated disclosures and industry data. Enabled for stocks; ETFs and crypto skip this role.
- Sentiment Analyst: Aggregates news headlines, StockTwits, and Reddit chatter into a single sentiment read to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions, determining the timing and magnitude of trades.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone TradingAgents:
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.12
conda activate tradingagents
```

Install the package and its dependencies:
```bash
pip install .
```

### Docker

Alternatively, run with Docker (the CLI service uses the `cli` profile):
```bash
cp .env.example .env  # add your API keys
docker compose --profile cli run --rm tradingagents
```

For local models with Ollama:
```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export DEEPSEEK_API_KEY=...        # DeepSeek
export DASHSCOPE_API_KEY=...       # Qwen — International (dashscope-intl.aliyuncs.com)
export DASHSCOPE_CN_API_KEY=...    # Qwen — China (dashscope.aliyuncs.com)
export ZHIPU_API_KEY=...           # GLM via Z.AI (international)
export ZHIPU_CN_API_KEY=...        # GLM via BigModel (China, open.bigmodel.cn)
export MINIMAX_API_KEY=...         # MiniMax — Global (api.minimax.io)
export MINIMAX_CN_API_KEY=...      # MiniMax — China (api.minimaxi.com)
export OPENROUTER_API_KEY=...      # OpenRouter
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
```

For Azure OpenAI, copy `.env.enterprise.example` to `.env.enterprise` and fill in your credentials.

For AWS Bedrock, install the extra with `pip install ".[bedrock]"`, set `llm_provider: "bedrock"`, configure AWS credentials (environment variables, `~/.aws/credentials`, or an IAM role) and `AWS_DEFAULT_REGION`, and use a Bedrock model ID, e.g. `us.anthropic.claude-opus-4-8-v1:0`.

For local models, configure Ollama with `llm_provider: "ollama"`. The default endpoint is `http://localhost:11434/v1`; set `OLLAMA_BASE_URL` to point at a remote `ollama-serve`. Pull models with `ollama pull <name>`, and pick "Custom model ID" in the CLI for any model not listed by default.

For any other OpenAI-compatible server (vLLM, LM Studio, llama.cpp, or a custom relay), use `llm_provider: "openai_compatible"` and set the endpoint via `backend_url` (or `TRADINGAGENTS_LLM_BACKEND_URL`), e.g. `http://localhost:8000/v1` for vLLM or `http://localhost:1234/v1` for LM Studio. The model is whatever your server serves. No key is needed for local servers; set `OPENAI_COMPATIBLE_API_KEY` when the endpoint requires one.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### CLI Usage

Launch the interactive CLI:
```bash
tradingagents          # installed command
python -m cli.main     # alternative: run directly from source
```
You will see a screen where you can select your desired tickers, analysis date, LLM provider, research depth, and more.

### 按次付费股票报告网站

可选的付费网站支持 Creem 收银台、11 种报告语言、订单关联的报告持久保存和邮件通知。
价格、币种和税费方式从 Creem 商品读取，首页与订单使用相同来源；订单保存下单时的价格供支付核验。
安装 `pip install ".[mvp]"` 并补齐 `.env` 后，运行 `tradingagents-mvp serve --mode test` 或 `--mode prod`。
同一 `.env` 用 `CREEM_TEST_*`、`CREEM_PROD_*` 区分 Creem 凭据，其他配置共用；缺少必填项会直接拒绝启动并列出配置项。
付费入口默认使用 FMP 行情、证券资料、基本面、财报、新闻和内部人交易，指标在本地计算；填写 `FMP_API_KEY` 并确认对应接口权限后使用。本地 CLI 和 Web 工作台的这些分类默认仍使用 yfinance。新股票报告默认增加行业与产业链分析师，其 SEC、FMP、FRED 和官方文件来源独立组合，配置见[行业分析部署说明](docs/付费报告部署与运行.md#35-行业与产业链分析配置)。

中文文档：[系统架构](docs/付费报告系统架构.md) · [部署与运行](docs/付费报告部署与运行.md) · [文档入口](docs/PAID_REPORT_MVP.md)。
部署文档包含订单离线查询、报告导出、数据库备份恢复，以及 [FMP 配置与回滚](docs/付费报告部署与运行.md#34-fmp-数据源配置与回滚)、真实接口验收和常见故障处理。

### Local Web Workbench

Install the optional Streamlit interface and launch it locally:

```bash
pip install ".[web]"
tradingagents-web
```

The browser workbench runs on `127.0.0.1` and provides a Chinese UI for queued
analysis runs, live agent/report progress, checkpoint-based recovery, and report
history. API keys remain in `.env` or environment variables and are never shown
or persisted by the interface. The existing `tradingagents` CLI remains
available.

### Markets and tickers

With the default Yahoo configuration, TradingAgents supports the markets below using exchange-suffixed tickers. Company identity and the alpha benchmark resolve automatically per market. The [FMP and Marketstack adapters](#fmp-primary-data-source) currently support US stocks/ETFs; they do not replace Yahoo coverage for other markets.

- US: `AAPL`, `SPY`
- Hong Kong: `0700.HK` · Tokyo: `7203.T` · London: `AZN.L`
- India: `RELIANCE.NS`, `.BO` · Canada: `.TO` · Australia: `.AX`
- China A-shares: Shanghai `.SS`, Shenzhen `.SZ` (e.g. `600519.SS` for Kweichow Moutai)
- Crypto: `BTC-USD`, `ETH-USD`

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### FMP primary data source

FMP uses the pinned `fmpsdk==20260824.0` client for its stable API. Defaults
are isolated by entry point; setting only `FMP_API_KEY` preserves this split:

| Entry point | Prices / identity | Indicators | Fundamentals / statements | News / insider trades |
| --- | --- | --- | --- | --- |
| Paid storefront (`tradingagents-mvp serve`, test or prod) | `fmp` | `local` | `fmp` | `fmp` |
| Local CLI / Web workbench | `yfinance` | `yfinance` | `yfinance` | `yfinance` |

```dotenv
FMP_API_KEY=your-key
# Optional overrides apply to EVERY entry point using this environment:
#TRADINGAGENTS_CORE_STOCK_VENDOR=fmp
#TRADINGAGENTS_TECHNICAL_INDICATORS_VENDOR=local
#TRADINGAGENTS_INSTRUMENT_VENDOR=fmp
#TRADINGAGENTS_FUNDAMENTAL_VENDOR=fmp
#TRADINGAGENTS_NEWS_VENDOR=fmp
```

The equivalent configuration is:

```python
from copy import deepcopy
from tradingagents.default_config import DEFAULT_CONFIG

config = deepcopy(DEFAULT_CONFIG)
config["data_vendors"].update({
    "core_stock_apis": "fmp",
    "technical_indicators": "local",
    "instrument_data": "fmp",
    "fundamental_data": "fmp",
    "news_data": "fmp",  # includes insider transactions
})
```

`tool_vendors` takes precedence over category settings. `get_stock_data`
controls prices, local indicators, verification and realized returns;
`get_instrument_info` controls identity and purchase eligibility. Explicit
`CommerceService(base_config=...)` is respected. Fallbacks use only the
configured comma-separated chain: selecting `fmp` alone never calls Yahoo.
`default` allows all registered providers and should not be used when the
source must be restricted. The legacy `yfinance` indicator option still
explicitly uses Yahoo prices.

Daily OHLC is dividend adjusted; raw volume is joined from the unadjusted
endpoint by exact symbol and trading date. Missing prices or volume are
rejected. The five-year indicator window, inclusive dates, stale-data guards
and provider-isolated caches are shared with existing adapters. Reports show
the latest available trading day; daily prices are not real-time quotes.
`BRK.B` and `BRK-B` resolve to the same share class. ETFs work for prices and
benchmarks. Paid ETF reports are disabled by default. When enabled, the reviewed
allowlist supports SPY, QQQ, VOO, IVV, VTI, DIA and IWM. Missing instrument flags or unsupported
exchanges block checkout; US-listed ADRs use listing country, not headquarters.

FMP fundamentals use current profile, quote, TTM ratios and key metrics.
Unavailable fields are marked `N/A`; current snapshots are unavailable for
historical analysis. Statements retain financial period, currency and
filing/acceptance dates, and omit records without verifiable disclosure dates
or disclosed after the cutoff. This does not provide historical restatement
vintages. News is paginated, date-filtered and deduplicated; `global_news_queries`
applies only to Yahoo search. Insider data is the latest up to 100 records,
with transaction and filing dates, not a historical as-of query.

ETF reports use `get_etf_fundamentals` in the existing `fundamental_data`
category (`fmp` or `yfinance`, with the same tool overrides and explicit fallback
rules). FMP calls `etf/info` and `etf/holdings`; Yahoo uses fund data for local
analysis. The fund analyst covers strategy, fees, AUM, holdings and exposures;
it does not use corporate financial statements. These are current snapshots,
with source and available update dates; historical ETF snapshots, tracking error
and synchronized NAV premiums are unavailable. Missing fields remain `N/A`.

Set `TRADINGAGENTS_ETF_REPORTS_ENABLED=true` (or
`config["etf_reports_enabled"] = True`) and restart to enable new paid ETF reports.
The default is `false`: the storefront shows stocks only and rejects ETF orders
before fund-data preflight. Existing orders, local analysis and benchmark prices
continue to work. Explicit `base_config` takes precedence over environment values.

Use `TRADINGAGENTS_ETF_ALLOWLIST=SPY,QQQ,VOO,IVV,VTI,DIA,IWM` (or
`config["etf_allowlist"]`) to select paid ETFs; `none` disables new ETF purchases.
The list is reviewed by the operator because provider metadata does not reliably
identify leverage or inverse exposure. Listing and ETF type must still be verified;
FMP also requires active trading. ETF checkout verifies fund-data availability
using the storefront's own vendor settings before creating a payable order.
The storefront displays the configured list.
New orders persist `asset_type="etf"` through payment and queue recovery. Changing
the list does not cancel existing purchases. Local CLI/Web infer ETF mode from
resolved instrument metadata; explicit ETF requests remain ETF during identity
outages. Confirm access to both ETF endpoints and run a complete sample report
before enabling sales; profile access alone does not establish holdings access.

SDK retries are disabled. The adapter retries transient failures at most twice,
but does not retry authentication, plan permissions or exhausted quotas. Logs
contain endpoint names, request counts, latency and data dates, never API keys
or upstream exception bodies. Enable INFO on `tradingagents.dataflows.fmp` and
`tradingagents.dataflows.market_data`; the storefront log-level flag only
controls commerce/MVP logs.

New orders save effective vendor settings without keys. Existing orders retain
their original settings. To roll back new paid orders to Yahoo, explicitly set
all five selectors to `yfinance` and remove conflicting tool overrides. Removing
overrides restores FMP defaults for the storefront and Yahoo for local CLI/Web.

The SDK's BSD license does not license FMP data for paid reports. FMP requires
an appropriate data display/redistribution agreement; confirm endpoint access,
report/chart display, derived outputs and caching with your commercial contract.
Other sources (macro, prediction markets, social feeds) retain their own routing
and licensing. See [FMP commercial plans](https://site.financialmodelingprep.com/developer/docs/pricing?planType=commercial),
[configuration and rollback](docs/付费报告部署与运行.md#34-fmp-数据源配置与回滚)
and [live smoke tests](docs/付费报告部署与运行.md#113-fmp-真实接口验收).

### Industry and supply-chain analyst

New stock runs include a fifth analyst, `industry`, after fundamentals and before
the research debate. It explains the industry cycle, upstream/downstream
transmission, bargaining power, company exposures, and catalysts/falsification.
Fundamentals retains financial quality and valuation; news retains recent events.
CLI and local Web selections can disable the role. ETFs and crypto skip it,
including ETFs identified after ticker lookup. Explicit analyst lists in old
tasks and orders keep their original selection.

The initial scope is US stocks and SEC-reporting ADRs. Evidence combines SEC
annual/quarterly filings and companyfacts, FMP profiles/revenue splits/peer
candidates, official company documents, and applicable FRED, Census, EIA or
WSTS indicators. NVIDIA and Microsoft have tested IR discovery paths; arbitrary
company websites are not guaranteed. SEC-reported foreign issuers may have an
annual filing without a US-style quarterly report. Related-company research is
limited to three issuers and one level. Peer lists do not establish supply links.

```python
config["industry_sources"] = [
    "sec", "fmp", "fred", "company_ir", "census", "eia", "wsts",
]
config["industry_max_related_companies"] = 3  # integer 0–3
```

These sources are combined, independently of the existing category fallback
chains. FMP and FRED use their existing keys. SEC needs no key: set
`SEC_USER_AGENT="TradingAgents contact@your-domain.com"`; this project otherwise
uses its authorized `COMMERCE_SUPPORT_EMAIL`. The contact is sent only to SEC,
never saved in orders or reports. Requests share a one-per-second SEC limiter
inside the application process. A 403/429 stops further SEC network requests for
that research run; transient failures retry at most once. Indexes cache for one
hour, current public documents for six hours, accession-addressed filings
indefinitely, under the existing run/order cache directory. Multiple deployment
processes would need a shared external rate limiter.

Historical runs filter SEC disclosure dates and request FRED vintages. Current
FMP splits and IR/Census/EIA/WSTS files are excluded when their historical
availability cannot be established. Each source retains dates, periods, units,
URLs and availability notes. Missing sources remain explicit; no usable issuer
disclosure produces an insufficient-evidence report without a model call.
Generated reports receive bounded checks for output truncation, invented numerical conditions,
unprovided citation URLs and WSTS unit conversion, with at most one revision.
These checks do not replace review of all narrative claims.

`industry_report` is supplied to both researchers, all three risk analysts and
the research/portfolio managers. Exports include `1_analysts/industry.md` and the
complete report. New paid orders save the source selection and related-company
limit without contact addresses or API keys. Selecting the role or changing its
source policy isolates checkpoints from incompatible runs.

```bash
.venv/bin/python scripts/check_industry_access.py --symbols NVDA XOM CAT MSFT --output-dir /tmp/industry-check
.venv/bin/python scripts/check_industry_access.py --symbols NVDA --related TSM AMD MU --output-dir /tmp/industry-related
.venv/bin/python scripts/check_industry_access.py --symbols NVDA --date 2025-06-01 --output-dir /tmp/industry-history
```

See the [source verification report](docs/行业分析数据源接入验证.md) for actual
coverage, subscription failures, runtime and historical limitations.

### Marketstack daily market data

Marketstack remains available for US daily prices and instrument identity.
Set `MARKETSTACK_API_KEY`, select `marketstack` for `core_stock_apis` and
`instrument_data`, and `local` for indicators. Fundamentals and news retain
their separately configured provider (FMP in the paid storefront). Marketstack
uses V2 EOD/ticker endpoints, complete pagination, adjusted OHLC/raw volume and
isolated caches. See [Marketstack plans](https://marketstack.com/pricing) and
[service agreements](https://www.ideracorp.com/legal/APILayer) for licensing.

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen (Alibaba DashScope, international and China endpoints), GLM (Zhipu), MiniMax (global + China), OpenRouter, Ollama for local models, and Azure OpenAI for enterprise.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # e.g. openai, google, anthropic, deepseek, groq, ollama; openai_compatible covers any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp, ...)
config["deep_think_llm"] = "gpt-5.6"      # Model for complex reasoning
config["quick_think_llm"] = "gpt-5.6-luna" # Model for quick tasks
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

## Persistence and Recovery

TradingAgents persists two kinds of state across runs.

### Decision log

The decision log is always on. Each completed run appends its decision to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, TradingAgents fetches the realised return (raw and alpha vs SPY), generates a one-paragraph reflection, and injects the most recent same-ticker decisions plus recent cross-ticker lessons into the Portfolio Manager prompt, so each analysis carries forward what worked and what didn't.

Override the path with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### Checkpoint resume

Checkpoint resume is opt-in via `--checkpoint`. When enabled, LangGraph saves state after each node so a crashed or interrupted run resumes from the last successful step instead of starting over. On a resume run you will see `Resuming from step N for <TICKER> on <date>` in the logs; on a new run you will see `Starting fresh`. Checkpoints are cleared automatically on successful completion.

Per-ticker SQLite databases live at `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override the base with `TRADINGAGENTS_CACHE_DIR`). Use `--clear-checkpoints` to reset all of them before a run.

```bash
tradingagents analyze --checkpoint           # enable for this run
tradingagents analyze --clear-checkpoints    # reset before running
```

```python
config = DEFAULT_CONFIG.copy()
config["checkpoint_enabled"] = True
ta = TradingAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

## Reproducibility

TradingAgents is LLM-driven, so two runs of the same ticker and date can differ. This is expected for a research tool built on language models, not a defect. The variation comes from a few distinct sources, and it helps to separate them.

Language model sampling is non-deterministic. Even at a fixed temperature, providers do not guarantee byte-identical output across calls, and reasoning models (the default GPT-5.x family, and any thinking-mode model) vary the most because their internal reasoning is itself sampled.

Live data moves. News, StockTwits, and Reddit return different content as time passes, so a run today sees different inputs than a run last week even for the same historical trade date. Pin the analysis date to hold the price and indicator window fixed, but the social and news sources still reflect "now".

To reduce variation you can lower the sampling temperature. Set `temperature` in your config (or `TRADINGAGENTS_TEMPERATURE` in `.env`); lower values make models that honor it more repeatable. The current curated models are reasoning-first and largely ignore temperature, so for tighter reproducibility use a non-reasoning model, which you can set explicitly via the Custom model ID option.

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["temperature"] = 0.0
# Reasoning models ignore temperature. For tighter reproducibility, set a
# non-reasoning deep/quick model explicitly (e.g. via the Custom model ID option).
```

What does not vary anymore: the analyzed company identity is resolved deterministically from the ticker before any agent runs, and the market analyst grounds exact price and indicator claims in a verified data snapshot. Earlier reports of "different companies" or fabricated price levels across runs are addressed by these two mechanisms.

Backtest results are not guaranteed to match any published figure. Returns depend on the model, the temperature, the date range, data quality, and the sampling above. Treat the framework as a research scaffold for studying multi-agent analysis, not as a strategy with a fixed, replicable return.

## Contributing

Contributions are welcome: bug fixes, documentation, and feature ideas; past contributions are credited per release in [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
