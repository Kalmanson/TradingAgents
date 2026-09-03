# TradingAgents 本地 Web 工作台

## 目标

在保留现有 CLI 的基础上提供中文 Streamlit 本地工作台，让用户能够创建并排队分析任务、实时查看 Agent 和报告进度、停止或恢复任务，以及浏览和下载历史报告。

第一版面向本机单用户，不提供账号系统、公网部署、并行分析或行情图表。API 密钥继续通过 `.env` 或环境变量配置，界面只显示对应变量是否存在。

## 架构

- `tradingagents.application.runner` 提供前端无关的请求、事件、结果和流式运行器，由 CLI 与 Web 界面共同消费。
- `tradingagents.application.task_store` 使用 SQLite 保存任务、状态和运行事件，不保存 API 密钥。
- `tradingagents.application.task_manager` 启动单一后台 worker，严格按照入队顺序执行任务。
- `tradingagents.web` 提供 Streamlit 页面和 `tradingagents-web` 启动命令。
- UI 运行产物存放在 `DEFAULT_CONFIG["results_dir"]/ui/runs/`，现有项目 `reports/` 目录作为只读 legacy 历史来源。

## 行为约定

- 界面显式填写的配置优先，未展示配置继承环境覆盖后的 `DEFAULT_CONFIG`。
- checkpoint 默认启用。活动任务停止后保留 checkpoint；恢复操作创建关联的新任务并复用原参数。
- 排队任务可立即取消；活动任务在当前 LLM 或工具调用返回后的 graph chunk 边界停止。
- 应用重启时，尚未开始的任务继续排队，原运行中任务改为 `interrupted` 并等待用户手动恢复。
- 成功任务生成完整 Markdown 报告树，支持下载单文件和 ZIP。
- legacy 报告只推断 ticker 和生成时间，不猜测缺失配置或交易信号。

## 界面

1. **新建分析**：配置 ticker、日期、分析师、研究深度、报告语言、Provider 和模型；高级区域配置 endpoint、推理强度和 checkpoint。
2. **运行中心**：展示当前任务、等待队列、阶段时间线、Agent 状态、调用统计、报告片段和折叠日志。
3. **历史工作台**：筛选和查看任务，复制已完成任务参数，恢复失败/停止/中断任务，并下载报告。

## 验收

- 两个连续提交的任务严格串行执行。
- 任务状态、Agent 状态、消息、工具调用、token 和报告片段能够实时刷新。
- 停止、失败、应用重启和 checkpoint 恢复符合上述约定。
- SQLite、错误摘要和下载内容不包含环境中的 API 密钥。
- 现有 `reports/` 可以重复扫描且不会生成重复记录。
- 原 `tradingagents` CLI 保持可用，新界面通过 `pip install ".[web]"` 后运行 `tradingagents-web` 启动。
