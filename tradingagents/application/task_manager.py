"""Persistent single-worker queue for analysis requests."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from tradingagents.application.runner import (
    AnalysisEvent,
    AnalysisRequest,
    AnalysisRunner,
    CancellationToken,
)
from tradingagents.application.task_store import TaskStore

logger = logging.getLogger(__name__)


class AnalysisTaskManager:
    """Own one daemon worker and persist every task transition."""

    def __init__(
        self,
        store: TaskStore | None = None,
        *,
        legacy_reports_dir: str | Path | None = None,
        runner_factory: Callable[..., AnalysisRunner] = AnalysisRunner,
    ) -> None:
        self.store = store or TaskStore()
        self.runner_factory = runner_factory
        self.store.recover_interrupted()
        if legacy_reports_dir is not None:
            self.store.import_legacy_reports(legacy_reports_dir)
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="tradingagents-ui-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, request: AnalysisRequest) -> str:
        run_id = self.store.create_run(request)
        logger.info("用户提交分析任务 event=task_submitted run_id=%s ticker=%s", run_id, request.ticker)
        self._wake.set()
        return run_id

    def rerun(self, run_id: str) -> str:
        new_id = self.store.clone_run(run_id, resume=False)
        logger.info("用户重新运行任务 event=task_rerun run_id=%s new_run_id=%s", run_id, new_id)
        self._wake.set()
        return new_id

    def resume(self, run_id: str) -> str:
        new_id = self.store.clone_run(run_id, resume=True)
        logger.info("用户恢复中断任务 event=task_resumed run_id=%s new_run_id=%s", run_id, new_id)
        self._wake.set()
        return new_id

    def stop(self, run_id: str) -> str:
        status = self.store.request_stop(run_id)
        logger.info("用户请求停止任务 event=task_stop_requested run_id=%s status=%s", run_id, status)
        self._wake.set()
        return status

    def shutdown(self, timeout: float = 2.0) -> None:
        self._shutdown.set()
        self._wake.set()
        self._worker.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            row = self.store.claim_next()
            if row is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            self._execute(row)

    def _execute(self, row: dict) -> None:
        run_id = row["id"]
        terminal = False
        try:
            request = AnalysisRequest.from_dict(json.loads(row["request_json"]))
            logger.info("开始执行任务 event=task_started run_id=%s ticker=%s", run_id, request.ticker)
            runner = self.runner_factory(
                request,
                artifact_dir=self.store.artifact_dir(run_id),
            )
            token = CancellationToken(lambda: self.store.is_stop_requested(run_id))
            for event in runner.stream(token):
                self.store.add_event(run_id, event)
                if event.event_type == "completed" and event.result is not None:
                    self.store.finish(
                        run_id,
                        "completed",
                        signal=event.result.signal,
                        report_path=event.result.report_path,
                    )
                    terminal = True
                elif event.event_type == "stopped":
                    self.store.finish(run_id, "stopped")
                    terminal = True
                elif event.event_type == "failed":
                    self.store.finish(
                        run_id,
                        "failed",
                        error_summary=event.payload.get("error", "分析失败"),
                    )
                    terminal = True
            if not terminal:
                error = "运行器未返回终态。"
                logger.error("任务缺少终态 event=task_failed run_id=%s reason=%s", run_id, error)
                self.store.add_event(run_id, AnalysisEvent("failed", {"error": error}))
                self.store.finish(run_id, "failed", error_summary=error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for key, value in os.environ.items():
                if ("KEY" in key or "TOKEN" in key or "SECRET" in key) and value:
                    error = error.replace(value, "***")
            error = error[:2000]
            # 意外异常（非运行器自身上报的失败）单独记录，便于区分框架错误与分析错误。
            logger.error("任务执行异常 event=task_crashed run_id=%s error=%s", run_id, error)
            self.store.add_event(run_id, AnalysisEvent("failed", {"error": error}))
            self.store.finish(run_id, "failed", error_summary=error)
