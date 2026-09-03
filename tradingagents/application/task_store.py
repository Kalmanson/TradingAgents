"""SQLite persistence for the local web workbench queue and event history."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.application.runner import AnalysisEvent, AnalysisRequest
from tradingagents.default_config import DEFAULT_CONFIG

RUN_STATUSES = (
    "queued",
    "running",
    "stop_requested",
    "stopped",
    "completed",
    "failed",
    "interrupted",
    "cancelled",
)
ACTIVE_STATUSES = ("queued", "running", "stop_requested")
RESUMABLE_STATUSES = ("stopped", "failed", "interrupted")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


class TaskStore:
    """Small, thread-safe SQLite repository used by the UI and queue worker."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        default = Path(DEFAULT_CONFIG["results_dir"]) / "ui"
        self.base_dir = Path(base_dir) if base_dir else default
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "runs.sqlite3"
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    parent_run_id TEXT,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    analysis_date TEXT,
                    request_json TEXT,
                    signal TEXT,
                    report_path TEXT,
                    external_path TEXT UNIQUE,
                    error_summary TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(parent_run_id) REFERENCES runs(id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status_created
                    ON runs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_ticker
                    ON runs(ticker);
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_id
                    ON run_events(run_id, id);
                """
            )

    def recover_interrupted(self) -> int:
        """Turn stale active executions into explicitly resumable history rows."""
        now = _utc_now()
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                   SET status = 'interrupted', finished_at = ?,
                       error_summary = COALESCE(error_summary, '应用退出导致任务中断')
                 WHERE status IN ('running', 'stop_requested')
                """,
                (now,),
            )
            return cursor.rowcount

    def create_run(
        self,
        request: AnalysisRequest,
        *,
        source: str = "ui",
        parent_run_id: str | None = None,
    ) -> str:
        normalized = request.normalized()
        normalized.validate()
        run_id = uuid.uuid4().hex[:12]
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, parent_run_id, source, status, ticker, analysis_date,
                    request_json, created_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    parent_run_id,
                    source,
                    normalized.ticker,
                    normalized.analysis_date,
                    json.dumps(normalized.to_dict(), ensure_ascii=False),
                    _utc_now(),
                ),
            )
        return run_id

    def clone_run(self, run_id: str, *, resume: bool) -> str:
        row = self.get_run(run_id)
        if row is None or row["source"] == "legacy":
            raise ValueError("该历史记录没有可复用的运行参数。")
        if resume and row["status"] not in RESUMABLE_STATUSES:
            raise ValueError("只有停止、失败或中断的任务可以从断点恢复。")
        request = AnalysisRequest.from_dict(json.loads(row["request_json"]))
        if resume and not request.checkpoint_enabled:
            raise ValueError("原任务未启用 checkpoint，无法从断点恢复。")
        return self.create_run(
            request,
            source="resume" if resume else "rerun",
            parent_run_id=run_id,
        )

    def claim_next(self) -> dict[str, Any] | None:
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            started_at = _utc_now()
            connection.execute(
                "UPDATE runs SET status = 'running', started_at = ?, finished_at = NULL WHERE id = ?",
                (started_at, row["id"]),
            )
            connection.commit()
            claimed = dict(row)
            claimed["status"] = "running"
            claimed["started_at"] = started_at
            return claimed

    def request_stop(self, run_id: str) -> str:
        with self._write_lock, self._connect() as connection:
            row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("任务不存在。")
            status = row["status"]
            if status == "queued":
                connection.execute(
                    "UPDATE runs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                    (_utc_now(), run_id),
                )
                return "cancelled"
            if status == "running":
                connection.execute(
                    "UPDATE runs SET status = 'stop_requested' WHERE id = ?", (run_id,)
                )
                return "stop_requested"
            return status

    def is_stop_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
            return bool(row and row["status"] == "stop_requested")

    def add_event(self, run_id: str, event: AnalysisEvent) -> None:
        payload = json.dumps(
            event.persisted_payload(),
            ensure_ascii=False,
            default=_json_default,
        )
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO run_events (run_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (run_id, event.event_type, payload, event.created_at),
            )

    def finish(
        self,
        run_id: str,
        status: str,
        *,
        signal: str | None = None,
        report_path: str | Path | None = None,
        error_summary: str | None = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"未知任务状态：{status}")
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                   SET status = ?, signal = ?, report_path = ?, error_summary = ?, finished_at = ?
                 WHERE id = ?
                """,
                (
                    status,
                    signal,
                    str(report_path) if report_path else None,
                    error_summary,
                    _utc_now(),
                    run_id,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
        ticker: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            parameters.extend(statuses)
        if ticker:
            clauses.append("ticker LIKE ?")
            parameters.append(f"%{ticker.upper()}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?",
                parameters,
            ).fetchall()
            return [dict(row) for row in rows]

    def list_events(self, run_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, event_type, payload_json, created_at
                  FROM run_events WHERE run_id = ? ORDER BY id DESC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def artifact_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def import_legacy_reports(self, reports_dir: str | Path) -> int:
        root = Path(reports_dir)
        if not root.exists():
            return 0
        imported = 0
        pattern = re.compile(r"^(?P<ticker>.+)_(?P<stamp>\d{8}_\d{6})$")
        for complete_report in sorted(root.glob("*/complete_report.md")):
            resolved = str(complete_report.resolve())
            match = pattern.match(complete_report.parent.name)
            ticker = match.group("ticker") if match else complete_report.parent.name
            created_at = datetime.fromtimestamp(
                complete_report.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="milliseconds")
            if match:
                try:
                    parsed = datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S")
                    created_at = parsed.astimezone().isoformat(timespec="milliseconds")
                except ValueError:
                    pass
            run_id = "legacy-" + hashlib.sha256(resolved.encode()).hexdigest()[:12]
            with self._write_lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO runs (
                        id, source, status, ticker, report_path, external_path,
                        created_at, finished_at
                    ) VALUES (?, 'legacy', 'completed', ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        ticker,
                        resolved,
                        resolved,
                        created_at,
                        created_at,
                    ),
                )
                imported += cursor.rowcount
        return imported
