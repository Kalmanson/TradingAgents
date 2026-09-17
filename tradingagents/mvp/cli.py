"""付费网站启动入口，以及无需运行 Web 服务的订单查询和报告导出工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def main(argv=None) -> None:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Paid report storefront and offline order tools")
    parser.add_argument("--data-dir", type=Path, help="Override COMMERCE_DATA_DIR")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start one HTTP server and durable worker")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--mode", choices=("test", "prod"),
                       help="选择 Creem 测试或生产配置，优先于 CREEM_MODE；两者均未设置时使用 test")
    serve.add_argument("--log-level", choices=("debug", "info", "warning", "error"), default="info",
                       help="业务和服务器日志级别，默认 info")
    orders = commands.add_parser("orders", help="List saved orders without network calls")
    orders.add_argument("--status")
    orders.add_argument("--ticker")
    orders.add_argument("--limit", type=int, default=30)
    show = commands.add_parser("order", help="Find by local order, Creem order or checkout ID")
    show.add_argument("order_id")
    export = commands.add_parser("export", help="Export the preserved Markdown snapshot")
    export.add_argument("order_id")
    export.add_argument("--output", type=Path, required=True)
    backup = commands.add_parser("backup", help="Consistent SQLite backup including every report body")
    backup.add_argument("--output", type=Path, required=True)
    commands.add_parser("emails", help="List pending/failed delivery records")
    commands.add_parser("events", help="List rejected payment events for manual review")
    retry = commands.add_parser("retry-email", help="Requeue a failed email after checking Resend")
    retry.add_argument("delivery_id")
    retry.add_argument("--checked-provider", action="store_true", required=True,
                       help="Confirm the original email was not accepted by Resend")
    args = parser.parse_args(argv)
    if args.data_dir:
        os.environ["COMMERCE_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    if args.command == "serve":
        from tradingagents.commerce.config import CommerceSettings

        if args.mode:
            os.environ["CREEM_MODE"] = args.mode
        try:
            CommerceSettings.from_env()
        except ValueError as exc:
            parser.error(str(exc))
        os.umask(0o077)
        import uvicorn
        from uvicorn.config import LOGGING_CONFIG

        # 只配置新增模块的业务日志，避免开启模型 SDK 的请求正文/凭证调试日志。
        # 输出到 stderr，离线命令的 stdout 仍保持纯 JSON 或导出路径。
        log_config = deepcopy(LOGGING_CONFIG)
        log_config["formatters"]["commerce"] = {
            "format": "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        }
        log_config["handlers"]["commerce"] = {
            "class": "logging.StreamHandler", "formatter": "commerce", "stream": "ext://sys.stderr",
        }
        for name in ("tradingagents.commerce", "tradingagents.mvp"):
            log_config["loggers"][name] = {
                "handlers": ["commerce"], "level": args.log_level.upper(), "propagate": False,
            }
        # 私密报告令牌位于 URL 中，业务日志开启也不能恢复 HTTP 访问日志。
        uvicorn.run("tradingagents.mvp.app:create_app", factory=True, host=args.host, port=args.port,
                    workers=1, access_log=False,
                    log_config=log_config, log_level=args.log_level,
                    forwarded_allow_ips=os.getenv("COMMERCE_FORWARDED_ALLOW_IPS", "127.0.0.1"))
        return
    # 离线查询不导入分析图、不新建数据库、不启动恢复，也不联系任何外部服务。
    # 只有人工邮件重试需要写库，其他命令均以只读模式打开现有数据库。
    root = Path(os.getenv("COMMERCE_DATA_DIR", str(Path.home() / ".tradingagents/commerce"))).expanduser().resolve()
    db_path = root / "runs.sqlite3"
    if not db_path.is_file():
        parser.error(f"No saved order database at {db_path}")
    writable = args.command == "retry-email"
    with closing(sqlite3.connect(db_path.as_uri() + ("?mode=rw" if writable else "?mode=ro"), uri=True, timeout=30)) as connection:
        connection.row_factory = sqlite3.Row
        if args.command == "orders":
            if not 1 <= args.limit <= 1000:
                parser.error("--limit must be between 1 and 1000")
            clauses, values = [], []
            for field in ("status", "ticker"):
                value = getattr(args, field)
                if value:
                    clauses.append(f"{field} = ?")
                    values.append(value.upper())
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = connection.execute("SELECT id,ticker,language,status,analysis_date,created_at,report_path FROM trade_orders" + where + " ORDER BY created_at DESC LIMIT ?", (*values, args.limit)).fetchall()
            print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))
        elif args.command in {"order", "export"}:
            row = connection.execute("""SELECT * FROM trade_orders
                WHERE id=? OR creem_order_id=? OR creem_checkout_id=?""", (args.order_id,) * 3).fetchone()
            if not row:
                parser.error("Order not found")
            if args.command == "export":
                if not row["report_markdown"]:
                    parser.error("This order has no completed report snapshot")
                content = row["report_markdown"]
                if hashlib.sha256(content.encode()).hexdigest() != row["report_sha256"]:
                    parser.error("Report checksum mismatch; inspect backups before exporting")
                try:
                    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(content)
                except FileExistsError:
                    parser.error("Output already exists; choose a new path")
                print(str(args.output.resolve()))
            else:
                result = dict(row)
                for field in ("idempotency_key", "status_token", "report_token", "checkout_url", "customer_email", "report_markdown"):
                    result.pop(field, None)
                result["params"] = json.loads(result.pop("params_json"))
                result["report_saved"] = bool(row["report_markdown"])
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "backup":
            try:
                descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
            except FileExistsError:
                parser.error("Backup already exists; choose a new path")
            with closing(sqlite3.connect(args.output)) as destination:
                connection.backup(destination)
            print(str(args.output.resolve()))
        elif args.command == "emails":
            rows = connection.execute("""SELECT id,order_id,template_type,status,attempt_count,error_summary,next_attempt_at
                FROM email_deliveries WHERE status IN ('PENDING','SENDING','FAILED') ORDER BY next_attempt_at LIMIT 100""").fetchall()
            print(json.dumps([dict(row) for row in rows], indent=2))
        elif args.command == "events":
            rows = connection.execute("""SELECT event_id,event_type,order_id,processing_status,error_summary,received_at
                FROM webhook_events WHERE processing_status='REJECTED' ORDER BY received_at DESC LIMIT 100""").fetchall()
            print(json.dumps([dict(row) for row in rows], indent=2))
        elif args.command == "retry-email":
            new_id = uuid.uuid4().hex
            with connection:
                cursor = connection.execute("""UPDATE email_deliveries SET id=?,status='PENDING',attempt_count=0,
                    first_attempt_at=NULL,next_attempt_at=?,error_summary=NULL WHERE id=? AND status='FAILED'
                    AND order_id IN (SELECT id FROM trade_orders WHERE status IN ('COMPLETED','FAILED'))""",
                                            (new_id, datetime.now(timezone.utc).isoformat(), args.delivery_id))
                if cursor.rowcount != 1:
                    parser.error("No failed delivery eligible for retry")
            print(f"Requeued delivery {new_id}")


if __name__ == "__main__":
    main()
