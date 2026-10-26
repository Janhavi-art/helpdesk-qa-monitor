"""Small JSONL telemetry helpers; no external telemetry service is used."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import API_USAGE_LOG, INTERACTIONS_LOG, LOG_DIR


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except json.JSONDecodeError:
                continue
    return rows


def log_interaction(record: dict[str, Any]) -> None:
    append_jsonl(INTERACTIONS_LOG, record)


def log_api_usage(record: dict[str, Any]) -> None:
    append_jsonl(API_USAGE_LOG, record)


def today_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    today = datetime.now().date().isoformat()
    return [r for r in rows if str(r.get("timestamp", ""))[:10] == today]


def usage_summary() -> dict[str, Any]:
    all_calls = read_jsonl(API_USAGE_LOG)
    todays = today_records(all_calls)
    successful = [r for r in todays if r.get("status") == "success"]
    interactions = read_jsonl(INTERACTIONS_LOG)
    return {
        "api_calls_today": len(todays),
        "successful_api_calls_today": len(successful),
        "tokens_today": sum(int(r.get("total_tokens") or 0) for r in successful),
        "input_tokens_today": sum(int(r.get("input_tokens") or 0) for r in successful),
        "output_tokens_today": sum(int(r.get("output_tokens") or 0) for r in successful),
        "interaction_count": len(interactions),
        "cache_hits": sum(bool(r.get("cache_hit")) for r in interactions),
        "local_abstentions": sum(bool(r.get("local_abstention")) for r in interactions),
        "api_errors_today": sum(r.get("status") == "error" for r in todays),
    }


def clear_logs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in (API_USAGE_LOG, INTERACTIONS_LOG):
        path.write_text("", encoding="utf-8")
    last_call = LOG_DIR / "last_api_call.txt"
    if last_call.exists():
        last_call.unlink()
