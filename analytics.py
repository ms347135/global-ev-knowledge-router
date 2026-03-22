from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from config import ANALYTICS_DIR


EVENT_LOG_PATH = ANALYTICS_DIR / "events.jsonl"


def log_event(event_type: str, payload: dict) -> None:
    ANALYTICS_DIR.mkdir(exist_ok=True)
    record = {"timestamp": datetime.utcnow().isoformat(), "event_type": event_type, **payload}
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_events() -> dict:
    if not EVENT_LOG_PATH.exists():
        return {
            "total_queries": 0,
            "fallback_rate": 0,
            "most_common_collections": {},
            "confidence_distribution": {},
        }

    total_queries = 0
    fallback_queries = 0
    collection_counter = Counter()
    confidence_counter = Counter()

    for line in EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") != "query":
            continue
        total_queries += 1
        if event.get("used_fallback"):
            fallback_queries += 1
        collection_counter.update([event.get("primary_collection", "unknown")])
        confidence_counter.update([event.get("confidence_label", "unknown")])

    fallback_rate = round((fallback_queries / total_queries), 3) if total_queries else 0
    return {
        "total_queries": total_queries,
        "fallback_rate": fallback_rate,
        "most_common_collections": dict(collection_counter.most_common(5)),
        "confidence_distribution": dict(confidence_counter),
    }
