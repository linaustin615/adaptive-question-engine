"""A local event sink, deliberately not a production audit database."""

import json
from datetime import datetime, timezone
from pathlib import Path


class MemoryAudit:
    def __init__(self):
        self._events = []

    def emit(self, kind: str, **payload) -> None:
        event = {
            "sequence": len(self._events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
        }
        self._events.append(json.dumps(event, sort_keys=True, allow_nan=False))

    @property
    def events(self):
        return tuple(json.loads(event) for event in self._events)

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._events) + "\n", encoding="utf-8")
