"""Ordered local JSONL evidence for actual session attempts."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SessionLog:
    def __init__(self, directory: Path, session_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{session_id}.jsonl"
        self.session_id = session_id
        self._sequence = 0
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = os.fdopen(descriptor, "w", encoding="utf-8")

    def record(
        self,
        event: str,
        *,
        speaker: str | None = None,
        utterance_id: int | None = None,
        text: str | None = None,
        interrupted: bool | None = None,
        latency_ms: float | None = None,
        latency_basis: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        self._sequence += 1
        entry = {
            "session_id": self.session_id,
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "event": event,
            "speaker": speaker,
            "utterance_id": utterance_id,
            "text": text,
            "interrupted": interrupted,
            "latency_ms": latency_ms,
            "latency_basis": latency_basis,
            **details,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()
        return entry

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
