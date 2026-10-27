"""Synthetic event fixtures for evidence export; no fixture is a real session."""

import json
from pathlib import Path

import pytest

from rehearse_ai.export_session import build_export, export_session, load_events


def event(sequence: int, time_ms: float, name: str, **details: object) -> dict[str, object]:
    return {
        "session_id": "synthetic-export-test",
        "sequence": sequence,
        "session_elapsed_ms": time_ms,
        "event": name,
        **details,
    }


def fixture() -> list[dict[str, object]]:
    return [
        event(1, 100, "candidate_started", turn_number=1),
        event(2, 180, "stt_final_segment", turn_number=1, text="First chunk"),
        event(3, 200, "vad_stopped", turn_number=1, stop_secs=0.1),
        event(4, 250, "stt_final_segment", turn_number=1, text="second chunk"),
        event(5, 300, "candidate_final", turn_number=1, text="First chunk second chunk"),
        event(6, 310, "inference_started"),
        event(7, 400, "agent_prepared", utterance_id=10, text="A question?"),
        event(8, 410, "provider_tts_requested", utterance_id=10),
        event(9, 450, "provider_first_audio", utterance_id=10),
        event(10, 500, "agent_output_started", utterance_id=10),
        event(11, 700, "agent_delivery_reported", utterance_id=10),
        event(12, 1000, "candidate_started", turn_number=2),
        event(13, 1100, "stt_final_segment", turn_number=2, text="Second answer"),
        event(14, 1200, "vad_stopped", turn_number=2, stop_secs=0.05),
        event(15, 1300, "candidate_final", turn_number=2, text="Second answer"),
        event(16, 1400, "agent_prepared", utterance_id=11, text="Unplayed question?"),
        event(17, 1450, "agent_interrupted", utterance_id=11),
        event(18, 1451, "agent_cancellation", utterance_id=11),
        event(19, 1500, "candidate_started", turn_number=3),
        event(20, 1600, "vad_stopped", turn_number=3, stop_secs=0.1),
        event(21, 1650, "candidate_final", turn_number=3, text="Third answer"),
        event(22, 1700, "agent_prepared", utterance_id=12, text="Partial output"),
        event(23, 1720, "provider_first_audio", utterance_id=12),
        event(24, 1730, "agent_output_started", utterance_id=12),
        event(25, 1800, "agent_interrupted", utterance_id=12),
    ]


def test_export_preserves_order_missing_values_and_cancelled_output(tmp_path: Path) -> None:
    events = fixture()
    rows, transcript, summary = build_export(events)
    assert rows[0]["user_stopped_ms"] == 100
    assert rows[0]["stt_final_ms"] == 250  # last of two final STT segments
    assert rows[0]["tts_first_audio_ms"] == 500  # server output, not provider chunk at 450
    assert rows[0]["llm_first_token_ms"] is None
    assert rows[0]["interrupted"] is False
    assert rows[1]["stt_final_ms"] == 1100  # STT can precede VAD stop
    assert rows[1]["tts_first_audio_ms"] is None
    assert rows[1]["interrupted"] is True
    assert rows[2]["stt_final_ms"] is None
    assert rows[2]["interrupted"] is True
    assert "Unplayed question?" not in transcript
    assert (
        "Partial output [Interrupted; only output onset observed, spoken portion unknown]"
        in transcript
    )
    assert transcript.index("You: First chunk") < transcript.index("RehearseAI: A question?")
    assert "p50: 400.0 ms; p95: 400.0 ms" in summary
    assert "2 missing/overlapping" in summary  # STT before speech end is not negative latency

    source = tmp_path / "raw.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")
    output = tmp_path / "evidence"
    export_session(source, output)
    assert source.read_text(encoding="utf-8").count("\n") == len(events)
    assert len((output / "latency.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert (output / "transcript.txt").read_text(encoding="utf-8") == transcript
    with pytest.raises(ValueError, match="not empty"):
        export_session(source, output)


def test_nearest_rank_percentiles_and_missing_event_rejection(tmp_path: Path) -> None:
    events = []
    sequence = 0
    for turn, output_time in enumerate([200, 400, 600, 800, 1000], start=1):
        sequence += 1
        events.append(event(sequence, turn * 2000, "candidate_started", turn_number=turn))
        sequence += 1
        events.append(
            event(sequence, turn * 2000 + 100, "vad_stopped", turn_number=turn, stop_secs=0.1)
        )
        sequence += 1
        events.append(event(sequence, turn * 2000 + 150, "candidate_final", turn_number=turn))
        sequence += 1
        events.append(event(sequence, turn * 2000 + 160, "agent_prepared", utterance_id=turn))
        sequence += 1
        events.append(
            event(
                sequence, turn * 2000 + 100 + output_time, "agent_output_started", utterance_id=turn
            )
        )
    _, _, summary = build_export(events)
    assert "p50: 700.0 ms; p95: 1100.0 ms" in summary
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(event(1, 2, "candidate_final")) + "\n")
    assert len(load_events(bad)) == 1
    bad.write_text(
        json.dumps(event(2, 2, "candidate_final"))
        + "\n"
        + json.dumps(event(1, 3, "candidate_final"))
        + "\n"
    )
    with pytest.raises(ValueError, match="ordered"):
        load_events(bad)
