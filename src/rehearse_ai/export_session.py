"""Export one recorded session's transcript and cautiously derived latency evidence."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _ms(event: dict[str, Any] | None) -> float | None:
    value = event.get("session_elapsed_ms") if event else None
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 1)


def _display_ms(value: float | None) -> str:
    return f"{value} ms" if value is not None else "unavailable"


def _duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return round(end - start, 1)


def _speech_end_from_vad(stop: dict[str, Any] | None) -> float | None:
    """Estimate last speech from VAD stop minus its configured silence threshold."""
    stopped = _ms(stop)
    silence = stop.get("stop_secs") if stop else None
    if stopped is None or not isinstance(silence, (int, float)) or silence < 0:
        return None
    estimate = stopped - silence * 1000
    return round(estimate, 1) if estimate >= 0 else None


def load_events(path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not events:
        raise ValueError("Session log is empty")
    session_ids = {event.get("session_id") for event in events}
    if len(session_ids) != 1 or None in session_ids:
        raise ValueError("Expected events from exactly one session")
    sequences = [event.get("sequence") for event in events]
    if any(not isinstance(s, int) for s in sequences) or sequences != sorted(set(sequences)):
        raise ValueError("Session event sequences must be unique and ordered")
    times = [_ms(event) for event in events]
    if any(value is None for value in times) or times != sorted(times):
        raise ValueError("Session elapsed milliseconds must be present and ordered")
    return events


def build_export(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    """Return latency rows, transcript, and summary without inferring absent timestamps."""
    finals = [event for event in events if event["event"] == "candidate_final"]
    rows: list[dict[str, Any]] = []
    transcript_items: list[tuple[int, str]] = []
    by_id: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        utterance_id = event.get("utterance_id")
        if isinstance(utterance_id, int):
            by_id.setdefault(utterance_id, []).append(event)

    for index, final in enumerate(finals):
        turn = final.get("turn_number")
        prior_sequence = finals[index - 1]["sequence"] if index else 0
        next_start = next(
            (
                event["sequence"]
                for event in events
                if event["sequence"] > final["sequence"] and event["event"] == "candidate_started"
            ),
            math.inf,
        )
        same_turn = [
            event
            for event in events
            if event.get("turn_number") == turn and event["sequence"] <= final["sequence"]
        ]
        starts = [event for event in same_turn if event["event"] == "candidate_started"]
        window_start = starts[0]["sequence"] if starts else prior_sequence + 1
        stops = [event for event in same_turn if event["event"] == "vad_stopped"]
        segments = [event for event in same_turn if event["event"] == "stt_final_segment"]
        after = [event for event in events if final["sequence"] < event["sequence"] < next_start]
        prepared = [event for event in after if event["event"] == "agent_prepared"]
        response_ids = {event["utterance_id"] for event in prepared}
        output = next(
            (
                event
                for event in after
                if event["event"] == "agent_output_started"
                and event.get("utterance_id") in response_ids
            ),
            None,
        )
        output_id = output.get("utterance_id") if output else None
        provider_audio = next(
            (
                event
                for event in after
                if event["event"] == "provider_first_audio"
                and event.get("utterance_id") == output_id
            ),
            None,
        )
        tts_request = next(
            (
                event
                for event in after
                if event["event"] == "provider_tts_requested"
                and event.get("utterance_id") == output_id
            ),
            None,
        )
        inference = next((e for e in after if e["event"] == "inference_started"), None)
        cancellations = [
            event
            for event in events
            if window_start <= event["sequence"] < next_start
            and event["event"] in {"agent_interrupted", "agent_cancellation"}
            and (
                event.get("utterance_id") in response_ids or event["sequence"] <= final["sequence"]
            )
        ]
        interrupted = bool(cancellations) or bool(final.get("interrupted"))
        # The assignment names this field for playback. Current logs have only a
        # correlated server output event; the basis column makes that explicit.
        row = {
            "turn": turn,
            "user_stopped_ms": _speech_end_from_vad(stops[-1]) if stops else None,
            "stt_final_ms": _ms(segments[-1]) if segments else None,
            "llm_first_token_ms": None,
            "tts_first_audio_ms": _ms(output),
            "interrupted": interrupted,
            "tts_first_audio_basis": (
                "server_output_start_proxy; browser playback unobserved" if output else None
            ),
            "turn_completed_ms": _ms(final),
            "llm_request_ms": _ms(inference),
            "provider_first_audio_ms": _ms(provider_audio),
            "provider_tts_request_ms": _ms(tts_request),
            "response_utterance_id": output_id,
        }
        rows.append(row)
        start_sequence = starts[0]["sequence"] if starts else final["sequence"]
        transcript_items.append((start_sequence, f"You: {final.get('text') or '[no transcript]'}"))

    for utterance_id, related in by_id.items():
        prepared = next((e for e in related if e["event"] == "agent_prepared"), None)
        output = next((e for e in related if e["event"] == "agent_output_started"), None)
        if prepared is None or output is None:
            continue  # A generated but unplayed response is not a spoken transcript turn.
        cancelled = any(e["event"] == "agent_interrupted" for e in related)
        delivered = any(e["event"] == "agent_delivery_reported" for e in related)
        status = (
            " [Interrupted; only output onset observed, spoken portion unknown]"
            if cancelled
            else ""
        )
        if not cancelled and not delivered:
            status = " [Server output began; completion unconfirmed]"
        transcript_items.append(
            (output["sequence"], f"RehearseAI: {prepared.get('text') or ''}{status}")
        )
    transcript_items.sort(key=lambda item: item[0])
    transcript = (
        "Server-observed transcript. Agent output start/completion does not confirm browser "
        "playback or human hearing. Candidate text is the accumulated final STT turn.\n\n"
        + "\n".join(line for _, line in transcript_items)
        + "\n"
    )

    eligible = [
        _duration(row["user_stopped_ms"], row["tts_first_audio_ms"])
        for row in rows
        if not row["interrupted"]
    ]
    response_latencies = [value for value in eligible if value is not None]

    def average(label: str, pairs: list[tuple[float | None, float | None]]) -> str:
        values = [elapsed for start, end in pairs if (elapsed := _duration(start, end)) is not None]
        result = f"{statistics.mean(values):.1f} ms" if values else "unavailable"
        return (
            f"- {label}: {result} (n={len(values)}; {len(pairs) - len(values)} missing/overlapping)"
        )

    summary = "\n".join(
        [
            "# Latency summary",
            "",
            f"Session: `{events[0]['session_id']}`. All timestamps use server monotonic "
            "milliseconds from this session's log origin; UTC wall timestamps are not "
            "used for subtraction.",
            "",
            "`user_stopped_ms` estimates the last speech as the final Silero VAD stop "
            "event in the candidate turn minus that event's `stop_secs` silence threshold. "
            "It precedes Pipecat's configured turn-completion wait and remains an estimate, "
            "not a measured last audible phoneme. Missing VAD thresholds remain null.",
            "`tts_first_audio_ms` is a correlated **server output start proxy**, not measured "
            "browser playback. Provider first-audio is separately recorded in the extra column. "
            "The current structured LLM call exposes no first-token event, so that field is null.",
            "",
            "## User speech end to first server output",
            "",
            f"- Candidate turns: {len(rows)}; included: {len(response_latencies)}; "
            "excluded (interrupted, cancelled, missing, or reversed): "
            f"{len(rows) - len(response_latencies)}.",
            f"- p50: {_display_ms(_percentile(response_latencies, 0.50))}; "
            f"p95: {_display_ms(_percentile(response_latencies, 0.95))}.",
            "- Percentiles use nearest rank: sorted value at ceil(p × n), with no interpolation.",
            "",
            "## Stage observations",
            "",
            average(
                "speech end → last STT final",
                [(r["user_stopped_ms"], r["stt_final_ms"]) for r in rows],
            ),
            average(
                "last STT final → turn completion",
                [(r["stt_final_ms"], r["turn_completed_ms"]) for r in rows],
            ),
            "- LLM request → first token: unavailable (no first-token event; request timestamps "
            "are retained separately).",
            average(
                "provider TTS request → provider first audio",
                [(r["provider_tts_request_ms"], r["provider_first_audio_ms"]) for r in rows],
            ),
            average(
                "speech end → server output start",
                [(r["user_stopped_ms"], r["tts_first_audio_ms"]) for r in rows],
            ),
            "Stage averages use all turns with observed, nonnegative endpoints, including "
            "interrupted output when it began. Negative differences from overlapping "
            "stages are excluded, not reported as processing durations. Interrupted turns "
            "remain in JSONL and are excluded from the primary response percentile. "
            "No browser onset is available.",
            "",
        ]
    )
    return rows, transcript, summary


def export_session(source: Path, output: Path) -> None:
    events = load_events(source)
    rows, transcript, summary = build_export(events)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcript.txt").write_text(transcript, encoding="utf-8")
    (output / "latency.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "latency_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="Existing raw session JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Empty evidence output folder")
    args = parser.parse_args()
    export_session(args.session, args.output)
    print(f"Exported session evidence to {args.output}")


if __name__ == "__main__":
    main()
