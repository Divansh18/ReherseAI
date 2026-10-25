# RehearseAI

RehearseAI is a local voice AI mock-interview coach under development. **Milestones 1 and 2 are implemented:** the project installs and has a provider-independent interview controller with deterministic tests. It does not yet run a voice session, connect a browser, or call an STT, LLM, or TTS provider.

## Requirements and checks

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync --group dev
uv run python -c "import rehearse_ai; print(rehearse_ai.__name__)"
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The controller is in `src/rehearse_ai/interview.py`. `InterviewController.start()` prepares the first fixed question. The caller plays a returned `Utterance` and then calls `delivery_finished(utterance.id)` only when playback is reported complete. A candidate answer goes through `submit_answer(text)`. The future LLM adapter can call `propose(ReplyKind, text)` to request a follow-up, nudge, clarification, or advancement; the controller chooses the next main question and enforces limits. After question four, the caller must supply feedback grounded in actual answers to `prepare_feedback(text)`. The controller validates its 2–3-line shape, and completion waits for delivery of that utterance.

An interruption invalidates the current utterance ID. `retry_interrupted()` returns the same text with a fresh ID, so an old completion event cannot skip a question. Acknowledged playback does not prove the person heard every word. Empty answers repeat the current question once. The future adapter routes explicit repeat requests to `request_repeat()`, using the same one-per-question clarification allowance. Each question also allows one substantive follow-up or nudge. When the clarification allowance is exhausted, another empty answer or repeat raises `LimitReached` and the controller continues waiting for a substantive answer. The future voice adapter must handle this case in the user experience.

Copy `.env.example` to `.env` only when provider selection is confirmed. Its key names are provisional; no provider key is needed for the current controller. `web/` remains reserved for local browser assets. See [PLAN.md](PLAN.md) for the remaining voice pipeline, logging, and real-session verification work.
