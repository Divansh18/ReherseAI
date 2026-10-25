# TrainerTwin

TrainerTwin is a planned local voice AI mock-interview coach. **Milestone 1 only is implemented:** this repository contains the project plan and an installable Python package scaffold. It does not yet run an interview, connect a browser, or call any speech or LLM service.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

## Set up the scaffold

```sh
uv sync --group dev
uv run python -c "import interview_coach; print(interview_coach.__name__)"
uv run ruff check .
```

Copy `.env.example` to `.env` when provider selection is confirmed. The sample key names are provisional; no key is needed for Milestone 1. Local `.env` files are ignored by Git.

`tests/` is reserved for behavior tests starting in Milestone 2. There are no tests yet because the package has no behavior to test. `web/` is reserved for local browser assets starting in Milestone 3. See [PLAN.md](PLAN.md) for architecture, acceptance checks, and open decisions.
