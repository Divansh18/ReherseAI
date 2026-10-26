"""Load the versioned prompts that are sent to the text LLM."""

from importlib.resources import files


def load_prompt(name: str) -> str:
    if name not in {"decision", "feedback", "early_feedback", "opening_ack", "tts_style"}:
        raise ValueError("Unknown prompt")
    return files("rehearse_ai").joinpath("prompts", f"{name}.txt").read_text(encoding="utf-8")
