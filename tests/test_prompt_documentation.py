"""Keep the submitted full prompt text aligned with what the app loads."""

import re
from pathlib import Path

from rehearse_ai.prompts import load_prompt


def test_documented_runtime_prompts_match_loaded_files() -> None:
    document = Path("PROMPTS.md").read_text(encoding="utf-8")
    for name in ("opening_ack", "decision", "feedback", "early_feedback", "tts_style"):
        match = re.search(rf"### `{name}\.txt`\n\n```text\n(.*?)\n```", document, re.S)
        assert match is not None, name
        assert match.group(1) == load_prompt(name).rstrip("\n")
