"""Validated local runtime settings; secrets never enter browser responses."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    deepgram_api_key: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    host: str = "127.0.0.1"
    port: int = 7860
    session_dir: Path = Path("sessions")
    stt_model: str = "nova-3-general"
    llm_model: str = "gpt-4.1-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "coral"
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.35
    turn_pause_secs: float = 1.1

    @classmethod
    def load(
        cls,
        env: Mapping[str, str] | None = None,
        env_file: str | Path = ".env",
    ) -> "Settings":
        file_values = dotenv_values(env_file)
        values = {key: value for key, value in file_values.items() if value is not None}
        values.update(os.environ if env is None else env)
        missing = [
            name
            for name in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY")
            if not values.get(name, "").strip()
        ]
        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(
                f"Missing {names}. Add the key(s) to .env using .env.example, then restart."
            )

        def number(name: str, default: str, *, low: float, high: float) -> float:
            raw = values.get(name, default)
            try:
                result = float(raw)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc
            if not low <= result <= high:
                raise ConfigurationError(f"{name} must be between {low} and {high}")
            return result

        host = values.get("REHEARSE_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigurationError("REHEARSE_HOST must be a loopback address for local use")
        port = int(number("REHEARSE_PORT", "7860", low=1, high=65535))
        if str(port) != values.get("REHEARSE_PORT", "7860"):
            raise ConfigurationError("REHEARSE_PORT must be a whole number")

        model_options = {
            "REHEARSE_STT_MODEL": {"nova-3-general", "nova-3"},
            "REHEARSE_LLM_MODEL": {"gpt-4.1-mini", "gpt-4.1"},
            "REHEARSE_TTS_MODEL": {"gpt-4o-mini-tts", "tts-1"},
        }
        for name, allowed in model_options.items():
            if values.get(name, next(iter(sorted(allowed)))) not in allowed:
                raise ConfigurationError(f"{name} must be one of: {', '.join(sorted(allowed))}")
        voices = {
            "alloy",
            "ash",
            "ballad",
            "cedar",
            "coral",
            "echo",
            "fable",
            "marin",
            "nova",
            "onyx",
            "sage",
            "shimmer",
            "verse",
        }
        voice = values.get("REHEARSE_TTS_VOICE", "coral")
        if voice not in voices:
            raise ConfigurationError("REHEARSE_TTS_VOICE is not a supported OpenAI built-in voice")

        return cls(
            deepgram_api_key=values["DEEPGRAM_API_KEY"],
            openai_api_key=values["OPENAI_API_KEY"],
            host=host,
            port=port,
            session_dir=Path(values.get("REHEARSE_SESSION_DIR", "sessions")),
            stt_model=values.get("REHEARSE_STT_MODEL", "nova-3-general"),
            llm_model=values.get("REHEARSE_LLM_MODEL", "gpt-4.1-mini"),
            tts_model=values.get("REHEARSE_TTS_MODEL", "gpt-4o-mini-tts"),
            tts_voice=voice,
            vad_confidence=number("REHEARSE_VAD_CONFIDENCE", "0.7", low=0.1, high=1.0),
            vad_start_secs=number("REHEARSE_VAD_START_SECS", "0.2", low=0.05, high=2.0),
            vad_stop_secs=number("REHEARSE_VAD_STOP_SECS", "0.35", low=0.1, high=2.0),
            turn_pause_secs=number("REHEARSE_TURN_PAUSE_SECS", "1.1", low=0.2, high=5.0),
        )
