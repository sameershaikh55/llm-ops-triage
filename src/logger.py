import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from src.config import DEFAULT_MODEL, DEFAULT_PROMPT_VERSION, LOG_LEVEL, get_git_sha
from src.prompts import load_prompt


def _resolve_default_model(prompt_version: str) -> str:
    """Safely resolve the model configured in the active YAML prompt."""
    try:
        return load_prompt(version=prompt_version).model
    except Exception:  # noqa: BLE001
        return DEFAULT_MODEL


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats every Python LogRecord as a single-line JSON object guaranteed to
    carry 'prompt_version', 'model', and 'git_sha' provenance metadata.
    """

    def format(self, record: logging.LogRecord) -> str:
        prompt_version = (
            getattr(record, "prompt_version", None) or DEFAULT_PROMPT_VERSION
        )
        model = getattr(record, "model", None) or _resolve_default_model(prompt_version)
        git_sha = getattr(record, "git_sha", None) or get_git_sha()

        log_payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "prompt_version": prompt_version,
            "model": model,
            "git_sha": git_sha,
        }

        extra_fields: dict[str, Any] = getattr(record, "extra_fields", {})
        if extra_fields:
            log_payload.update(extra_fields)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, ensure_ascii=False)


def get_logger(name: str = "triage") -> logging.Logger:
    """Return a singleton structured JSON logger writing to stdout."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    prompt_version: str | None = None,
    model: str | None = None,
    git_sha: str | None = None,
    **fields: Any,
) -> None:
    """
    Emit a structured JSON log line with explicit provenance and arbitrary
    telemetry key-value pairs (e.g., latency_ms, cost_usd, error_class).
    """
    logger.log(
        level,
        event,
        extra={
            "prompt_version": prompt_version,
            "model": model,
            "git_sha": git_sha,
            "extra_fields": fields,
        },
    )
