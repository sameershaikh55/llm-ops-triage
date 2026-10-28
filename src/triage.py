import json
import logging
import time
from typing import Any

from groq import APIStatusError, RateLimitError
from pydantic import ValidationError

from src.config import (
    DEFAULT_PROMPT_VERSION,
    MAX_RETRIES,
    calculate_cost_usd,
    get_git_sha,
    get_groq_client,
)
from src.logger import get_logger, log_event
from src.prompts import load_prompt
from src.schema import TriageOutput

logger = get_logger("triage")


def triage_ticket(
    ticket_text: str,
    model: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    """
    Load the versioned prompt artifact, send the support ticket to Groq,
    validate the response against TriageOutput, and return the prediction
    with full operational and provenance metadata (prompt_version, model, git_sha).
    """
    prompt_cfg = load_prompt(name="triage", version=prompt_version)
    active_model = model or prompt_cfg.model
    git_sha = get_git_sha()
    messages = prompt_cfg.render_messages(ticket_text=ticket_text)

    client = get_groq_client()

    response = None
    latency_ms = 0.0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Timer starts inside the retry block so rate-limit sleeps don't pollute latency_ms
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=active_model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=prompt_cfg.temperature,
                max_tokens=prompt_cfg.max_tokens,
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            break
        except RateLimitError:
            wait_sec = 10 * attempt
            log_event(
                logger,
                logging.warning,
                "groq_rate_limit_retry",
                prompt_version=prompt_cfg.version,
                model=active_model,
                git_sha=git_sha,
                error_class="rate_limit",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                wait_sec=wait_sec,
            )
            time.sleep(wait_sec)
        except APIStatusError as exc:
            error_class = "provider_5xx" if exc.status_code >= 500 else "client_error"
            if attempt == MAX_RETRIES:
                log_event(
                    logger,
                    logging.ERROR,
                    "groq_api_failed_permanently",
                    prompt_version=prompt_cfg.version,
                    model=active_model,
                    git_sha=git_sha,
                    error_class=error_class,
                    status_code=exc.status_code,
                    attempt=attempt,
                )
                raise
            wait_sec = 3 * attempt
            log_event(
                logger,
                logging.WARNING,
                "groq_api_error_retry",
                prompt_version=prompt_cfg.version,
                model=active_model,
                git_sha=git_sha,
                error_class=error_class,
                status_code=exc.status_code,
                attempt=attempt,
                max_retries=MAX_RETRIES,
                wait_sec=wait_sec,
            )
            time.sleep(wait_sec)

    if response is None:
        raise RuntimeError(f"Groq request failed after {MAX_RETRIES} attempts.")

    raw_content = response.choices[0].message.content or "{}"
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    cost_usd = calculate_cost_usd(prompt_tokens, completion_tokens)

    # Validate the raw JSON against our strict Pydantic contract
    try:
        validated_data = TriageOutput.model_validate_json(raw_content)
        prediction_dict = validated_data.model_dump()

        log_event(
            logger,
            logging.INFO,
            "triage_completed",
            prompt_version=prompt_cfg.version,
            model=active_model,
            git_sha=git_sha,
            schema_valid=True,
            category=prediction_dict["category"],
            priority=prediction_dict["priority"],
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )

        return {
            "schema_valid": True,
            "prediction": prediction_dict,
            "error": None,
            "prompt_version": prompt_cfg.version,
            "model": active_model,
            "git_sha": git_sha,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "raw_output": raw_content,
        }
    except ValidationError as exc:
        log_event(
            logger,
            logging.WARNING,
            "triage_schema_validation_failed",
            prompt_version=prompt_cfg.version,
            model=active_model,
            git_sha=git_sha,
            schema_valid=False,
            error_class="schema_fail",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )

        return {
            "schema_valid": False,
            "prediction": None,
            "error": str(exc),
            "prompt_version": prompt_cfg.version,
            "model": active_model,
            "git_sha": git_sha,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "raw_output": raw_content,
        }


if __name__ == "__main__":
    sample_ticket = (
        "Hi, I tried to pay for my Pro Subscription twice with my Visa card "
        "and both times it gave me an error, but I see a pending charge on my bank app! "
        "Please help me right away."
    )

    print("Running triage with externalized YAML prompt registry...\n")
    result = triage_ticket(sample_ticket)
    print(json.dumps(result, indent=2))
