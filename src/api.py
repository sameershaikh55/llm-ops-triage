import logging

from fastapi import FastAPI, HTTPException, Response, status
from groq import APIStatusError, RateLimitError

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT_VERSION,
    get_git_sha,
    get_groq_client,
)
from src.logger import get_logger, log_event
from src.prompts import load_prompt
from src.schema import HealthResponse, TriageRequest, TriageResponse
from src.triage import triage_ticket

logger = get_logger("api")

app = FastAPI(
    title="Support Ticket Triage API",
    description="Production LLM Ops service for automated customer support ticket triage.",
    version="0.1.0",
)


def _current_provenance(prompt_version: str | None = None) -> tuple[str, str, str]:
    """Return (prompt_version, model, git_sha) safely even if a prompt file is broken."""
    active_version = prompt_version or DEFAULT_PROMPT_VERSION
    git_sha = get_git_sha()
    try:
        model = load_prompt(name="triage", version=active_version).model
    except Exception:  # noqa: BLE001
        model = DEFAULT_MODEL
    return active_version, model, git_sha


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """
    Liveness probe: confirms the HTTP server process is running.
    Never performs slow external I/O.
    """
    prompt_version, model, git_sha = _current_provenance()
    return HealthResponse(
        status="ok",
        prompt_version=prompt_version,
        model=model,
        git_sha=git_sha,
        checks={"process": "alive"},
    )


@app.get("/readyz", response_model=HealthResponse, tags=["ops"])
def readyz(response: Response) -> HealthResponse:
    """
    Readiness probe: confirms the YAML prompt artifact is valid and the
    Groq API key is configured before accepting traffic.
    """
    prompt_version, model, git_sha = _current_provenance()
    checks: dict[str, str] = {}
    is_ready = True

    # Check 1: Can we load and validate the active YAML prompt?
    try:
        prompt_cfg = load_prompt(name="triage", version=prompt_version)
        model = prompt_cfg.model
        checks["prompt_registry"] = f"ok ({prompt_cfg.name}:{prompt_cfg.version})"
    except Exception as exc:  # noqa: BLE001
        is_ready = False
        checks["prompt_registry"] = f"error: {exc}"

    # Check 2: Is GROQ_API_KEY present and client initializable?
    try:
        get_groq_client()
        checks["groq_client"] = "ok"
    except Exception as exc:  # noqa: BLE001
        is_ready = False
        checks["groq_client"] = f"error: {exc}"

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log_event(
            logger,
            logging.ERROR,
            "readiness_check_failed",
            prompt_version=prompt_version,
            model=model,
            git_sha=git_sha,
            checks=checks,
        )
        return HealthResponse(
            status="not_ready",
            prompt_version=prompt_version,
            model=model,
            git_sha=git_sha,
            checks=checks,
        )

    return HealthResponse(
        status="ok",
        prompt_version=prompt_version,
        model=model,
        git_sha=git_sha,
        checks=checks,
    )


@app.post("/triage", response_model=TriageResponse, tags=["triage"])
def triage_endpoint(request: TriageRequest) -> TriageResponse:
    """
    Triage an incoming support ticket and return structured classification,
    draft reply, telemetry, and full provenance metadata.
    """
    active_version = request.prompt_version or DEFAULT_PROMPT_VERSION
    prompt_version, model, git_sha = _current_provenance(active_version)

    try:
        result = triage_ticket(
            ticket_text=request.ticket_text, prompt_version=active_version
        )
    except FileNotFoundError as exc:
        log_event(
            logger,
            logging.WARNING,
            "invalid_prompt_version_requested",
            prompt_version=active_version,
            model=model,
            git_sha=git_sha,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(exc),
                "prompt_version": active_version,
                "model": model,
                "git_sha": git_sha,
            },
        ) from exc
    except RateLimitError as exc:
        log_event(
            logger,
            logging.ERROR,
            "api_rate_limit_exhausted",
            prompt_version=prompt_version,
            model=model,
            git_sha=git_sha,
            error_class="rate_limit",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Upstream LLM rate limit exhausted after retries.",
                "prompt_version": prompt_version,
                "model": model,
                "git_sha": git_sha,
            },
        ) from exc
    except APIStatusError as exc:
        log_event(
            logger,
            logging.ERROR,
            "api_upstream_provider_error",
            prompt_version=prompt_version,
            model=model,
            git_sha=git_sha,
            status_code=exc.status_code,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": f"Upstream LLM provider error (HTTP {exc.status_code}).",
                "prompt_version": prompt_version,
                "model": model,
                "git_sha": git_sha,
            },
        ) from exc

    if not result["schema_valid"] or result["prediction"] is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "LLM output failed schema validation.",
                "validation_error": result["error"],
                "prompt_version": result["prompt_version"],
                "model": result["model"],
                "git_sha": result["git_sha"],
            },
        )

    prediction = result["prediction"]
    return TriageResponse(
        category=prediction["category"],
        priority=prediction["priority"],
        product=prediction["product"],
        summary=prediction["summary"],
        draft_reply=prediction["draft_reply"],
        prompt_version=result["prompt_version"],
        model=result["model"],
        git_sha=result["git_sha"],
        latency_ms=result["latency_ms"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        cost_usd=result["cost_usd"],
    )
