from typing import Any

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from src.api import app

client = TestClient(app)


def test_healthz_returns_ok_and_provenance() -> None:
    """Liveness probe must return 200 OK with M1 provenance fields."""
    response = client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["prompt_version"] == "v1"
    assert data["model"] == "openai/gpt-oss-20b"
    assert isinstance(data["git_sha"], str) and len(data["git_sha"]) > 0
    assert data["checks"]["process"] == "alive"


def test_readyz_returns_ok_when_configured(monkeypatch: MonkeyPatch) -> None:
    """Readiness probe must return 200 OK when prompt and Groq client initialize."""
    monkeypatch.setattr("src.api.get_groq_client", lambda: object())

    response = client.get("/readyz")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "ok" in data["checks"]["prompt_registry"]
    assert data["checks"]["groq_client"] == "ok"


def test_readyz_returns_503_when_groq_key_missing(monkeypatch: MonkeyPatch) -> None:
    """Readiness probe must return 503 when GROQ_API_KEY is missing or invalid."""

    def _raise_missing_key() -> None:
        raise RuntimeError("GROQ_API_KEY is missing")

    monkeypatch.setattr("src.api.get_groq_client", _raise_missing_key)

    response = client.get("/readyz")
    assert response.status_code == 503

    data = response.json()
    assert data["status"] == "not_ready"
    assert "error" in data["checks"]["groq_client"]
    assert data["prompt_version"] == "v1"


def test_triage_endpoint_success_and_provenance(monkeypatch: MonkeyPatch) -> None:
    """POST /triage must return structured fields and M1 provenance metadata."""

    def _mock_triage_ticket(
        ticket_text: str,
        model: str | None = None,
        prompt_version: str = "v1",
    ) -> dict[str, Any]:
        return {
            "schema_valid": True,
            "prediction": {
                "category": "PAYMENT",
                "priority": "urgent",
                "product": "Pro Subscription",
                "summary": "Customer experienced duplicate pending charge after payment error.",
                "draft_reply": "We apologize for the payment error and are checking your pending charge immediately.",
            },
            "error": None,
            "prompt_version": prompt_version,
            "model": model or "openai/gpt-oss-20b",
            "git_sha": "abc1234",
            "latency_ms": 215.4,
            "prompt_tokens": 310,
            "completion_tokens": 190,
            "cost_usd": 0.00008025,
            "raw_output": "{}",
        }

    monkeypatch.setattr("src.api.triage_ticket", _mock_triage_ticket)

    response = client.post(
        "/triage",
        json={
            "ticket_text": "My Visa card was charged twice for my Pro Subscription!",
            "prompt_version": "v1",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert data["category"] == "PAYMENT"
    assert data["priority"] == "urgent"
    assert data["product"] == "Pro Subscription"
    assert data["prompt_version"] == "v1"
    assert data["model"] == "openai/gpt-oss-20b"
    assert data["git_sha"] == "abc1234"
    assert data["latency_ms"] == 215.4


def test_triage_endpoint_rejects_short_ticket_422() -> None:
    """Tickets shorter than 5 characters must be rejected by Pydantic with HTTP 422."""
    response = client.post("/triage", json={"ticket_text": "hi"})
    assert response.status_code == 422


def test_triage_endpoint_returns_502_on_schema_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """If the LLM output fails Pydantic validation, /triage must return HTTP 502 with provenance."""

    def _mock_broken_triage(
        ticket_text: str,
        model: str | None = None,
        prompt_version: str = "v1",
    ) -> dict[str, Any]:
        return {
            "schema_valid": False,
            "prediction": None,
            "error": "1 validation error for TriageOutput",
            "prompt_version": prompt_version,
            "model": "openai/gpt-oss-20b",
            "git_sha": "abc1234",
            "latency_ms": 180.0,
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "cost_usd": 0.0000375,
            "raw_output": '{"bad": "json"}',
        }

    monkeypatch.setattr("src.api.triage_ticket", _mock_broken_triage)

    response = client.post(
        "/triage",
        json={"ticket_text": "Can you help me reset my account password please?"},
    )
    assert response.status_code == 502

    detail = response.json()["detail"]
    assert detail["prompt_version"] == "v1"
    assert detail["model"] == "openai/gpt-oss-20b"
    assert detail["git_sha"] == "abc1234"
