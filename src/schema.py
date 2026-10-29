from typing import Literal

from pydantic import BaseModel, Field

TicketCategory = Literal[
    "ACCOUNT",
    "CANCEL",
    "CONTACT",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING",
    "SUBSCRIPTION",
]

TicketPriority = Literal["low", "medium", "high", "urgent"]


class TriageOutput(BaseModel):
    """Strict output contract for the Support Ticket Triage LLM service."""

    category: TicketCategory = Field(
        description="The primary support category of the ticket."
    )
    priority: TicketPriority = Field(
        description="Urgency level: 'urgent' for account lockouts/payment failures, 'high' for blocked orders/refunds, 'medium' for standard issues, 'low' for general inquiries."
    )
    product: str = Field(
        description="The specific product, service, or account feature mentioned in the ticket, or 'General' if none is specified."
    )
    summary: str = Field(
        description="A concise 1-sentence summary of the customer's core issue (maximum 25 words)."
    )
    draft_reply: str = Field(
        description="A polite, empathetic, and actionable response to the customer addressing their issue."
    )


class TriageRequest(BaseModel):
    """HTTP POST request body for the /triage endpoint."""

    ticket_text: str = Field(
        min_length=5,
        max_length=10_000,
        description="Raw customer support ticket text to triage.",
        examples=[
            "Hi, I tried to pay for my Pro Subscription twice with my Visa card and got an error, but I see a pending charge!"
        ],
    )
    prompt_version: str | None = Field(
        default=None,
        description="Optional prompt version override (e.g. 'v1'). Defaults to server PROMPT_VERSION.",
    )


class TriageResponse(TriageOutput):
    """
    HTTP POST response body for /triage.
    Inherits the 5 core TriageOutput fields and attaches M1 provenance & telemetry.
    """

    prompt_version: str = Field(description="Version ID of the YAML prompt used.")
    model: str = Field(description="LLM model identifier that generated the response.")
    git_sha: str = Field(description="Git commit SHA of the running service.")
    latency_ms: float = Field(
        description="End-to-end LLM call latency in milliseconds."
    )
    prompt_tokens: int = Field(description="Number of input prompt tokens consumed.")
    completion_tokens: int = Field(
        description="Number of output completion tokens generated."
    )
    cost_usd: float = Field(description="Estimated USD cost of the request.")


class HealthResponse(BaseModel):
    """HTTP response contract for /healthz and /readyz endpoints."""

    status: Literal["ok", "not_ready"]
    prompt_version: str
    model: str
    git_sha: str
    checks: dict[str, str] = Field(default_factory=dict)
