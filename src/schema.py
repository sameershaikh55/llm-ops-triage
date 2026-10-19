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
