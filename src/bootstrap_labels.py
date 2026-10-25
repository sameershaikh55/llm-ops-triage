import argparse
import json
import time
from typing import Any

from groq import APIStatusError, Groq, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from src.config import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    RAW_BOOTSTRAPPED_PATH,
    RAW_SAMPLED_PATH,
    SLEEP_BETWEEN_REQUESTS_SEC,
    get_groq_client,
)
from src.io_utils import append_jsonl, load_jsonl
from src.schema import TicketPriority, TriageOutput


class BootstrapFields(BaseModel):
    """Schema for the 4 fields missing from the raw Bitext dataset."""

    priority: TicketPriority = Field(
        description=(
            "Urgency level based on the ticket and intent: "
            "'urgent' for account lockouts, unauthorized charges, or payment failures; "
            "'high' for order cancellations, refunds, or failed deliveries; "
            "'medium' for standard account/order updates or invoice requests; "
            "'low' for general questions, feedback, or newsletter inquiries."
        )
    )
    product: str = Field(
        description="The specific product, service, or feature mentioned, or 'General' if none is specified."
    )
    summary: str = Field(
        description="A concise 1-sentence summary of the customer's core issue (maximum 25 words)."
    )
    draft_reply: str = Field(
        description="A polite, concise, and actionable customer support reply adapted from the reference response."
    )


TEACHER_SYSTEM_PROMPT = f"""You are a senior customer support data annotator creating a high-quality Golden Dataset.
You are given:
1. The raw customer support ticket.
2. The human-verified category and specific intent from our dataset.
3. A human reference response.

Use this privileged context to generate the missing ground-truth fields as a valid JSON object matching this exact schema:

{json.dumps(BootstrapFields.model_json_schema(), indent=2)}

Rules:
1. Output strictly valid JSON with exact keys: priority, product, summary, draft_reply.
2. 'priority' MUST be one of: low, medium, high, urgent.
3. Follow the urgency rubric in the schema closely based on the verified intent.
4. Keep 'summary' under 25 words.
"""


def generate_missing_fields(client: Groq, raw_record: dict[str, Any]) -> dict[str, Any]:
    """Call Groq with automatic retry/backoff to synthesize the 4 missing ground-truth fields."""
    user_prompt = (
        f"Customer Ticket: {raw_record['ticket_text']}\n"
        f"Verified Category: {raw_record['ground_truth_category']}\n"
        f"Verified Intent: {raw_record['ground_truth_intent']}\n"
        f"Reference Response: {raw_record['reference_response']}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            raw_json = response.choices[0].message.content or "{}"
            bootstrapped = BootstrapFields.model_validate_json(raw_json)

            # Combine human-verified category with the 4 bootstrapped fields
            combined_ground_truth = {
                "category": raw_record["ground_truth_category"],
                "priority": bootstrapped.priority,
                "product": bootstrapped.product,
                "summary": bootstrapped.summary,
                "draft_reply": bootstrapped.draft_reply,
            }

            # Final validation against our full TriageOutput contract from src/schema.py
            validated_full = TriageOutput.model_validate(combined_ground_truth)
            return validated_full.model_dump()

        except RateLimitError:
            wait_sec = 10 * attempt
            print(
                f"  [Rate Limit 429] Waiting {wait_sec}s before retry {attempt}/{MAX_RETRIES}..."
            )
            time.sleep(wait_sec)
        except (ValidationError, APIStatusError) as exc:
            wait_sec = 3 * attempt
            print(
                f"  [Warning] Attempt {attempt}/{MAX_RETRIES} failed ({type(exc).__name__}). Retrying in {wait_sec}s..."
            )
            time.sleep(wait_sec)

    raise RuntimeError(
        f"Failed to bootstrap {raw_record['ticket_id']} after {MAX_RETRIES} attempts."
    )


def run_bootstrap(limit: int | None = None) -> None:
    """Main loop with checkpoint/resume support."""
    client = get_groq_client()

    raw_records = load_jsonl(RAW_SAMPLED_PATH, missing_ok=False)
    existing_record = load_jsonl(RAW_BOOTSTRAPPED_PATH, missing_ok=True)
    completed_ids: set[str] = {r["ticket_id"] for r in existing_record}

    pending_records = [r for r in raw_records if r["ticket_id"] not in completed_ids]

    if limit is not None:
        pending_records = pending_records[:limit]

    print(f"Total raw tickets: {len(raw_records)}")
    print(f"Already bootstrapped (checkpointed): {len(completed_ids)}")
    print(f"Tickets to process in this run: {len(pending_records)}\n")

    if not pending_records:
        print("All 200 tickets are already bootstrapped! Nothing left to do.")
        return

    for idx, raw_record in enumerate(pending_records, start=1):
        ticket_id = raw_record["ticket_id"]
        category = raw_record["ground_truth_category"]
        intent = raw_record["ground_truth_intent"]

        print(
            f"[{idx}/{len(pending_records)}] Bootstrapping {ticket_id} ({category} / {intent})...",
            end=" ",
            flush=True,
        )

        ground_truth = generate_missing_fields(client, raw_record)

        labeled_record = {
            "ticket_id": ticket_id,
            "ticket_text": raw_record["ticket_text"],
            "ground_truth_intent": intent,
            "flags": raw_record["flags"],
            "ground_truth": ground_truth,
        }

        append_jsonl(labeled_record, RAW_BOOTSTRAPPED_PATH)
        print(f"Done -> priority={ground_truth['priority']}")

        # Respect Groq free-tier rate limits
        if idx < len(pending_records):
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    total_done = len(load_jsonl(RAW_BOOTSTRAPPED_PATH, missing_ok=True))
    print(
        f"\nRun complete! Total bootstrapped rows in {RAW_BOOTSTRAPPED_PATH}: {total_done}/200"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap missing ground-truth labels using Groq."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of tickets to process in this run (useful for a quick 5-ticket smoke test).",
    )
    args = parser.parse_args()
    run_bootstrap(limit=args.limit)
