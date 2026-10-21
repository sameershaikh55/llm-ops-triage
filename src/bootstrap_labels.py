import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import APIStatusError, Groq, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from src.schema import TicketPriority, TriageOutput

load_dotenv()

INPUT_PATH = Path("data/raw/sampled_200.jsonl")
OUTPUT_PATH = Path("data/raw/bootstrapped_200.jsonl")

# using Groq model
BOOTSTRAP_MODEL = "openai/gpt-oss-20b"

# Groq Free Tier typically allows 30 requests/minute -> 2.1s pause keeps us safely at ~28 RPM
SLEEP_BETWEEN_REQUESTS_SEC = 2.1
MAX_RETRIES = 5


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all JSON lines from a file."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append a single JSON record to a .jsonl file and flush immediately to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


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
                model=BOOTSTRAP_MODEL,
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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing or invalid in .env")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. Did you run 'uv run python -m src.download_data'?"
        )

    raw_records = load_jsonl(INPUT_PATH)
    existing_record = load_jsonl(OUTPUT_PATH)
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

    client = Groq(api_key=api_key)

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

        append_jsonl(labeled_record, OUTPUT_PATH)
        print(f"Done -> priority={ground_truth['priority']}")

        # Respect Groq free-tier rate limits
        if idx < len(pending_records):
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    total_done = len(load_jsonl(OUTPUT_PATH))
    print(f"\nRun complete! Total bootstrapped rows in {OUTPUT_PATH}: {total_done}/200")


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
