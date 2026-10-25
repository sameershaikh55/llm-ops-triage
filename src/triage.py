import json
import time
from typing import Any

from groq import APIStatusError, RateLimitError
from pydantic import ValidationError

from src.config import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    calculate_cost_usd,
    get_groq_client,
)
from src.schema import TriageOutput

SYSTEM_PROMPT = f"""You are an expert customer support triage engine.
Analyze the incoming customer support ticket and output ONLY a valid JSON object matching this exact JSON Schema:

{json.dumps(TriageOutput.model_json_schema(), indent=2)}

Rules:
1. Output strictly valid JSON with the exact keys: category, priority, product, summary, draft_reply.
2. 'category' MUST be one of the exact uppercase strings allowed in the schema.
3. 'priority' MUST be one of: low, medium, high, urgent.
    - 'urgent': account lockouts, unauthorized charges, or payment failures
    - 'high': order cancellations, refunds, or failed deliveries
    - 'medium': standard account/order updates or invoice requests
    - 'low': general questions, feedback, or newsletter inquiries
4. Keep 'summary' concise (under 25 words).
5. Do not wrap the JSON in markdown code blocks or add any extra keys.
"""


def triage_ticket(ticket_text: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """
    Send a support ticket to Groq, validate the response against TriageOutput,
    and return both the structured prediction and operational telemetry.
    """
    client = get_groq_client()

    response = None
    latency_ms = 0.0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Timer starts inside the retry block so rate-limit sleeps don't pollute latency_ms
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Support Ticket:\n{ticket_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,  # Zero temperature for deterministic, repeatable evaluation
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            break
        except RateLimitError:
            wait_sec = 10 * attempt
            print(
                f"\n  [Rate Limit 429] Waiting {wait_sec}s before retry {attempt}/{MAX_RETRIES}..."
            )
            time.sleep(wait_sec)
        except APIStatusError as exc:
            if attempt == MAX_RETRIES:
                raise
            wait_sec = 3 * attempt
            print(
                f"\n  [API Warning {exc.status_code}] Retrying in {wait_sec}s ({attempt}/{MAX_RETRIES})..."
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
        return {
            "schema_valid": True,
            "prediction": validated_data.model_dump(),
            "error": None,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "raw_output": raw_content,
        }
    except ValidationError as exc:
        return {
            "schema_valid": False,
            "prediction": None,
            "error": str(exc),
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

    print("Sending sample ticket to Groq...\n")
    result = triage_ticket(sample_ticket)
    print(json.dumps(result, indent=2))
