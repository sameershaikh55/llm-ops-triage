import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from src.schema import TriageOutput

# Load GROQ_API_KEY from .env into environment variables
load_dotenv()

# Default fast, low-cost model for M0 baseline
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Groq pricing for openai/gpt-oss-20b ($0.075 / 1M input tokens, $0.30 / 1M output tokens)
INPUT_COST_PER_M = 0.075
OUTPUT_COST_PER_M = 0.30

SYSTEM_PROMPT = f"""You are an expert customer support triage engine.
Analyze the incoming customer support ticket and output ONLY a valid JSON object matching this exact JSON Schema:

{json.dumps(TriageOutput.model_json_schema(), indent=2)}

Rules:
1. Output strictly valid JSON with the exact keys: category, priority, product, summary, draft_reply.
2. 'category' MUST be one of the exact uppercase strings allowed in the schema.
3. 'priority' MUST be one of: low, medium, high, urgent.
4. Do not wrap the JSON in markdown code blocks or add any extra keys.
"""


def calculate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated USD cost of a single Groq request."""
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 8)


def triage_ticket(ticket_text: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """
    Send a support ticket to Groq, validate the response against TriageOutput,
    and return both the structured prediction and operational telemetry.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is missing or still set to the placeholder. Check your .env file!"
        )

    client = Groq(api_key=api_key)

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
