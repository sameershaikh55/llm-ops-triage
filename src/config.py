import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env once for the entire application
load_dotenv()

# --- Reproducibility ---
SEED = 42
TARGET_SAMPLE_SIZE = 200

# --- Project File Paths ---
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RAW_SAMPLED_PATH = Path("data/raw/sampled_200.jsonl")
RAW_BOOTSTRAPPED_PATH = Path("data/raw/bootstrapped_200.jsonl")
GOLDEN_DEV_PATH = Path("data/golden_dev.jsonl")
GOLDEN_TEST_PATH = Path("data/golden_test.jsonl")
BASELINE_RESULTS_PATH = Path("results/baseline.json")

SPLIT_PATHS: dict[str, Path] = {
    "dev": GOLDEN_DEV_PATH,
    "test": GOLDEN_TEST_PATH,
}

# --- LLM Model & Pricing Configuration ---
DEFAULT_MODEL = "openai/gpt-oss-20b"
INPUT_COST_PER_M = 0.075
OUTPUT_COST_PER_M = 0.30

# --- Operational & Rate-Limit Safeguards ---
SLEEP_BETWEEN_REQUESTS_SEC = 2.1
MAX_RETRIES = 5

_CLIENT: Groq | None = None


def get_groq_client() -> Groq:
    """Return a singleton Groq client to reuse HTTP connections across the entire app."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "gsk_your_api_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is missing or still set to placeholder. Check your .env file!"
            )
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def calculate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated USD cost of a single Groq request."""
    input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return round(input_cost + output_cost, 8)
