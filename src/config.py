import os
import subprocess
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env once for the entire application
load_dotenv()

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

SAMPLED_RAW_PATH = RAW_DATA_DIR / "sampled_200.jsonl"
BOOTSTRAPPED_RAW_PATH = RAW_DATA_DIR / "bootstrapped_200.jsonl"
GOLDEN_DEV_PATH = DATA_DIR / "golden_dev.jsonl"
GOLDEN_TEST_PATH = DATA_DIR / "golden_test.jsonl"
BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline.json"

SPLIT_PATHS: dict[str, Path] = {
    "dev": GOLDEN_DEV_PATH,
    "test": GOLDEN_TEST_PATH,
}

# --- Dataset & Sampling Constants ---
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
TARGET_SAMPLE_SIZE = 200
SEED = 42

# --- Prompt Registery & Provenance Conifguration ---
DEFAULT_PROMPT_NAME = "triage"
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

# --- Groq Model and Pricing ---
DEFAULT_MODEL = "openai/gpt-oss-20b"
INPUT_COST_PER_M = 0.075  # $0.075 per 1M input tokens
OUTPUT_COST_PER_M = 0.30  # $0.30 per 1M output tokens

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


@lru_cache(maxsize=1)
def get_git_sha() -> str:
    """
    Resolve the current Git commit SHA once and cache it in memory.
    Checks the GIT_SHA env var first (for Docker containers), then falls back to git CLI.
    """
    env_sha = os.getenv("GIT_SHA", "").strip()
    if env_sha:
        return env_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
