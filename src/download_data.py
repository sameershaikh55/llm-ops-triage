import json
import random
from collections import Counter
from typing import Any, get_args

from datasets import load_dataset

from src.config import (
    DATASET_NAME,
    RAW_SAMPLED_PATH,
    SEED,
    TARGET_SAMPLE_SIZE,
)
from src.io_utils import save_jsonl
from src.schema import TicketCategory


def download_and_sample_dataset() -> list[dict[str, Any]]:
    """
    Download the Bitext dataset from Hugging Face, verify category alignment
    with our Pydantic schema, and perform deterministic stratified sampling
    across all 11 categories to produce 200 diverse tickets.
    """
    print(f"Downloading '{DATASET_NAME}' from Hugging Face...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print(f"Total rows downloaded: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}\n")

    # 1. Group all rows by their ground-truth category
    allowed_categories = set(get_args(TicketCategory))
    rows_by_category: dict[str, list[dict[str, any]]] = {
        cat: [] for cat in sorted(allowed_categories)
    }

    unexpected_categories: set[str] = set()
    for row in dataset:
        cat = str(row["category"]).strip().upper()
        if cat in rows_by_category:
            rows_by_category[cat].append(dict(row))
        else:
            unexpected_categories.add(cat)

    if unexpected_categories:
        raise ValueError(
            f"Dataset contains categories not in TicketCategory schema: {unexpected_categories}"
        )

    # 2. Deterministic stratified sampling across all 11 categories
    rng = random.Random(SEED)
    categories_sorted = sorted(rows_by_category.keys())
    num_categories = len(categories_sorted)

    base_per_category = TARGET_SAMPLE_SIZE // num_categories  # 200 // 11 = 18
    remainder = TARGET_SAMPLE_SIZE % num_categories  # 200 % 11 = 2

    sampled_raw: list[dict[str, Any]] = []
    for idx, cat in enumerate(categories_sorted):
        # First 2 categories get 19 samples, remaining 9 categories get 18 samples -> 200 total
        quota = base_per_category + (1 if idx < remainder else 0)
        pool = rows_by_category[cat]
        sampled_raw.extend(rng.sample(pool, quota))

    # Shuffle the combined 200 samples so categories are interleaved
    rng.shuffle(sampled_raw)

    # 3. Format into clean records with sequential ticket IDs
    formatted_records: list[dict[str, Any]] = []
    for i, row in enumerate(sampled_raw, start=1):
        formatted_records.append(
            {
                "ticket_id": f"ticket_{i:03d}",
                "ticket_text": str(row["instruction"]).strip(),
                "ground_truth_category": str(row["category"]).strip().upper(),
                "ground_truth_intent": str(row["intent"]).strip(),
                "flags": str(row["flags"]).strip(),
                "reference_response": str(row["response"]).strip(),
            }
        )

    return formatted_records


if __name__ == "__main__":
    records = download_and_sample_dataset()
    save_jsonl(records, RAW_SAMPLED_PATH)

    # Print summary verification
    category_counts = Counter(r["ground_truth_category"] for r in records)
    print("Stratified Sampling Complete!")
    print(f"Saved {len(records)} tickets to: {RAW_SAMPLED_PATH}\n")
    print("Category Distribution across the 200 sampled tickets:")
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat:<14}: {count} tickets")

    print("\nSample Record #1:")
    print(json.dumps(records[0], indent=2))
