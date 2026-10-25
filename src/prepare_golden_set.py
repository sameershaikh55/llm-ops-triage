import random
from collections import Counter
from typing import Any

from src.config import (
    GOLDEN_DEV_PATH,
    GOLDEN_TEST_PATH,
    RAW_BOOTSTRAPPED_PATH,
    SEED,
    TARGET_SAMPLE_SIZE,
)
from src.io_utils import load_jsonl, save_jsonl
from src.schema import TriageOutput

PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}


def audit_records(records: list[dict[str, Any]]) -> list[str]:
    """
    Validate all 200 bootstrapped records against our Pydantic TriageOutput
    contract and flag any suspiciously long summaries or empty strings.
    """
    if len(records) != TARGET_SAMPLE_SIZE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_SIZE} records in {RAW_BOOTSTRAPPED_PATH}, found {len(records)}."
        )

    seen_ids: set[str] = set()
    warnings: list[str] = []

    for record in records:
        ticket_id = record["ticket_id"]
        if ticket_id in seen_ids:
            raise ValueError(f"Duplicate ticket_id detected: {ticket_id}")
        seen_ids.add(ticket_id)

        # Strict Pydantic validation of the ground_truth block
        gt = TriageOutput.model_validate(record["ground_truth"])

        # Soft quality checks for human review
        summary_words = len(gt.summary.split())
        if summary_words > 30:
            warnings.append(
                f"{ticket_id}: summary is {summary_words} words (target is under 25 words)"
            )
        if len(gt.draft_reply.strip()) < 20:
            warnings.append(
                f"{ticket_id}: draft_reply is unusually short ({len(gt.draft_reply)} chars)"
            )

        if not gt.product.strip():
            warnings.append(f"{ticket_id}: product field is blank")

    return warnings


def stratified_split(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Perform a deterministic 50/50 split stratified by (category, priority)
    so both dev (100 rows) and test (100 rows) share identical distributions.
    """
    rng = random.Random(SEED)

    # Shuffle first with our fixed seed so ties within (category, priority) are random
    shuffled = list(records)
    rng.shuffle(shuffled)

    # Sort by (category, priority_rank) so identical strata sit adjacent to each other
    sorted_by_strata = sorted(
        shuffled,
        key=lambda r: (
            r["ground_truth"]["category"],
            PRIORITY_ORDER[r["ground_truth"]["priority"]],
        ),
    )

    # Alternate dealing into dev (even indices) and test (odd indices)
    dev_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    for idx, record in enumerate(sorted_by_strata):
        if idx % 2 == 0:
            dev_records.append(record)
        else:
            test_records.append(record)

    # Shuffle each split independently so categories aren't grouped sequentially
    rng.shuffle(dev_records)
    rng.shuffle(test_records)

    return dev_records, test_records


def print_distribution_report(
    dev_records: list[dict[str, Any]], test_records: list[dict[str, Any]]
) -> None:
    """Print side-by-side Category and Priority distributions for Dev vs. Test."""
    dev_cats = Counter(r["ground_truth"]["category"] for r in dev_records)
    test_cats = Counter(r["ground_truth"]["category"] for r in test_records)
    all_cats = sorted(set(dev_cats.keys()) | set(test_cats.keys()))

    print("\n1. Category Distribution (Dev vs. Test):")
    print(f"  {'CATEGORY':<14} | {'DEV':>5} | {'TEST':>5} | {'TOTAL':>5}")
    print("  " + "-" * 38)
    for cat in all_cats:
        d_count = dev_cats[cat]
        t_count = test_cats[cat]
        print(f"  {cat:<14} | {d_count:>5} | {t_count:>5} | {d_count + t_count:>5}")

    dev_prios = Counter(r["ground_truth"]["priority"] for r in dev_records)
    test_prios = Counter(r["ground_truth"]["priority"] for r in test_records)

    print("\n2. Priority Distribution (Dev vs. Test):")
    print(f"  {'PRIORITY':<14} | {'DEV':>5} | {'TEST':>5} | {'TOTAL':>5}")
    print("  " + "-" * 38)
    for prio in ["urgent", "high", "medium", "low"]:
        d_count = dev_prios[prio]
        t_count = test_prios[prio]
        print(f"  {prio:<14} | {d_count:>5} | {t_count:>5} | {d_count + t_count:>5}")


def print_spot_check_samples(dev_records: list[dict[str, Any]]) -> None:
    """Print 1 sample per priority level from golden_dev for quick human inspection."""
    print("\n3. Human Spot-Check Samples (from golden_dev.jsonl):")
    print("  " + "=" * 68)
    for prio in ["urgent", "high", "medium", "low"]:
        matches = [r for r in dev_records if r["ground_truth"]["priority"] == prio]
        if not matches:
            continue
        sample = matches[0]
        gt = sample["ground_truth"]
        print(
            f"  [{sample['ticket_id']}] Priority: {gt['priority'].upper()} | "
            f"Category: {gt['category']} | Intent: {sample['ground_truth_intent']}"
        )
        print(f"  Ticket  : {sample['ticket_text']}")
        print(f"  Product : {gt['product']}")
        print(f"  Summary : {gt['summary']}")
        print(f"  Reply   : {gt['draft_reply'][:120]}...")
        print("  " + "-" * 68)


if __name__ == "__main__":
    print(f"Loading bootstrapped dataset from {RAW_BOOTSTRAPPED_PATH}...")
    all_records = load_jsonl(RAW_BOOTSTRAPPED_PATH)

    warnings = audit_records(all_records)
    print("Audit passed: 200/200 records conform to TriageOutput schema.")

    if warnings:
        print(f"\nSoft Quality Warnings ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  - {w}")
    else:
        print("Zero soft quality warnings found!")

    dev_set, test_set = stratified_split(all_records)
    save_jsonl(dev_set, GOLDEN_DEV_PATH)
    save_jsonl(test_set, GOLDEN_TEST_PATH)

    print(f"\nSaved {len(dev_set)} records -> {GOLDEN_DEV_PATH}")
    print(f"Saved {len(test_set)} records -> {GOLDEN_TEST_PATH}")

    print_distribution_report(dev_set, test_set)
    print_spot_check_samples(dev_set)
