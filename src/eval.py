import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from src.triage import DEFAULT_MODEL, triage_ticket

SPLIT_PATHS: dict[str, Path] = {
    "dev": Path("data/golden_dev.jsonl"),
    "test": Path("data/golden_test.jsonl"),
}
DEFAULT_OUTPUT_PATH = Path("results/baseline.json")
SLEEP_BETWEEN_REQUESTS_SEC = 2.1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all JSON records from a .jsonl file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_percentile(values: list[float], percentile: float) -> float:
    """
    Compute the given percentile (e.g., 50 or 95) from a list of floats
    using standard linear interpolation (matching numpy.percentile).
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)

    rank = (percentile / 100.0) * (len(sorted_vals) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    weight = rank - lower_idx

    interpolated = (
        sorted_vals[lower_idx] * (1.0 - weight) + sorted_vals[upper_idx] * weight
    )
    return round(interpolated, 2)


def evaluate_dataset(
    split: str = "test",
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Run the triage pipeline across the chosen Golden Set split and save metrics."""
    dataset_path = SPLIT_PATHS[split]
    records = load_jsonl(dataset_path)

    if limit is not None:
        records = records[:limit]

    total = len(records)
    if total == 0:
        raise ValueError(f"No records found in {dataset_path}")

    print(f"Starting M0 Evaluation | Model: {model} | Split: {split} ({total} tickets)")
    print("-" * 78)

    per_ticket_results: list[dict[str, Any]] = []
    latencies: list[float] = []
    costs: list[float] = []

    schema_valid_count = 0
    category_correct_count = 0
    priority_correct_count = 0
    joint_correct_count = 0

    for idx, record in enumerate(records, start=1):
        ticket_id = record["ticket_id"]
        ticket_text = record["ticket_text"]
        gt = record["ground_truth"]

        res = triage_ticket(ticket_text=ticket_text, model=model)

        is_valid = bool(res["schema_valid"])
        pred = res["prediction"] or {}

        cat_match = is_valid and (pred.get("category") == gt["category"])
        prio_match = is_valid and (pred.get("priority") == gt["priority"])
        joint_match = cat_match and prio_match

        if is_valid:
            schema_valid_count += 1
        if cat_match:
            category_correct_count += 1
        if prio_match:
            priority_correct_count += 1
        if joint_match:
            joint_correct_count += 1

        latencies.append(float(res["latency_ms"]))
        costs.append(float(res["cost_usd"]))

        cat_status = (
            "OK" if cat_match else f"ERR({pred.get('category')}!={gt['category']})"
        )
        prio_status = (
            "OK" if prio_match else f"ERR({pred.get('priority')}!={gt['priority']})"
        )

        print(
            f"[{idx:03d}/{total:03d}] {ticket_id} | "
            f"valid={str(is_valid):<5} | "
            f"cat={cat_status:<22} | "
            f"prio={prio_status:<18} | "
            f"{res['latency_ms']:>6.1f}ms"
        )

        per_ticket_results.append(
            {
                "ticket_id": ticket_id,
                "schema_valid": is_valid,
                "category_correct": cat_match,
                "priority_correct": prio_match,
                "ground_truth": {
                    "category": gt["category"],
                    "priority": gt["priority"],
                    "product": gt["product"],
                },
                "prediction": pred if is_valid else None,
                "error": res["error"],
                "latency_ms": res["latency_ms"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "cost_usd": res["cost_usd"],
            }
        )

        if idx < total:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    schema_valid_rate = round(schema_valid_count / total, 4)
    category_accuracy = round(category_correct_count / total, 4)
    priority_accuracy = round(priority_correct_count / total, 4)
    overall_accuracy = round((category_accuracy + priority_accuracy) / 2.0, 4)
    joint_accuracy = round(joint_correct_count / total, 4)

    p50_latency_ms = compute_percentile(latencies, 50.0)
    p95_latency_ms = compute_percentile(latencies, 95.0)

    total_cost_usd = round(sum(costs), 6)
    avg_cost_per_request_usd = round(total_cost_usd / total, 8)

    report: dict[str, Any] = {
        "milestone": "M0 — Baseline",
        "model": model,
        "dataset_split": str(dataset_path),
        "num_samples": total,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "schema_valid_rate": schema_valid_rate,
            "category_accuracy": category_accuracy,
            "priority_accuracy": priority_accuracy,
            "overall_accuracy": overall_accuracy,
            "joint_exact_match_rate": joint_accuracy,
            "p50_latency_ms": p50_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "avg_cost_per_request_usd": avg_cost_per_request_usd,
            "total_cost_usd": total_cost_usd,
        },
        "per_ticket_results": per_ticket_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_scorecard(report, output_path)
    return report


def print_scorecard(report: dict[str, Any], output_path: Path) -> None:
    """Print a clean formatted summary scorecard to the terminal."""
    m = report["metrics"]
    print("\n" + "=" * 60)
    print(f"  M0 BASELINE EVALUATION SCORECARD ({report['model']})")
    print("=" * 60)
    print(
        f"  Dataset Split          : {report['dataset_split']} (n={report['num_samples']})"
    )
    print(f"  Schema-Valid Rate      : {m['schema_valid_rate'] * 100:>6.2f}%")
    print(f"  Category Accuracy      : {m['category_accuracy'] * 100:>6.2f}%")
    print(f"  Priority Accuracy      : {m['priority_accuracy'] * 100:>6.2f}%")
    print(f"  Overall Accuracy (Avg) : {m['overall_accuracy'] * 100:>6.2f}%")
    print(f"  Joint Exact Match      : {m['joint_exact_match_rate'] * 100:>6.2f}%")
    print("-" * 60)
    print(f"  Latency p50 (Median)   : {m['p50_latency_ms']:>8.2f} ms")
    print(f"  Latency p95 (Tail)     : {m['p95_latency_ms']:>8.2f} ms")
    print(f"  Cost / Request (Avg)   : ${m['avg_cost_per_request_usd']:.6f} USD")
    print(f"  Total Eval Cost        : ${m['total_cost_usd']:.6f} USD")
    print("=" * 60)
    print(f"  Saved full artifact to : {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run M0 baseline evaluation on the Support Ticket Golden Set."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
        help="Which Golden Set split to evaluate ('dev' or 'test'). Default: 'test'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of tickets to evaluate (e.g., --limit 5 for a quick smoke test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path. Default: results/baseline.json",
    )
    args = parser.parse_args()

    evaluate_dataset(
        split=args.split,
        limit=args.limit,
        output_path=args.output,
    )
