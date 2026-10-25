import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, missing_ok: bool = False) -> list[dict[str, Any]]:
    """
    Load all JSON records from a .jsonl file.
    If missing_ok=True and the file does not exist, returns an empty list [].
    Otherwise raises FileNotFoundError.
    """
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Required dataset file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dictionaries to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append a single JSON record to a .jsonl file and flush immediately to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write a dictionary to a formatted .json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
