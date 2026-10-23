"""Read and write JSON Lines files (one JSON document per line)."""
import json
from pathlib import Path


def read_jsonl(path):
    """Return the documents in a .jsonl file as a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    """Write documents to a .jsonl file with stable key order and LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    return path