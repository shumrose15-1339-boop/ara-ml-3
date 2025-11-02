
import json, os
from typing import Dict, Iterator

def read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def load_dataset(root: str, file: str, max_rows: int = None):
    path = os.path.join(root, file) if not os.path.isabs(file) else file
    rows = []
    for i, ex in enumerate(read_jsonl(path)):
        rows.append(ex)
        if max_rows is not None and i+1 >= max_rows:
            break
    return rows
