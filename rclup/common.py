"""Shared paths, hashing, partitioning and small IO helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR_HOOSIER = REPO / "vendor" / "hoosierhelp"
RUNS = Path(os.environ.get("RCLUP_RUNS", "/ocean/projects/cis260058p/kshi4/rclup_runs"))
DATA = RUNS / "data"
MODEL_DIR = os.environ.get(
    "RCLUP_MODEL_DIR",
    "/ocean/projects/cis260058p/kshi4/hf_cache_data/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
)
HASH_SALT = "rclup-v2-protocol"

# Appendix B.1: five non-overlapping partitions of the training roots.
PARTITIONS = (
    ("representation", 55),
    ("policy_risk", 15),
    ("selection", 15),
    ("action_calibration", 10),
    ("policy_calibration", 10),
)


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def bucket(*parts, modulus: int = 100) -> int:
    key = "|".join([HASH_SALT, *map(str, parts)])
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) % modulus


def partition(dataset: str, root_id: str) -> str:
    b = bucket(dataset, "partition", root_id)
    acc = 0
    for name, share in PARTITIONS:
        acc += share
        if b < acc:
            return name
    return PARTITIONS[-1][0]


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(canonical(row) + "\n")
        f.flush()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def load_openai_key() -> str:
    """Load OPENAI_API_KEY from the environment or ~/.api_keys_handoff."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / ".api_keys_handoff"
    contents = path.read_text()
    match = re.search(r"(?m)^\s*(?:export\s+)?OPENAI_API_KEY\s*[:=]\s*(.*?)\s*$", contents)
    if not match:
        raise RuntimeError("OPENAI_API_KEY not found")
    values = shlex.split(match.group(1), comments=True)
    if len(values) != 1 or not values[0]:
        raise RuntimeError("Malformed OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = values[0]
    return values[0]


def ensure_vendor_on_path() -> None:
    """HoosierHelp uses absolute `tools`/`eval` imports."""
    src = str(VENDOR_HOOSIER)
    if src not in sys.path:
        sys.path.insert(0, src)


class Timer:
    def __init__(self):
        self.start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start
