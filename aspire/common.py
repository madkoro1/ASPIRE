"""Hashing and crash-safe JSON/JSONL persistence. No model or solver imports."""
import hashlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path, recover_tail=False):
    """Read JSONL; optionally drop a torn final line left by an interrupted run."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    records, offset = [], 0
    lines = data.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if recover_tail and i == len(lines) - 1 and not line.endswith(b"\n"):
                with open(path, "r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"invalid JSONL: {path}:{i + 1}")
        records.append(record)
        offset += len(line)
    if recover_tail and data and records and not data.endswith(b"\n") and offset == len(data):
        with open(path, "ab") as stream:
            stream.write(b"\n")
    return records


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pin_json(path, value):
    """Write a run configuration once; refuse to resume under a different one."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"configuration changed: {path}; use a new run directory")
    else:
        write_json(path, value)


def code_hashes():
    return {str(p.relative_to(PACKAGE_ROOT.parent)).replace("\\", "/"): file_hash(p)
            for p in sorted(PACKAGE_ROOT.rglob("*.py"))}
