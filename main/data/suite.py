"""Load a frozen public/private suite and restrict it to the paper's families."""
import json
from pathlib import Path

from aspire.common import digest, file_hash, read_jsonl
from aspire.families import select


def load_suite(directory, split, partition="test", families="paper"):
    """Verify manifest and file hashes, then return (manifest, public_rows, private_by_id)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if digest({k: v for k, v in manifest.items() if k != "suite_id"}) != manifest["suite_id"]:
        raise ValueError("frozen manifest modified")
    for name, sha in manifest["files"].items():
        if file_hash(directory / name) != sha:
            raise ValueError(f"frozen file modified: {name}")
    public = read_jsonl(directory / f"{partition}_{split}.public.jsonl")
    private = {r["sample_id"]: r for r in read_jsonl(directory / f"{partition}_{split}.private.jsonl")}
    if len(public) != len(private) or {r["sample_id"] for r in public} != set(private):
        raise ValueError("public/private sample mismatch")
    public = select(public, families)
    return manifest, public, {r["sample_id"]: private[r["sample_id"]] for r in public}
