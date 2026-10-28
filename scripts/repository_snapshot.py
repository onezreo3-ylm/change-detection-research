"""Verify/restore public text evidence without importing research dependencies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence/snapshot_manifest.json"
CHUNK = 1024 * 1024


def local_path(value: str) -> Path:
    parts = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or parts.is_absolute()
            or ".." in parts.parts):
        raise ValueError(f"Unsafe snapshot path: {value!r}")
    result = (ROOT / value).resolve()
    if result == ROOT or not result.is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {value!r}")
    return result


def digest(stream, copy_to=None) -> tuple[int, str]:
    size, sha = 0, hashlib.sha256()
    while chunk := stream.read(CHUNK):
        sha.update(chunk)
        size += len(chunk)
        if copy_to is not None:
            copy_to.write(chunk)
    return size, sha.hexdigest()


def check_file(record: dict) -> None:
    path = local_path(record["path"])
    with path.open("rb") as stream:
        size, sha = digest(stream)
    if (size, sha) != (record["bytes"], record["sha256"]):
        raise ValueError(f"Snapshot byte identity differs: {record['path']}")


def check_original(record: dict, copy_to=None) -> None:
    with gzip.open(local_path(record["path"]), "rb") as stream:
        size, sha = digest(stream, copy_to)
    if (size, sha) != (record["original_bytes"], record["original_sha256"]):
        raise ValueError(f"Decompressed identity differs: {record['original_path']}")


def restore_one(record: dict) -> str:
    check_file(record)
    destination = local_path(record["original_path"])
    if not record["original_path"].startswith("runs/"):
        raise ValueError("Evidence restoration is restricted to runs/")
    if destination.exists():
        with destination.open("rb") as stream:
            size, sha = digest(stream)
        if (size, sha) != (record["original_bytes"], record["original_sha256"]):
            raise FileExistsError(f"Preserve different existing file: {destination}")
        return "already_present"
    # Validate the full payload before creating any destination file.
    with tempfile.TemporaryFile() as validated:
        check_original(record, validated)
        validated.seek(0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            shutil.copyfileobj(validated, output, CHUNK)
    return "restored"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--deep", action="store_true",
                        help="also stream-verify original uncompressed text bytes")
    restore = commands.add_parser("restore")
    selection = restore.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--path", action="append", dest="paths")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported snapshot manifest")
    records = manifest["compressed_records"]
    if args.command == "verify":
        for record in manifest["files"]:
            check_file(record)
        if args.deep:
            for record in records:
                check_original(record)
        print(json.dumps(dict(status="verified", files=len(manifest["files"]),
                              compressed_records=len(records), deep=args.deep)))
    else:
        by_path = {record["original_path"]: record for record in records}
        requested = list(by_path) if args.all else args.paths
        unknown = sorted(set(requested) - set(by_path))
        if unknown:
            raise ValueError(f"Not a compressed evidence path: {unknown}")
        outcomes = {path: restore_one(by_path[path]) for path in requested}
        print(json.dumps(dict(status="complete", restored=sum(
            value == "restored" for value in outcomes.values()),
            already_present=sum(value == "already_present" for value in outcomes.values()))))


if __name__ == "__main__":
    main()
