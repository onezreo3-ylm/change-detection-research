"""Resume a hash-pinned HTTP archive with bounded parallel byte ranges."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import time
import urllib.request

import yaml

from scripts.audit_whu_candidate_remote import remote_identity


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def byte_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    if total <= 0 or chunk_size <= 0:
        raise ValueError("Total and chunk size must be positive")
    return [
        (start, min(total - 1, start + chunk_size - 1))
        for start in range(0, total, chunk_size)
    ]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download_part(
    url: str, etag: str, directory: Path, index: int, start: int, end: int
) -> Path:
    part = directory / f"{index:05d}.part"
    expected = end - start + 1
    if part.exists():
        if part.stat().st_size == expected:
            return part
        raise RuntimeError(f"Incomplete existing chunk: {part}")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "If-Range": etag,
                    "User-Agent": "Mozilla/5.0 (research dataset intake)",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"Chunk {index} HTTP {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"Chunk {index} bad range: {content_range}")
                payload = response.read()
            if len(payload) != expected:
                raise RuntimeError(f"Chunk {index} short body: {len(payload)}")
            temp = part.with_suffix(".tmp")
            with temp.open("wb") as target:
                target.write(payload)
            temp.replace(part)
            return part
        except Exception as error:
            last_error = error
            time.sleep(attempt + 1)
    raise RuntimeError(f"Chunk {index} failed: {last_error}")


def download(
    *,
    url: str,
    expected_size: int,
    expected_etag: str,
    expected_sha256: str,
    output: Path,
    workers: int,
    chunk_size: int,
) -> Path:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("Expected lowercase SHA256")
    if workers < 1 or workers > 24:
        raise ValueError("workers must be between 1 and 24")
    if output.exists():
        if output.stat().st_size == expected_size and _digest(output) == expected_sha256:
            return output
        raise RuntimeError(f"Existing output is not the expected archive: {output}")
    incomplete = output.with_name(output.name + ".assembling")
    if incomplete.exists():
        if incomplete.stat().st_size == expected_size and _digest(incomplete) == expected_sha256:
            incomplete.replace(output)
            return output
        raise RuntimeError(f"Existing assembled file is not the expected archive: {incomplete}")
    actual_size, etag = remote_identity(url)
    if actual_size != expected_size:
        raise RuntimeError("Remote archive size changed")
    if etag != expected_etag:
        raise RuntimeError("Remote archive ETag changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.with_name(output.name + ".chunks")
    directory.mkdir(exist_ok=True)
    ranges = byte_ranges(expected_size, chunk_size)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_part, url, etag, directory, i, first, last): i
            for i, (first, last) in enumerate(ranges)
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 8 == 0 or completed == len(ranges):
                print(f"chunks={completed}/{len(ranges)}", flush=True)
    h = hashlib.sha256()
    with incomplete.open("wb") as target:
        for index, (first, last) in enumerate(ranges):
            part = directory / f"{index:05d}.part"
            if part.stat().st_size != last - first + 1:
                raise RuntimeError(f"Chunk length changed: {part}")
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    h.update(block)
                    target.write(block)
    if incomplete.stat().st_size != expected_size or h.hexdigest() != expected_sha256:
        raise RuntimeError("Assembled archive SHA256 mismatch")
    incomplete.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "train", "val"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = cfg["splits"][args.split]
    result = download(
        url=spec["url"],
        expected_size=int(spec["size_bytes"]),
        expected_etag=spec["etag"],
        expected_sha256=spec["sha256"],
        output=args.output_root.resolve() / f"{args.split}.zip",
        workers=args.workers,
        chunk_size=args.chunk_mib * 1024 * 1024,
    )
    print(f"verified={result} bytes={result.stat().st_size}")


if __name__ == "__main__":
    main()
