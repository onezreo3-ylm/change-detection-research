"""Full read-only inventory and pixel-label audit for local ValaisCD dev splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import yaml

from scripts.audit_valais_candidate_remote import split_members
from scripts.audit_whu_candidate_remote import ZipEntry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(archive: Path, split: str, expected_size: int, expected_sha: str):
    if split not in ("dev", "train", "val"):
        raise ValueError("Official test archive is outside this audit")
    if archive.stat().st_size != expected_size or _sha256(archive) != expected_sha:
        raise RuntimeError(f"Unverified {split} archive bytes")
    with zipfile.ZipFile(archive) as zf:
        entries = {
            info.filename: ZipEntry(
                info.filename,
                info.compress_type,
                info.CRC,
                info.compress_size,
                info.file_size,
                info.header_offset,
            )
            for info in zf.infolist()
        }
        groups = split_members(entries, split)
        all_positive = 0
        empty = 0
        full = 0
        label_values = set()
        image_modes = Counter()
        pair_digests = {}
        for identifier in sorted(groups["labels"], key=lambda value: int(value)):
            images = []
            for year in ("2017", "2023"):
                raw = zf.read(groups[year][identifier].name)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (256, 256) or image.mode != "RGB":
                        raise RuntimeError(f"Invalid {split}/{year}/{identifier} image")
                    pixels = image.tobytes()
                    image_modes[image.mode] += 1
                images.append(hashlib.sha256(pixels).digest())
            pair_digests[identifier] = hashlib.sha256(b"".join(images)).hexdigest()
            raw_label = zf.read(groups["labels"][identifier].name)
            with Image.open(io.BytesIO(raw_label)) as image:
                if image.size != (256, 256) or image.mode != "L":
                    raise RuntimeError(f"Invalid {split}/labels/{identifier} image")
                label = np.asarray(image)
            unique = set(np.unique(label).tolist())
            label_values.update(unique)
            if not unique.issubset({0, 1}):
                raise RuntimeError(f"Invalid label values: {split}/{identifier}: {unique}")
            positives = int(label.sum())
            all_positive += positives
            empty += positives == 0
            full += positives == label.size
        digest_counts = Counter(pair_digests.values())
        duplicate_pair_groups = sum(count > 1 for count in digest_counts.values())
        if duplicate_pair_groups:
            raise RuntimeError(f"Duplicate decoded image pairs within {split}")
        return {
            "split": split,
            "archive": str(archive.resolve()),
            "archive_sha256": expected_sha,
            "pairs": len(groups["labels"]),
            "label_values": sorted(label_values),
            "positive_pixels": all_positive,
            "empty_label_pairs": empty,
            "full_label_pairs": full,
            "decoded_pair_duplicate_groups": duplicate_pair_groups,
            "image_modes": dict(image_modes),
            "pair_pixel_sha256": pair_digests,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = cfg["splits"][args.split]
    report = audit_split(
        args.data_root / f"{args.split}.zip",
        args.split,
        int(spec["size_bytes"]),
        spec["sha256"],
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "split": args.split,
                "pairs": report["pairs"],
                "label_values": report["label_values"],
                "empty_labels": report["empty_label_pairs"],
                "duplicate_pairs": report["decoded_pair_duplicate_groups"],
            }
        )
    )


if __name__ == "__main__":
    main()
