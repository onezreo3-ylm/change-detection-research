"""Materialize only verified ValaisCD train/val archives for the existing loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import yaml


CATEGORIES = {"2017": "A", "2023": "B", "labels": "label"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_one(archive: Path, split: str, expected_pairs: int, destination: Path):
    counts = {category: 0 for category in CATEGORIES}
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        accepted = []
        for member in members:
            parts = member.filename.split("/")
            if member.is_dir() or member.filename.endswith(".DS_Store"):
                continue
            if re.fullmatch(rf"{split}/test_imgs_filtered_[0-9]+_2\.pkl", member.filename):
                # Author metadata is untrusted pickle and is not needed here.
                continue
            if (
                len(parts) != 3
                or parts[0] != split
                or parts[1] not in CATEGORIES
                or not parts[2].endswith(".tif")
                or not parts[2][:-4].isdigit()
            ):
                raise RuntimeError(f"Unexpected ZIP member: {member.filename}")
            accepted.append((member, parts[1], parts[2]))
            counts[parts[1]] += 1
        if any(count != expected_pairs for count in counts.values()):
            raise RuntimeError(f"Unexpected pair inventory in {archive}: {counts}")
        for member, category, name in accepted:
            folder = destination / split / CATEGORIES[category]
            folder.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, (folder / name).open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    return counts


def run(config_path: Path, archive_root: Path, output_root: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if output_root.exists():
        raise FileExistsError(output_root)
    temp_root = output_root.with_name(output_root.name + ".partial")
    if temp_root.exists():
        raise FileExistsError(temp_root)
    for split in ("train", "val"):
        spec = cfg["splits"][split]
        archive = archive_root / f"{split}.zip"
        if not archive.is_file() or archive.stat().st_size != spec["size_bytes"]:
            raise RuntimeError(f"Missing or wrong-size {split} archive")
        if sha256_file(archive) != spec["sha256"]:
            raise RuntimeError(f"Archive SHA256 mismatch: {split}")
    temp_root.mkdir(parents=True)
    reports = {}
    for split in ("train", "val"):
        spec = cfg["splits"][split]
        counts = _extract_one(
            archive_root / f"{split}.zip",
            split,
            int(spec["expected_pairs"]),
            temp_root,
        )
        reports[split] = counts
    report = {
        "status": "development_data_materialized",
        "official_test_archive_accessed": False,
        "source_revision": cfg["dataset_revision"],
        "source_archive_sha256": {
            split: cfg["splits"][split]["sha256"] for split in ("train", "val")
        },
        "inventory": reports,
        "label_encoding": "uint8 0/1; verified separately by local audit",
    }
    (temp_root / "materialization.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    temp_root.replace(output_root)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.config.resolve(), args.archive_root.resolve(), args.output_root.resolve()
    )
    print(json.dumps(report["inventory"]))


if __name__ == "__main__":
    main()
