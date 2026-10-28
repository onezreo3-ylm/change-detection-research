"""Metadata-only ValaisCD split audit plus two dev-region pixel samples.

Official test data and model weights are never requested by this entry point.
The output is a candidate data audit, not an experimental robustness result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from scripts.audit_whu_candidate_remote import (
    ZipEntry,
    fetch_member,
    list_remote_zip,
    remote_identity,
)


def split_members(entries: dict[str, ZipEntry], split: str) -> dict[str, dict[str, ZipEntry]]:
    groups = {}
    for category in ("2017", "2023", "labels"):
        prefix = f"{split}/{category}/"
        group = {
            name[len(prefix) : -4]: entry
            for name, entry in entries.items()
            if name.startswith(prefix) and name.endswith(".tif")
        }
        if not group:
            raise RuntimeError(f"No {category} TIFFs in {split}")
        groups[category] = group
    if set(groups["2017"]) != set(groups["2023"]) or set(groups["2017"]) != set(
        groups["labels"]
    ):
        raise RuntimeError(f"Unpaired {split} TIFF identifiers")
    return groups


def pixel_sample(url: str, groups: dict[str, dict[str, ZipEntry]], identifier: int):
    images = {}
    hashes = {}
    for category in ("2017", "2023", "labels"):
        entry = groups[category].get(str(identifier))
        if entry is None:
            raise RuntimeError(f"Missing dev sample {category}/{identifier}")
        data = fetch_member(url, entry)
        hashes[category] = hashlib.sha256(data).hexdigest()
        images[category] = Image.open(io.BytesIO(data))
        if images[category].size != (256, 256):
            raise RuntimeError("Expected 256x256 ValaisCD patches")
    if images["2017"].mode != "RGB" or images["2023"].mode != "RGB":
        raise RuntimeError("Expected RGB bi-temporal images")
    target = np.asarray(images["labels"])
    unique_values = np.unique(target).tolist()
    if not set(unique_values).issubset({0, 1}):
        raise RuntimeError(f"Expected 0/1 change labels, got {unique_values}")
    canvas = Image.new("RGB", (768, 256))
    canvas.paste(images["2017"], (0, 0))
    canvas.paste(images["2023"], (256, 0))
    canvas.paste(Image.fromarray(target.astype(np.uint8) * 255), (512, 0))
    return (
        {
            "id": identifier,
            "size": [256, 256],
            "label_values": unique_values,
            "positive_pixels": int((target == 1).sum()),
            "member_sha256": hashes,
            "preview": f"dev_{identifier}_qa.png",
        },
        canvas,
    )


def run(config_path: Path):
    raw_config = config_path.read_bytes()
    cfg = yaml.safe_load(raw_config)
    if cfg.get("kind") != "external_dataset_candidate_preflight":
        raise RuntimeError("Wrong preflight config kind")
    if cfg.get("rights_status") != "upstream_attribution_and_derived_terms_review_needed":
        raise RuntimeError("Candidate rights status must remain explicit")
    output_dir = (config_path.parent / cfg["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    splits = cfg["splits"]
    if set(splits) != {"dev", "train", "val"}:
        raise RuntimeError("This preflight must not request the official test archive")

    inventories = {}
    image_identity = {}
    dev_groups = None
    for split in ("dev", "train", "val"):
        spec = splits[split]
        url = spec["url"]
        size, etag = remote_identity(url)
        if size != spec["size_bytes"] or etag != spec["etag"]:
            raise RuntimeError(f"Remote {split} archive identity changed")
        entries = list_remote_zip(url, size, int(cfg["tail_bytes"]))
        groups = split_members(entries, split)
        if len(groups["2017"]) != spec["expected_pairs"]:
            raise RuntimeError(f"Unexpected {split} pair count")
        inventories[split] = {
            "archive_size_bytes": size,
            "etag": etag,
            "zip_entries": len(entries),
            "paired_tiff_ids": len(groups["2017"]),
            "nonempty_categories": list(groups),
        }
        for category in ("2017", "2023"):
            for identifier, entry in groups[category].items():
                fingerprint = (entry.crc32, entry.size)
                image_identity.setdefault(fingerprint, []).append(
                    f"{split}/{category}/{identifier}"
                )
        if split == "dev":
            dev_groups = groups

    cross_split_same_crc_size = 0
    for locations in image_identity.values():
        if len({location.split("/")[0] for location in locations}) > 1:
            cross_split_same_crc_size += 1
    samples = []
    previews = []
    dev_url = splits["dev"]["url"]
    for identifier in cfg["sample_dev_ids"]:
        sample, preview = pixel_sample(dev_url, dev_groups, int(identifier))
        samples.append(sample)
        previews.append((sample["preview"], preview))

    report = {
        "status": "candidate_preflight_only",
        "model_training_or_inference": False,
        "official_test_archive_requested": False,
        "official_test_pixels_accessed": False,
        "rights_status": cfg["rights_status"],
        "dataset_card": cfg["dataset_card"],
        "dataset_revision": cfg["dataset_revision"],
        "upstream_terms": cfg["upstream_terms"],
        "upstream_attribution": cfg["upstream_attribution"],
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "inventories": inventories,
        "cross_split_same_image_crc_size_groups": cross_split_same_crc_size,
        "crc_size_screen_is_not_cryptographic_duplicate_audit": True,
        "dev_samples": samples,
    }
    output_dir.mkdir(parents=True)
    for filename, preview in previews:
        preview.save(output_dir / filename)
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output_dir),
                "split_pairs": {
                    split: item["paired_tiff_ids"] for split, item in inventories.items()
                },
                "official_test_pixels": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
