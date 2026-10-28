"""Train-only remote metadata and label preflight for the WHU-CD candidate.

This does not load official test pixels, train a model, or certify a dataset.
It requests ZIP directory bytes and selected train members via HTTP ranges.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import urllib.request
import zlib

import numpy as np
from PIL import Image
import yaml


ROOT = "Building change detection dataset_add/1. The two-period image data/"
MAX_MEMBER_BYTES = 16_000_000
MAX_IMAGE_PIXELS = 400_000_000


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    local_offset: int


def _request(url: str, *, method: str = "GET", range_header: str | None = None):
    headers = {"User-Agent": "Mozilla/5.0 (WHU research preflight)"}
    if range_header is not None:
        headers["Range"] = range_header
    request = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(request, timeout=90)


def read_range(url: str, first: int, last: int) -> bytes:
    if first < 0 or last < first:
        raise ValueError("Invalid HTTP byte range")
    with _request(url, range_header=f"bytes={first}-{last}") as response:
        expected = f"bytes {first}-{last}/"
        actual = response.headers.get("Content-Range", "")
        if response.status != 206 or not actual.startswith(expected):
            raise RuntimeError(f"Unexpected range response: {response.status}, {actual}")
        payload = response.read()
    if len(payload) != last - first + 1:
        raise RuntimeError("Incomplete HTTP range payload")
    return payload


def remote_identity(url: str) -> tuple[int, str]:
    with _request(url, method="HEAD") as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HEAD status: {response.status}")
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag", "")
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("Source does not advertise byte ranges")
    return size, etag


def _zip64_offset(extra: bytes, size: int, compressed_size: int) -> int:
    position = 0
    while position < len(extra):
        kind, length = struct.unpack_from("<HH", extra, position)
        position += 4
        if kind == 1:
            position += 8 * (size == 0xFFFFFFFF)
            position += 8 * (compressed_size == 0xFFFFFFFF)
            return struct.unpack_from("<Q", extra, position)[0]
        position += length
    raise RuntimeError("Missing ZIP64 local offset")


def list_remote_zip(url: str, size: int, tail_bytes: int) -> dict[str, ZipEntry]:
    tail_start = size - tail_bytes
    tail = read_range(url, tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("Missing ZIP end directory")
    entry_count = struct.unpack_from("<H", tail, eocd + 10)[0]
    central_size = struct.unpack_from("<I", tail, eocd + 12)[0]
    central_start = struct.unpack_from("<I", tail, eocd + 16)[0]
    if entry_count == 0xFFFF or central_start == 0xFFFFFFFF:
        if eocd < 20 or tail[eocd - 20 : eocd - 16] != b"PK\x06\x07":
            raise RuntimeError("Expected ZIP64 end directory and locator")
        zip64_absolute = struct.unpack_from("<Q", tail, eocd - 12)[0]
        zip64_position = zip64_absolute - tail_start
        if zip64_position < 0 or tail[zip64_position : zip64_position + 4] != b"PK\x06\x06":
            raise RuntimeError("ZIP64 directory is outside the requested tail")
        entry_count = struct.unpack_from("<Q", tail, zip64_position + 32)[0]
        central_size = struct.unpack_from("<Q", tail, zip64_position + 40)[0]
        central_start = struct.unpack_from("<Q", tail, zip64_position + 48)[0]
    if central_start >= tail_start and central_start + central_size <= size:
        central = tail[central_start - tail_start : central_start - tail_start + central_size]
    else:
        central = read_range(url, central_start, central_start + central_size - 1)
    result: dict[str, ZipEntry] = {}
    position = 0
    for _ in range(entry_count):
        if central[position : position + 4] != b"PK\x01\x02":
            raise RuntimeError("Bad ZIP central entry signature")
        method = struct.unpack_from("<H", central, position + 10)[0]
        crc, compressed_size, uncompressed_size = struct.unpack_from(
            "<III", central, position + 16
        )
        name_length, extra_length, comment_length = struct.unpack_from(
            "<HHH", central, position + 28
        )
        local_offset = struct.unpack_from("<I", central, position + 42)[0]
        name_start = position + 46
        name = central[name_start : name_start + name_length].decode("utf-8")
        extra = central[
            name_start + name_length : name_start + name_length + extra_length
        ]
        if local_offset == 0xFFFFFFFF:
            local_offset = _zip64_offset(extra, uncompressed_size, compressed_size)
        if name in result:
            raise RuntimeError(f"Duplicate ZIP member: {name}")
        result[name] = ZipEntry(
            name, method, crc, compressed_size, uncompressed_size, local_offset
        )
        position = name_start + name_length + extra_length + comment_length
    if position != len(central):
        raise RuntimeError("ZIP central directory length mismatch")
    return result


def fetch_member(url: str, entry: ZipEntry) -> bytes:
    if entry.size > MAX_MEMBER_BYTES or entry.compressed_size > MAX_MEMBER_BYTES:
        raise RuntimeError(f"Member exceeds preflight size limit: {entry.name}")
    header = read_range(url, entry.local_offset, entry.local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local ZIP entry: {entry.name}")
    name_length, extra_length = struct.unpack_from("<HH", header, 26)
    payload_start = entry.local_offset + 30 + name_length + extra_length
    compressed = read_range(
        url, payload_start, payload_start + entry.compressed_size - 1
    )
    if entry.method == 8:
        data = zlib.decompress(compressed, -15)
    elif entry.method == 0:
        data = compressed
    else:
        raise RuntimeError(f"Unsupported ZIP method: {entry.method}")
    if len(data) != entry.size or zlib.crc32(data) != entry.crc32:
        raise RuntimeError(f"ZIP member integrity failure: {entry.name}")
    return data


def tile_position(index: int, columns: int) -> tuple[int, int]:
    if index < 0 or columns <= 0:
        raise ValueError("Invalid tile position")
    return divmod(index, columns)


def target_semantics(mask_2012: np.ndarray, mask_2016: np.ndarray, target: np.ndarray):
    if mask_2012.shape != mask_2016.shape or target.shape != mask_2012.shape:
        raise ValueError("Masks must have equal shapes")
    old = mask_2012.astype(bool)
    new = mask_2016.astype(bool)
    y = target.astype(bool)
    added = ~old & new
    removed = old & ~new
    unchanged = old == new
    return {
        "target_positive": int(y.sum()),
        "year_xor_positive": int((added | removed).sum()),
        "added_positive": int(added.sum()),
        "removed_positive": int(removed.sum()),
        "target_on_added": int((y & added).sum()),
        "target_on_removed": int((y & removed).sum()),
        "target_on_year_unchanged": int((y & unchanged).sum()),
        "target_equals_year_xor": bool(np.array_equal(y, added | removed)),
    }


def _image(data: bytes) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    return Image.open(io.BytesIO(data))


def _member_path(suffix: str) -> str:
    return ROOT + suffix


def _need(entries: dict[str, ZipEntry], suffix: str) -> ZipEntry:
    path = _member_path(suffix)
    if path not in entries:
        raise RuntimeError(f"Missing required member: {path}")
    return entries[path]


def _count(entries: dict[str, ZipEntry], suffix: str) -> int:
    pattern = re.compile(re.escape(_member_path(suffix)) + r"0_[0-9]+\.tif$")
    return sum(bool(pattern.fullmatch(name)) for name in entries)


def _preview(a: Image.Image, b: Image.Image, target: np.ndarray) -> Image.Image:
    if a.size != b.size or a.size != target.shape[::-1]:
        raise RuntimeError("Preview dimensions differ")
    width, height = a.size
    canvas = Image.new("RGB", (width * 3, height))
    canvas.paste(a.convert("RGB"), (0, 0))
    canvas.paste(b.convert("RGB"), (width, 0))
    canvas.paste(Image.fromarray((target.astype(np.uint8) * 255), mode="L"), (width * 2, 0))
    return canvas


def run(config_path: Path):
    raw = config_path.read_bytes()
    cfg = yaml.safe_load(raw)
    if cfg.get("kind") != "external_dataset_candidate_preflight":
        raise RuntimeError("Wrong preflight config kind")
    if cfg.get("license_status") != "unverified":
        raise RuntimeError("This preflight does not certify licensing")
    url = str(cfg["source_url"])
    output_dir = (config_path.parent / cfg["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    size, etag = remote_identity(url)
    if size != cfg["expected_archive_bytes"] or etag != cfg["expected_etag"]:
        raise RuntimeError("Remote archive identity changed")
    entries = list_remote_zip(url, size, int(cfg["tail_bytes"]))
    inventory = {}
    for year in (2012, 2016):
        for split in ("train", "test"):
            for kind in ("image", "label"):
                suffix = f"{year}/splited_images/{split}/{kind}/"
                count = _count(entries, suffix)
                expected = cfg[f"expected_{split}_tiles"]
                if count != expected:
                    raise RuntimeError(f"Unexpected {suffix} count: {count}")
                inventory[f"{year}/{split}/{kind}"] = count

    target_entry = _need(entries, "change_label/train/change_label.tif")
    target_bytes = fetch_member(url, target_entry)
    target_image = _image(target_bytes)
    if target_image.mode != "1":
        raise RuntimeError("Expected binary official train change mask")
    tile_size = int(cfg["tile_size"])
    columns = (target_image.width + tile_size - 1) // tile_size
    rows = (target_image.height + tile_size - 1) // tile_size
    if columns * rows != cfg["expected_train_tiles"]:
        raise RuntimeError("Train mosaic dimensions do not match tile count")
    full_masks = {}
    for year in (2012, 2016):
        entry = _need(
            entries, f"{year}/whole_image/train/label/{year}_train.tif"
        )
        full_masks[year] = _image(fetch_member(url, entry))
        if full_masks[year].size != target_image.size:
            raise RuntimeError("Year mask and change mask dimensions differ")

    sample_results = []
    previews = []
    for index in cfg["sample_tile_indices"]:
        row, column = tile_position(int(index), columns)
        if row >= rows - 1 or column >= columns - 1:
            raise RuntimeError("Preflight samples must avoid partial border tiles")
        identifier = f"0_{index}.tif"
        box = (
            column * tile_size,
            row * tile_size,
            (column + 1) * tile_size,
            (row + 1) * tile_size,
        )
        images = {}
        year_labels = {}
        hashes = {}
        for year in (2012, 2016):
            for kind in ("image", "label"):
                suffix = f"{year}/splited_images/train/{kind}/{identifier}"
                data = fetch_member(url, _need(entries, suffix))
                hashes[f"{year}/{kind}"] = hashlib.sha256(data).hexdigest()
                image = _image(data)
                if image.size != (tile_size, tile_size):
                    raise RuntimeError("Unexpected sample tile size")
                if kind == "image":
                    if image.mode != "RGB":
                        raise RuntimeError("Expected RGB train tile")
                    images[year] = image
                else:
                    values = np.asarray(image)
                    if not np.isin(values, [0, 255]).all():
                        raise RuntimeError("Expected 0/255 year label tile")
                    mask = values > 0
                    full_crop = np.asarray(full_masks[year].crop(box)) > 0
                    if not np.array_equal(mask, full_crop):
                        raise RuntimeError(f"Tile/full-label mismatch: {year}/{index}")
                    year_labels[year] = mask
        target = np.asarray(target_image.crop(box)) > 0
        stats = target_semantics(year_labels[2012], year_labels[2016], target)
        preview_name = f"train_{identifier[:-4]}_qa.png"
        previews.append((preview_name, _preview(images[2012], images[2016], target)))
        sample_results.append(
            {
                "index": index,
                "mosaic_row": row,
                "mosaic_column": column,
                "member_sha256": hashes,
                "target_semantics": stats,
                "preview": preview_name,
                "year_label_full_mosaic_match": True,
            }
        )

    report = {
        "status": "candidate_preflight_only",
        "official_test_pixels_accessed": False,
        "model_training_or_inference": False,
        "license_status": "unverified",
        "source_url": url,
        "source_page": cfg["source_page"],
        "source_size_bytes": size,
        "source_etag": etag,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "zip_entry_count": len(entries),
        "inventory": inventory,
        "train_mosaic_size": list(target_image.size),
        "train_grid_rows": rows,
        "train_grid_columns": columns,
        "official_train_change_label_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "samples": sample_results,
    }
    output_dir.mkdir(parents=True)
    for name, preview in previews:
        preview.save(output_dir / name)
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir),
                      "samples": len(sample_results), "test_pixels": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
