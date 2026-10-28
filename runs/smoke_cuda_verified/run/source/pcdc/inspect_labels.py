"""Read-only inventory of raw mask encodings. Never repairs or thresholds labels."""

from typing import Any

import numpy as np
from PIL import Image

from .config import Config
from .data import build_datasets, file_sha256


def inspect_labels(config: Config) -> dict[str, Any]:
    reports = {}
    for split, dataset in build_datasets(config, ("train", "val", "test")).items():
        aggregate = np.zeros(256, dtype=np.int64)
        invalid = []
        for record in dataset.records:
            with Image.open(record.mask) as image:
                mode, mask = image.mode, np.array(image)
            if mask.ndim == 3:
                if mask.shape[2] != 3 or not np.all(mask == mask[..., :1]):
                    invalid.append({"name": record.name, "reason": "disagreeing channels", "mode": mode})
                    continue
                mask = mask[..., 0]
            if mask.dtype != np.uint8 and mask.dtype != bool:
                invalid.append({"name": record.name, "reason": "unsupported dtype", "mode": mode})
                continue
            histogram = np.bincount(mask.reshape(-1).astype(np.int64), minlength=256)
            aggregate += histogram
            unknown = {str(value): int(histogram[value]) for value in np.flatnonzero(histogram)
                       if value not in config.values["data"]["label_values"]}
            if unknown:
                invalid.append({"name": record.name, "unexpected_values": unknown,
                                "mode": mode, "sha256": file_sha256(record.mask)})
        reports[split] = {"source_pairs": len(dataset.records),
                          "histogram": {str(value): int(aggregate[value]) for value in np.flatnonzero(aggregate)},
                          "invalid_masks": invalid}
    return {"operation": "read_only_label_encoding_inspection", "data_root": str(config.data_root),
            "expected_values": config.values["data"]["label_values"], "splits": reports,
            "decision": "No label remapping performed; research training requires a passing full audit."}
