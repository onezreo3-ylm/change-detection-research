"""Deliberately synthetic fixture and end-to-end engineering check, never research."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config, load_config
from .engine import evaluate, read_checkpoint, train
from .runtime import write_json


def create_fixture(config: Config) -> None:
    if config.values["experiment"]["kind"] != "smoke":
        raise ValueError("Synthetic generation requires experiment.kind: smoke")
    root = config.data_root
    if root.exists():
        raise FileExistsError(f"Will not replace data: {root}")
    size = config.values["data"]["expected_image_size"]
    counts = config.values["data"]["expected_pairs"]
    if size is None or size > 256 or any(n is None or n > 8 for n in counts.values()):
        raise ValueError("Smoke fixture must be explicitly small")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "SYNTHETIC_ONLY.json", {"kind": "smoke", "warning": "Not LEVIR-CD; no research conclusions"})
    rng = np.random.default_rng(config.values["experiment"]["seed"])
    settings = config.values["data"]
    for split, count in counts.items():
        if split == "test" and not config.run_test_after_training:
            continue
        for index in range(count):
            a = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            b = a.copy()
            mask = np.full((size, size), settings["label_values"][0], dtype=np.uint8)
            region = (slice(size // 4, size // 2), slice(size // 4, 3 * size // 4))
            b[region] = 255 - b[region]
            mask[region] = settings["label_values"][1]
            for key, array in (("image_a", a), ("image_b", b), ("mask", mask)):
                folder = root / settings["splits"][split] / settings["folders"][key]
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(folder / f"{split}_{index}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print("SYNTHETIC ENGINEERING CHECK ONLY - not research results", flush=True)
    create_fixture(config)
    result = train(config)
    if config.run_test_after_training:
        standalone = evaluate(config, config.output_dir / "best.pt", "test",
                              config.output_dir / "standalone_test.json")
        expected = result["test"]
    else:
        standalone = evaluate(config, config.output_dir / "best.pt", "val",
                              config.output_dir / "standalone_val.json")
        checkpoint = read_checkpoint(config.output_dir / "best.pt")
        expected = checkpoint["history"][-1]["val"]["metrics"]
    if expected != standalone["metrics"]:
        raise AssertionError("Standalone evaluation does not match its checkpoint-selected split")
    print(json.dumps({"smoke": "passed", "kind": "smoke", "output": str(config.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
