"""Wait for guarded attach before importing CUDA evaluation dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def wait_for_attach(path: Path, expected_fingerprint: str) -> None:
    deadline = time.monotonic() + 90
    while not path.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError("Independent validation guard did not attach")
        time.sleep(1)
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker != {"status": "guard_attached", "study_fingerprint": expected_fingerprint}:
        raise RuntimeError("Independent validation guard fingerprint differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-handshake", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    args = parser.parse_args()
    wait_for_attach(args.guard_handshake.resolve(), args.fingerprint)
    # Import after guard attachment so module imports cannot initialize CUDA early.
    from scripts.evaluate_valais_fp32val import evaluate

    result = evaluate(
        args.config.resolve(), args.checkpoint.resolve(),
        args.output.resolve(), args.guard_handshake.resolve(),
    )
    print(json.dumps({"status": result["status"], "test_accessed": False}))


if __name__ == "__main__":
    main()
