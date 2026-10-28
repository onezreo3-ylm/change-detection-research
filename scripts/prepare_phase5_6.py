"""Prepare or verify the approved Phase 5-6 train-only protocol artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcdc.phase5_protocol import (
    load_phase56_config,
    preflight,
    prepare_phase56_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        config = load_phase56_config(args.config, require_approval=False)
        result = prepare_phase56_protocol(config)
    else:
        config = load_phase56_config(args.config, require_approval=True)
        result = preflight(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
