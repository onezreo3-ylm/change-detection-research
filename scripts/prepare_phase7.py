"""Prepare the approved metadata-only Phase 7 source split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcdc.phase7_protocol import load_phase7_config, prepare_phase7_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_phase7_config(args.config, require_approval=False)
    result = prepare_phase7_protocol(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
