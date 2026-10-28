"""Create the immutable Phase 7 pre-validation seal after all selector runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcdc.phase7_evaluation import create_preval_seal
from pcdc.phase7_protocol import load_phase7_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = create_preval_seal(load_phase7_config(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
