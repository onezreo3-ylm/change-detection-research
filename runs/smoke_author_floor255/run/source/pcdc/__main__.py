"""Run `python -m pcdc --help` for explicit Phase 1 entry points."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import audit_data
from .engine import evaluate, train
from .inspect_labels import inspect_labels
from .runtime import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 clean LEVIR-CD baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "inspect-labels", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        if command in ("audit", "inspect-labels"):
            if command == "audit":
                sub.add_argument("--full", action="store_true", help="Decode all pairs and hash all content")
            sub.add_argument("--output", type=Path, required=True)
        elif command == "train":
            sub.add_argument("--resume", action="store_true", help="Resume last.pt under identical settings and source")
        else:
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--split", choices=("val", "test"), required=True)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "audit":
        result = audit_data(config, full=args.full)
        write_json(args.output, result)
        result = {"audit_level": result["audit_level"], "output": str(args.output),
                  "splits": {k: {field: value[field] for field in ("source_pairs", "tiles", "inventory_sha256")}
                             for k, value in result["splits"].items()}}
    elif args.command == "inspect-labels":
        result = inspect_labels(config)
        write_json(args.output, result)
    elif args.command == "train":
        result = train(config, resume=args.resume)
    else:
        result = evaluate(config, args.checkpoint, args.split, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
