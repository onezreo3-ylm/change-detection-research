"""Standalone clean-val replay for a completed independent numeric checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

from pcdc.data import audit_data, build_datasets, file_sha256, make_loader
from pcdc.engine import _digest, move_model, run_epoch, source_hashes
from pcdc.losses import ChangeDetectionLoss
from pcdc.model import build_model
from pcdc.runtime import environment_record, setup_runtime, write_json
from scripts.train_valais_fp32val import (
    _read_checkpoint, _verify_snapshot, _wait_guard, load_study,
)


def verify_guarded_completion(run_dir: Path, fingerprint: str) -> dict:
    marker = run_dir / "guard_completion.json"
    if not marker.is_file():
        raise RuntimeError("Durable guarded completion marker is missing")
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if (completion.get("status") != "guarded_independent_clean_completed"
            or completion.get("test_accessed") is not False
            or completion.get("study_fingerprint") != fingerprint):
        raise RuntimeError("Guarded completion identity differs")
    paths = {
        "status_sha256": run_dir / "status.json",
        "best_checkpoint_sha256": run_dir / "best.pt",
        "last_checkpoint_sha256": run_dir / "last.pt",
        "guard_state_sha256": Path(completion["guard_state_path"]).resolve(),
        "preflight_seal_sha256": Path(completion["preflight_seal_path"]).resolve(),
    }
    match = re.fullmatch(
        re.escape(run_dir.name) + r"(?:\.(session_[0-9]{4}))?\.preflight_seal\.json",
        paths["preflight_seal_sha256"].name,
    )
    if (not paths["guard_state_sha256"].is_relative_to(run_dir.resolve())
            or not paths["preflight_seal_sha256"].is_relative_to(run_dir.parent.resolve())
            or match is None):
        raise RuntimeError("Completion references an unexpected guard or seal path")
    expected_state = (
        run_dir / "temperature_guard_state.json" if match.group(1) is None
        else run_dir / "temperature_guard_sessions" / match.group(1)
             / "temperature_guard_state.json"
    )
    if paths["guard_state_sha256"] != expected_state.resolve():
        raise RuntimeError("Guard state does not belong to the final launch session")
    for field, path in paths.items():
        if not path.is_file() or file_sha256(path) != completion.get(field):
            raise RuntimeError(f"Guarded completion hash differs: {field}")
    guard = json.loads(paths["guard_state_sha256"].read_text(encoding="utf-8"))
    seal = json.loads(paths["preflight_seal_sha256"].read_text(encoding="utf-8"))
    if (guard.get("status") != "training_process_exited"
            or seal.get("status") != "independent_numeric_protocol_preflight_sealed"
            or seal.get("fingerprint") != fingerprint
            or seal.get("test_accessed") is not False):
        raise RuntimeError("Guard or preflight seal was not completed")
    bindings = seal.get("bound_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("Preflight seal has no source/input bindings")
    for name, expected in bindings.items():
        source = Path(name).resolve()
        if not source.is_file() or file_sha256(source) != expected:
            raise RuntimeError(f"Preflight-bound source/input changed: {name}")
    return completion


def evaluate(spec_path: Path, checkpoint_path: Path, output: Path,
             guard_handshake: Path):
    if output.exists():
        raise FileExistsError(output)
    spec, paths, base, fingerprint = load_study(spec_path)
    run_dir = paths["output_dir"]
    if not run_dir.is_dir() or checkpoint_path.resolve() != run_dir / "best.pt":
        raise RuntimeError("Only the selected independent best.pt is allowed")
    if (output.resolve() != run_dir / "val_independent.json"
            or guard_handshake.resolve() != run_dir / "validation_guard" / "guard_attached.json"):
        raise RuntimeError("Independent validation requires the fixed guarded output/handshake")
    verify_guarded_completion(run_dir, fingerprint)
    snapshot = json.loads((run_dir / "source_sha256.json").read_text(encoding="utf-8"))
    _verify_snapshot(run_dir, snapshot)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or status.get("test_evaluated") is not False
            or status.get("training_amp") is not True
            or status.get("validation_amp") is not False):
        raise RuntimeError("Independent run is not a completed val-only study")
    checkpoint = _read_checkpoint(checkpoint_path)
    if (checkpoint["study_fingerprint"] != fingerprint
            or checkpoint["validation_amp"] is not False
            or checkpoint["test_evaluated"] is not False
            or checkpoint["best_epoch"] != checkpoint["epoch"]):
        raise RuntimeError("Best checkpoint differs from the independent numeric protocol")
    device = setup_runtime(base)
    audit = audit_data(base, full=True)
    if _digest(audit) != checkpoint["data_audit_digest"]:
        raise RuntimeError("Validation data differs from checkpoint audit")
    if source_hashes() != {
        key: value for key, value in snapshot.items() if key.startswith("pcdc/")
    }:
        raise RuntimeError("Frozen pcdc source changed")
    dataset = build_datasets(base, ("val",))["val"]
    model = move_model(
        build_model(base, initialize_encoder=False), device,
        base.values["runtime"]["memory_format"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    _wait_guard(guard_handshake, fingerprint)
    result = run_epoch(
        model, make_loader(dataset, base, training=False),
        ChangeDetectionLoss(**base.values["loss"]), device,
        threshold=base.values["evaluation"]["threshold"],
        amp=False, memory_format=base.values["runtime"]["memory_format"],
        progress="independent_val_fp32",
    )
    if result["metrics"]["f1"] != checkpoint["best_val_f1"]:
        raise RuntimeError("Standalone clean-val F1 does not reproduce checkpoint selection")
    payload = {
        "status": "independent_validation_reproduced",
        "split": "val", "test_accessed": False,
        "selection": "clean_val_f1_at_threshold_0_5",
        "training_amp": True, "validation_amp": False,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "study_fingerprint": fingerprint,
        "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment_record(base, device),
        **result,
    }
    write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-handshake", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.config.resolve(), args.checkpoint.resolve(), args.output.resolve(),
        args.guard_handshake.resolve(),
    )
    print(json.dumps({"status": result["status"], "f1": result["metrics"]["f1"],
                      "test_accessed": False}))


if __name__ == "__main__":
    main()
