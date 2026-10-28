"""Sealed V1R2 wrapper: correct one-shot iterable materialization only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from scripts import audit_final_test_b1_v1r as core


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_test_b1_v1r2_audit.yaml"
DIR = ROOT / "runs/final_test_b1_v1r2_protocol"
SOURCE = ROOT / "scripts/audit_final_test_b1_v1r2.py"
TEST_SOURCE = ROOT / "tests/test_audit_final_test_b1_v1r2.py"
V1R_FAILURE = ROOT / "runs/final_test_b1_v1r_protocol/failure_audit.json"
V1R_SEAL = ROOT / "runs/final_test_b1_v1r_protocol/preaudit_seal.json"


def corrected_add_counts(values: Iterable[dict[str, int]]) -> dict[str, int]:
    """The sole V1R correction: consume a one-shot iterable exactly once."""
    rows = tuple(values)
    return {key: sum(row[key] for row in rows) for key in core.COUNT_KEYS}


def configure() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    core.require(isinstance(config, dict) and config["schema_version"] == 1 and
                 config["name"] == "levir_final_test_b1_seed42_v1r2_audit_only" and
                 config["status"] == "draft_before_audit_seal" and
                 config["protocol_document_sha256"] ==
                 core.sha((CONFIG.parent / config["protocol_document"]).resolve()),
                 "V1R2 protocol/config invalid")
    for key, expected in (("expected_sources", 128), ("expected_tiles", 2048),
                          ("expected_conditions", 45), ("expected_instances", 92160),
                          ("test_seed", 310944), ("bootstrap_seed", 20260926),
                          ("bootstrap_replicates", 5000)):
        core.require(type(config[key]) is int and config[key] == expected,
                     f"V1R2 frozen setting differs: {key}")
    core.CONFIG = CONFIG
    core.PROTOCOL_DIR = DIR
    core.add_counts = corrected_add_counts
    return config


def check_extra_approval(config: dict[str, Any]) -> None:
    approval = yaml.safe_load(core.path_for(config, "approval").read_text(encoding="utf-8"))
    for key, path in (("v1r2_wrapper_source_sha256", SOURCE),
                      ("v1r2_regression_test_sha256", TEST_SOURCE),
                      ("failed_v1r_audit_sha256", V1R_FAILURE),
                      ("failed_v1r_seal_sha256", V1R_SEAL)):
        core.require(approval[key] == core.sha(path), f"V1R2 extra approval differs: {key}")
    failed = core.json_file(V1R_FAILURE)
    core.require(failed["status"] == "auditor_implementation_failed_before_certification" and
                 failed["recovered_scores_certified"] is False and
                 failed["scores_reported_or_used_for_selection"] is False and
                 failed["preaudit_seal_sha256"] == core.sha(V1R_SEAL),
                 "Failed V1R audit binding differs")


def synthetic_smoke() -> dict[str, Any]:
    core.synthetic_smoke()
    rows = ({"tn": value, "fp": value + 1, "fn": value + 2, "tp": value + 3}
            for value in (1, 2))
    core.require(corrected_add_counts(rows) == {"tn": 3, "fp": 5, "fn": 7, "tp": 9},
                 "Generator materialization regression failed")
    return {"status": "passed", "synthetic_only": True, "raw_test_access": False,
            "model_inference": False, "correction": "materialize_counts_iterable_once"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "seal", "run", "synthetic-smoke"))
    args = parser.parse_args()
    config = configure()
    if args.action == "synthetic-smoke":
        result = synthetic_smoke()
        core.write_json_new(DIR / "synthetic_smoke.json", result)
    elif args.action == "snapshot":
        result = core.snapshot(config)
    elif args.action == "seal":
        check_extra_approval(config)
        result = core.create_seal(config)
    else:
        check_extra_approval(config)
        result = core.certify(config)
        certificate = core.path_for(config, "output_dir") / "recovery_certificate.json"
        core.write_json_new(certificate.parent / "v1r2_correction_addendum.json", {
            "schema_version": 1, "status": "v1r2_certified_addendum",
            "correction": "materialize_counts_iterable_once",
            "original_v1_valid": False, "failed_v1r_valid": False,
            "new_test_inference": False, "test_based_selection": False,
            "wrapper_source_sha256": core.sha(SOURCE),
            "regression_test_sha256": core.sha(TEST_SOURCE),
            "failed_v1r_audit_sha256": core.sha(V1R_FAILURE),
            "failed_v1r_seal_sha256": core.sha(V1R_SEAL),
            "v1r2_seal_sha256": core.sha(core.path_for(config, "seal")),
            "recovery_certificate_sha256": core.sha(certificate)})
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
