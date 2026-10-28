"""Verify frozen LEVIR-CD research decisions without raw data or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/research_closeout_20260927.yaml"
RECORD = ROOT / "runs/research_closeout_20260927/verification.json"
SHA = re.compile(r"[0-9a-f]{64}\Z")
STATIC_NAMES = (
    "original_protocol_document", "original_config", "original_approval",
    "original_pretest_seal", "original_evaluator", "original_guard_config",
    "original_corruption_config", "original_checkpoint", "failure_audit",
    "original_stdout", "original_stderr",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: UniqueLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        require(isinstance(key, str) and key not in result, "Duplicate/nonstring YAML key")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    require(isinstance(value, dict), f"Invalid YAML mapping: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {path}/{key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"Nonfinite JSON value: {token}")))
    require(isinstance(value, dict), f"Invalid JSON mapping: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(base: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative and
            not PurePosixPath(relative).is_absolute() and
            all(part not in (".", "..", "") for part in PurePosixPath(relative).parts),
            f"Unsafe path: {relative}")
    base_resolved = base.resolve(strict=True)
    target = base / Path(*PurePosixPath(relative).parts)
    require(target.is_file() and not target.is_symlink(), f"Missing/symlinked artifact: {relative}")
    require(target.resolve(strict=True).is_relative_to(base_resolved),
            f"Artifact escapes root: {relative}")
    current = target.parent
    while current != base:
        require(not current.is_symlink(), f"Symlinked ancestor: {relative}")
        current = current.parent
    return target


def verify_files(base: Path, artifacts: Mapping[str, str]) -> int:
    require(isinstance(artifacts, dict) and bool(artifacts), "Empty artifact manifest")
    for relative, expected in artifacts.items():
        require(isinstance(expected, str) and SHA.fullmatch(expected) is not None,
                f"Invalid SHA256: {relative}")
        require(sha(safe_path(base, relative)) == expected, f"Artifact hash differs: {relative}")
    return len(artifacts)


def verify_certificate_inputs(root: Path, certificate: dict[str, Any],
                              recovery_config: dict[str, Any]) -> int:
    inputs = certificate["input_sha256"]
    require(isinstance(inputs, dict) and len(inputs) == 70, "Recovery input count differs")
    require(set(key.removeprefix("static/") for key in inputs if key.startswith("static/")) ==
            set(STATIC_NAMES), "Recovery static input set differs")
    original_output = (root / "configs" / recovery_config["original_output"]).resolve(strict=True)
    require(original_output.is_dir() and original_output.is_relative_to(root.resolve(strict=True)),
            "Original output path differs")
    require(not (original_output / "guard_completion.json").exists(),
            "Original V1 completion marker unexpectedly exists")
    for key, expected in inputs.items():
        require(isinstance(expected, str) and SHA.fullmatch(expected) is not None,
                f"Invalid recovery SHA256: {key}")
        if key.startswith("static/"):
            name = key.removeprefix("static/")
            relative = recovery_config[name]
            require(isinstance(relative, str) and bool(relative) and
                    not Path(relative).is_absolute(),
                    f"Static input path differs: {name}")
            path = (root / "configs" / relative).resolve(strict=True)
            require(path.is_file() and path.is_relative_to(root.resolve(strict=True)),
                    f"Static input escapes workspace: {name}")
        else:
            require(key.startswith("v1/"), f"Unknown recovery input: {key}")
            path = safe_path(original_output, key.removeprefix("v1/"))
        require(sha(path) == expected, f"Recovery input drift: {key}")
    return len(inputs)


def verify(root: Path = ROOT, manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    require(set(manifest) == {"schema_version", "name", "scope", "artifacts"} and
            manifest["schema_version"] == 1 and
            manifest["name"] == "levir_cd_research_closeout_20260927" and
            manifest["scope"] == "derived_artifacts_only_no_raw_data_or_inference",
            "Closeout manifest schema/scope differs")
    artifacts = manifest["artifacts"]
    require(len(artifacts) == 14 and
            all(name.startswith(("docs/", "runs/", "configs/")) for name in artifacts),
            "Closeout artifact set differs")
    verified = verify_files(root, artifacts)

    phase7 = read_json(root / "runs/levir_phase7_val_seed42/decision.json")
    r1 = read_json(root / "runs/r1e_protocol_20260926/analysis.json")
    r1_audit = read_json(root / "runs/r1e_protocol_20260926/completion_audit.json")
    final_audit = read_json(root / "runs/final_test_b1_v1r2_protocol/completion_audit.json")
    certificate = read_json(root / "runs/final_test_b1_v1r2_audit/recovery_certificate.json")
    addendum = read_json(root / "runs/final_test_b1_v1r2_audit/v1r2_correction_addendum.json")
    snapshot = read_json(root / "runs/final_test_b1_v1r2_protocol/input_snapshot.json")
    recovery_config = read_yaml(root / "configs/final_test_b1_v1r2_audit.yaml")
    original_status = read_json(root / "runs/levir_b1_final_test_v1_seed42/status.json")

    require(phase7["selector_feasibility"]["decision"] ==
            "stop_whole_tile_selector_route" and
            phase7["selector_feasibility"]["passing_policies"] == [] and
            phase7["automatic_next_phase_authorized"] is False and
            phase7["test_evaluated"] is False, "Phase 7 decision differs")
    require(r1["selection_gate"]["decision"] == "retain_b1" and
            r1["status"] == "completed" and r1["test_evaluated"] is False and
            r1_audit["held_out_test_closed"] is True and
            r1_audit["test_evaluated"] is False, "R1/R1E decision differs")
    require(final_audit["status"] == "complete_audit_recovered_result_only" and
            final_audit["r1_development_decision"] == "retain_b1" and
            final_audit["original_v1_valid"] is False and
            final_audit["original_v1_guard_completion_exists"] is False and
            final_audit["v1r2_recovered_scores_certified"] is True and
            final_audit["new_test_inference"] is False and
            final_audit["test_based_model_selection"] is False,
            "Final completion audit decision differs")
    require(certificate["status"] == "certified_recovered_scores" and
            certificate["recovery_type"] == "audit_only_post_scoring_bookkeeping_failure" and
            certificate["original_v1_valid"] is False and
            certificate["recovered_scores_certified"] is True and
            certificate["new_test_inference"] is False and
            certificate["original_guard_completion_exists"] is False and
            certificate["test_based_selection"] is False and
            certificate["source_pair_count"] == 128 and
            certificate["tile_count"] == 2048 and
            certificate["condition_count"] == 45 and
            certificate["instance_count"] == 92160,
            "Recovery certificate scientific boundary differs")
    require(original_status["status"] == "failed_incomplete_no_automatic_retry" and
            original_status["guarded_result_valid"] is False and
            addendum["original_v1_valid"] is False and
            addendum["failed_v1r_valid"] is False and
            addendum["new_test_inference"] is False,
            "Original V1/V1R invalid status differs")
    require(addendum["recovery_certificate_sha256"] ==
            sha(root / "runs/final_test_b1_v1r2_audit/recovery_certificate.json") and
            addendum["v1r2_seal_sha256"] ==
            sha(root / "runs/final_test_b1_v1r2_protocol/preaudit_seal.json") and
            certificate["input_snapshot_sha256"] ==
            sha(root / "runs/final_test_b1_v1r2_protocol/input_snapshot.json") and
            snapshot["input_sha256"] == certificate["input_sha256"] and
            snapshot["scores_interpreted"] is False,
            "Recovery certificate hash chain differs")
    bound = verify_certificate_inputs(root, certificate, recovery_config)
    return {"schema_version": 1, "status": "verified", "mode": "derived_artifacts_only",
            "raw_test_data_accessed": False, "new_model_inference": False,
            "manifest_sha256": sha(manifest_path), "manifest_artifact_count": verified,
            "recovery_input_count": bound,
            "decisions": {"phase7": "stop_whole_tile_selector_route",
                          "r1": "retain_b1", "final": "v1r2_audit_recovered_only"}}


def synthetic_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "fixture.json"
        artifact.write_text('{"synthetic":true}\n', encoding="utf-8")
        require(verify_files(root, {"fixture.json": sha(artifact)}) == 1,
                "Synthetic manifest verification failed")
        artifact.write_text('{"synthetic":false}\n', encoding="utf-8")
        try:
            verify_files(root, {"fixture.json": hashlib.sha256(b'{"synthetic":true}\n').hexdigest()})
        except ValueError:
            pass
        else:
            raise ValueError("Synthetic tampering was not rejected")
    return {"status": "passed", "synthetic_only": True,
            "raw_test_data_accessed": False, "new_model_inference": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "record", "synthetic-smoke"))
    args = parser.parse_args()
    result = synthetic_smoke() if args.action == "synthetic-smoke" else verify()
    if args.action == "record":
        require(not RECORD.exists(), "Refuse to overwrite closeout verification record")
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        with RECORD.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
