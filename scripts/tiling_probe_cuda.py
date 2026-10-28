"""The single fixed largest-window pilot; imported only by a protected worker."""
from pathlib import Path
import time

import torch

from pcdc.engine import read_checkpoint
from pcdc.model import build_model
from pcdc.runtime import setup_runtime
from scripts.levir_tiling_probe import (WindowPairs, check_checkpoint_identity,
    load_settings, precision_context, state_digest, write_new)
from scripts.windows_shared_json import read_json


def pilot(path, session, *, check, progress):
    path, cfg, base = load_settings(path)
    root = path.parent.parent
    prepared = read_json(root / cfg["output_dir"] / "prepared.json")
    from scripts.tiling_probe_supervision import verify
    verify(prepared["bound_sha256"])
    check(); device = setup_runtime(base)
    if device.type != "cuda" or torch.cuda.current_device() != 0:
        raise RuntimeError("Original CUDA device required")
    name = sorted(prepared["source_records"])[0]
    records = {name: prepared["source_records"][name]}
    ds = WindowPairs(records, cfg["arms"]["context512"], cfg["source_size"],
                     base.values["data"]["mean"], base.values["data"]["std"])
    examples = [ds[i] for i in range(4)]
    a = torch.stack([e["image_a"] for e in examples]).to(device, memory_format=torch.channels_last)
    b = torch.stack([e["image_b"] for e in examples]).to(device, memory_format=torch.channels_last)
    results, pairs = {}, 0
    for model_name, identity in prepared["models"].items():
        check(); cp = read_checkpoint(Path(identity["checkpoint"]))
        check_checkpoint_identity(cp, base, identity, prepared["source_records"])
        model = build_model(base, initialize_encoder=False)
        model.load_state_dict(cp["model"], strict=True); del cp
        model = model.to(device, memory_format=torch.channels_last).eval().requires_grad_(False)
        before = state_digest(model); torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
        reference = None
        with torch.inference_mode():
            for repeat in range(2):
                check()
                with precision_context(device, True):
                    logits = model(a, b)["logits"]
                if not torch.isfinite(logits).all() or logits.dtype != torch.float16:
                    raise ValueError("Nonfinite or non-FP16 pilot logits")
                p = logits.float().sigmoid().cpu()
                pairs += len(p)
                if reference is not None and not torch.equal(p, reference):
                    raise ValueError("Largest-window deterministic repeat differs")
                reference = p.clone()
                progress({"stage": "largest_window_pilot", "model": model_name, "step": repeat + 1,
                          "pair_predictions": pairs, "numeric_alert": False, "optimizer_updates": 0})
                check()
        if state_digest(model) != before:
            raise ValueError("Pilot modified model parameters/buffers")
        results[model_name] = {"pairs": 8, "repeat_exact": True, "model_state_unchanged": True,
            "seconds": time.monotonic() - start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        del model, reference, p, logits
    if pairs != cfg["engineering"]["largest_window_pilot_pairs_upper_bound"]:
        raise ValueError("Pilot exposure differs from16 fixed pairs")
    result = {"status": "passed", "engineering_only": True, "source": name, "models": results,
        "pair_predictions": pairs, "label_scores_computed": False, "optimizer_updates": 0,
        "geometry": [e["x"] for e in examples], "precision": "float16", "device": str(device)}
    write_new(Path(session) / "pilot_verified.json", result)
    return result
