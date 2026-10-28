"""Frozen, label-region diagnostics of existing LEVIR checkpoints; no training.

Components belong to the full 1024 label map, not independently cut tile masks.
They are connected label regions, never certified physical building instances.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from pcdc.config import load_config, protocol_signature
from pcdc.data import LEVIRDataset, file_sha256
from pcdc.engine import move_model, read_checkpoint
from pcdc.model import build_model
from pcdc.runtime import setup_runtime
from scripts.evaluate_guarded import EvaluationTemperatureGuard, load_guard_config


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, replace=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w" if replace else "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_names(names, count, salt):
    if len(set(names)) != len(names) or not 0 < count <= len(names):
        raise ValueError("Unique names and a feasible source count are required")
    selected = sorted(names, key=lambda s: (hashlib.sha256((salt + "\0" + s).encode()).hexdigest(), s))[:count]
    return sorted(selected)


def region_records(mask, predictions, *, connectivity, tile_size):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or connectivity not in [4, 8]:
        raise ValueError("Expected a 2D binary label map and 4/8 connectivity")
    for p in predictions.values():
        if np.shape(p) != mask.shape or not np.isin(p, [0, 1]).all():
            raise ValueError("Predictions must be aligned binary maps")
    labels, total = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2))
    areas = np.bincount(labels.ravel(), minlength=total + 1)
    overlaps = {name: np.bincount(labels.ravel(), weights=np.asarray(p).ravel(), minlength=total + 1).astype(np.int64)
                for name, p in predictions.items()}
    records = []
    height, width = mask.shape
    for index, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            raise ValueError("Unexpected missing connected region")
        ys, xs = box
        # A connected path crossing a seam must hit its adjacent border pixels.
        grid = any(xs.start <= s <= xs.stop - 1 for s in range(tile_size - 1, width - 1, tile_size)) or any(
            xs.start <= s <= xs.stop - 1 for s in range(tile_size, width, tile_size)) or any(
            ys.start <= s <= ys.stop - 1 for s in range(tile_size - 1, height - 1, tile_size)) or any(
            ys.start <= s <= ys.stop - 1 for s in range(tile_size, height, tile_size))
        records.append({"region_id": index, "area": int(areas[index]),
                        "bbox_xyxy_inclusive": [xs.start, ys.start, xs.stop - 1, ys.stop - 1],
                        "touches_source_edge": xs.start == 0 or ys.start == 0 or xs.stop == width or ys.stop == height,
                        "touches_tile_grid": grid,
                        "overlap_pixels": {name: int(values[index]) for name, values in overlaps.items()}})
    if sum(r["area"] for r in records) != int(mask.sum()):
        raise ValueError("Connected-region area conservation failed")
    return records


def stitch(tiles, size, tile_size):
    result = np.zeros((size, size), dtype=np.uint8)
    coverage = np.zeros((size, size), dtype=np.uint8)
    for x, y, tile in tiles:
        if x % tile_size or y % tile_size or x < 0 or y < 0 or x + tile_size > size or y + tile_size > size:
            raise ValueError("Invalid original tile origin")
        if np.shape(tile) != (tile_size, tile_size) or not np.isin(tile, [0, 1]).all():
            raise ValueError("Invalid binary tile")
        result[y:y+tile_size, x:x+tile_size] = tile
        coverage[y:y+tile_size, x:x+tile_size] += 1
    if not np.all(coverage == 1):
        raise ValueError("Source reconstruction has missing or duplicate pixels")
    return result


def eligible(row, cfg):
    return (cfg["primary_min_area"] <= row["area"] <= cfg["primary_max_area"]
            and not (cfg["primary_exclude_source_edge"] and row["touches_source_edge"])
            and not (cfg["primary_exclude_tile_grid"] and row["touches_tile_grid"]))


def settings(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / cfg["workspace_root"]).resolve()
    base = load_config(root / cfg["base_config"])
    if cfg["test_access"] or cfg["new_training"] or base.run_test_after_training:
        raise ValueError("Diagnostic cannot open held-out test or train")
    if base.values["data"]["splits"] != {"train":"train", "val":"val", "test":"test"}:
        raise ValueError("Raw split aliases are forbidden")
    if base.values["runtime"]["device"] not in ["cuda", "cuda:0"]:
        raise ValueError("Real diagnostic requires the requested CUDA device")
    return cfg, root, base


def prepare(path):
    cfg, root, base = settings(path)
    output = root / cfg["output_dir"]
    output.mkdir(exist_ok=False)
    write_json(output / "diagnostic_config.json", cfg)
    guard_path = output / "guard.yaml"
    guard_path.write_text(yaml.safe_dump(cfg["temperature_guard"], sort_keys=False), encoding="utf-8")
    datasets = {s:LEVIRDataset(base, s) for s in ["train", "val"]}
    selected = {"train":select_names([r.name for r in datasets["train"].records], cfg["train_source_count"], cfg["train_selection_salt"]),
                "val":[r.name for r in datasets["val"].records]}
    if len(selected["val"]) != cfg["validation_source_count"]:
        raise ValueError("Original validation coverage differs")
    inputs = {str(Path(path).resolve()):file_sha256(Path(path)), str(base.source):file_sha256(base.source),
              str(Path(__file__).resolve()):file_sha256(Path(__file__)),
              str(root/cfg["protocol"]):file_sha256(root/cfg["protocol"])}
    for directory in [root / "pcdc"]:
        inputs.update({str(p):file_sha256(p) for p in directory.glob("*.py")})
    load_guard_config(guard_path).verify()
    models, reference_files = {}, None
    for name, item in cfg["models"].items():
        job_path = root / item["job_config"]
        job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
        checkpoint_path = (job_path.parent / job["candidate_best_checkpoint"]).resolve()
        expected = job["candidate_best_checkpoint_sha256"]
        if file_sha256(checkpoint_path) != expected:
            raise ValueError("Frozen checkpoint changed: " + name)
        checkpoint = read_checkpoint(checkpoint_path)
        if (checkpoint["training_flavor"] != "matched_robust_baseline_v1" or checkpoint["robust_mode"] != item["mode"]
                or checkpoint["kind"] != "research" or checkpoint["config"]["experiment"]["seed"] != item["seed"]
                or checkpoint["protocol_signature"] != protocol_signature(base.values)
                or checkpoint["epoch"] != checkpoint["best_epoch"]):
            raise ValueError("Checkpoint identity/compute/selection differs: " + name)
        files = {split:{record.name:checkpoint["file_sha256"][split][record.name]
                        for record in datasets[split].records if record.name in selected[split]} for split in selected}
        if reference_files is None:
            reference_files = files
        elif files != reference_files:
            raise ValueError("Checkpoint source content identities differ")
        clean = (job_path.parent / job["output_dir"] / "clean_metrics.json").resolve()
        models[name] = {"checkpoint":str(checkpoint_path),"checkpoint_sha256":expected,
                        "seed":item["seed"],"epoch":checkpoint["epoch"],"mode":item["mode"],
                        "saved_clean_metrics":str(clean)}
        for p in [job_path, checkpoint_path, clean]:
            inputs[str(p)] = file_sha256(p)
        del checkpoint
    records = {split:[] for split in selected}
    for split in selected:
        for record in datasets[split].records:
            if record.name not in selected[split]:
                continue
            entry = {"name":record.name, "files":{}}
            for role, expected in reference_files[split][record.name].items():
                p = getattr(record, role)
                if file_sha256(p) != expected:
                    raise ValueError("Original content changed: " + str(p))
                entry["files"][role] = {"path":str(p),"sha256":expected}
            records[split].append(entry)
    write_json(output / "prepared.json", {"selected":selected,"records":records,"models":models,
                                          "inputs_sha256":inputs,"selection_independent_of_model_scores":True,
                                          "test_accessed":False,"training_started":False})
    return {"status":"prepared","train_sources":len(selected["train"]),"val_sources":len(selected["val"]),"models":len(models)}


def infer(path):
    cfg, root, base = settings(path)
    output = root / cfg["output_dir"]
    prep = read_json(output / "prepared.json")
    for p, expected in prep["inputs_sha256"].items():
        if file_sha256(Path(p)) != expected:
            raise ValueError("Prepared input changed: " + p)
    for entries in prep["records"].values():
        for entry in entries:
            for info in entry["files"].values():
                if file_sha256(Path(info["path"])) != info["sha256"]:
                    raise ValueError("Selected source changed")
    execution = output / "inference"
    if execution.exists():
        raise FileExistsError("Preserve the earlier diagnostic inference directory")
    lock = root / cfg["managed_run_dir"] / "gpu_study.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid":os.getpid(),"config":str(Path(path).resolve()),"started_utc":datetime.now(timezone.utc).isoformat()}, stream)
    guard = EvaluationTemperatureGuard(load_guard_config(output / "guard.yaml"), execution)
    started = time.monotonic()
    try:
        execution.mkdir(exist_ok=False)
        guard(force=True)
        device = setup_runtime(base)
        datasets = {s:LEVIRDataset(base,s) for s in ["train","val"]}
        size, tile_size = base.values["data"]["expected_image_size"], base.values["data"]["tile_size"]
        tiles_predicted = 0
        for name, item in prep["models"].items():
            checkpoint = read_checkpoint(Path(item["checkpoint"]))
            model = move_model(build_model(base,initialize_encoder=False),device,base.values["runtime"]["memory_format"])
            model.load_state_dict(checkpoint["model"],strict=True)
            model.eval()
            del checkpoint
            for split,dataset in datasets.items():
                indices = [i for i,(record_index,_,_) in enumerate(dataset.tiles)
                           if dataset.records[record_index].name in prep["selected"][split]]
                loader = DataLoader(Subset(dataset,indices),batch_size=base.values["train"]["eval_batch_size"],
                                    shuffle=False,num_workers=base.values["train"]["num_workers"],pin_memory=True)
                prediction_tiles, label_tiles = {},{}
                with torch.inference_mode():
                    for batch_index,batch in enumerate(loader,1):
                        if time.monotonic()-started > cfg["max_wall_seconds"]:
                            raise TimeoutError("Frozen diagnostic runtime budget exhausted")
                        guard()
                        a,b=[batch[k].to(device=device,non_blocking=True,memory_format=torch.channels_last) for k in ["image_a","image_b"]]
                        with torch.autocast(device_type="cuda",enabled=base.values["train"]["amp"]):
                            logits=model(a,b)["logits"]
                        if not torch.isfinite(logits).all().item():
                            raise FloatingPointError("Nonfinite diagnostic logits")
                        predictions=(logits.float().sigmoid()>=base.values["evaluation"]["threshold"]).cpu().numpy()[:,0].astype(np.uint8)
                        targets=batch["mask"].numpy()[:,0].astype(np.uint8)
                        for sid,prediction,label in zip(batch["sample_id"],predictions,targets):
                            match=re.fullmatch(r"(train|val)/([^/]+)@x(\d+)_y(\d+)",sid)
                            if match is None or match[1]!=split:
                                raise ValueError("Unexpected source ID")
                            key,x,y=match[2],int(match[3]),int(match[4])
                            prediction_tiles.setdefault(key,[]).append((x,y,prediction))
                            label_tiles.setdefault(key,[]).append((x,y,label))
                        tiles_predicted += len(predictions)
                        write_json(output/"status.json",{"status":"inference_running","model":name,"split":split,"batch":batch_index,
                                   "tiles_predicted":tiles_predicted,"pid":os.getpid(),"training_started":False},replace=True)
                maps={key:stitch(value,size,tile_size) for key,value in prediction_tiles.items()}
                labels={key:stitch(value,size,tile_size) for key,value in label_tiles.items()}
                counts={key:{"tn":int(((labels[key]==0)&(p==0)).sum()),"fp":int(((labels[key]==0)&(p==1)).sum()),
                             "fn":int(((labels[key]==1)&(p==0)).sum()),"tp":int(((labels[key]==1)&(p==1)).sum())} for key,p in maps.items()}
                if split=="val":
                    saved=read_json(item["saved_clean_metrics"])
                    if counts!={sid.split('/')[1]:v["counts"] for sid,v in saved["by_source"].items()}:
                        raise ValueError("Clean validation replay differs: "+name)
                np.savez_compressed(execution/f"{name}_{split}_predictions.npz",**{key:np.packbits(p,axis=1) for key,p in maps.items()})
                label_path=execution/f"{split}_labels.npz"
                if not label_path.exists():
                    np.savez_compressed(label_path,**{key:np.packbits(p,axis=1) for key,p in labels.items()})
                write_json(execution/f"{name}_{split}_counts.json",counts)
                print(f"{name}/{split}: {len(maps)} complete sources; validation replay checked={split=='val'}",flush=True)
            del model
            torch.cuda.empty_cache()
        result={"status":"inference_completed","tiles_predicted":tiles_predicted,"training_started":False,"test_accessed":False,
                "seconds":time.monotonic()-started,"all_four_original_validation_count_maps_exactly_replayed":True}
        guard.finish(result)
        write_json(execution/"guard_completion.json",result)
        write_json(output/"status.json",result,replace=True)
        return result
    except BaseException as error:
        guard.abort(error)
        write_json(output/"status.json",{"status":"failed","error":f"{type(error).__name__}: {error}","training_started":False},replace=True)
        raise
    finally:
        lock.unlink()


def analyze(path):
    cfg, root, base = settings(path)
    output=root/cfg["output_dir"]
    prep=read_json(output/"prepared.json")
    if read_json(output/"status.json")["status"]!="inference_completed":
        raise ValueError("Only a completed, guarded diagnostic may be analyzed")
    groups={}
    records=[]
    for split in ["train","val"]:
        with np.load(output/"inference"/f"{split}_labels.npz") as z:
            labels={key:np.unpackbits(z[key],axis=1) for key in z.files}
        predictions={}
        for model in cfg["models"]:
            with np.load(output/"inference"/f"{model}_{split}_predictions.npz") as z:
                predictions[model]={key:np.unpackbits(z[key],axis=1) for key in z.files}
        if set(labels)!=set(prep["selected"][split]):
            raise ValueError("Saved source coverage differs")
        for key,mask in labels.items():
            for connectivity in cfg["connectivities"]:
                rows=region_records(mask,{name:maps[key] for name,maps in predictions.items()},connectivity=connectivity,
                                    tile_size=base.values["data"]["tile_size"])
                for row in rows:
                    row.update(source_id=f"{split}/{key}",split=split,connectivity=connectivity,primary_eligible=eligible(row,cfg))
                records.extend(rows)
        selected=[row for row in records if row["split"]==split and row["connectivity"]==cfg["primary_connectivity"] and row["primary_eligible"]]
        common=[row for row in selected if all(row["overlap_pixels"][m]==0 for m in cfg["resource_gate"]["comparison_models"])]
        groups[split]={"eligible_regions":len(selected),"common_missed_regions":len(common),
                       "common_miss_sources":sorted({r["source_id"] for r in common}),
                       "common_miss_fraction":len(common)/len(selected) if selected else None,
                       "zero_overlap_by_model":{m:sum(row["overlap_pixels"][m]==0 for row in selected) for m in cfg["models"]},
                       "common_missed_region_keys":[[r["source_id"],r["region_id"],r["area"]] for r in common]}
    gate=cfg["resource_gate"]
    val=groups["val"]
    passed=(val["common_missed_regions"]>=gate["minimum_common_missed_regions"]
            and len(val["common_miss_sources"])>=gate["minimum_sources_with_common_misses"]
            and (val["common_miss_fraction"] or 0)>=gate["minimum_common_miss_fraction"])
    sensitivity={}
    for split in ["train","val"]:
        for connectivity in cfg["connectivities"]:
            selected=[r for r in records if r["split"]==split and r["connectivity"]==connectivity and r["primary_eligible"]]
            sensitivity[f"{split}_cc{connectivity}"]={"eligible_regions":len(selected),"zero_overlap_by_model":{
                m:sum(r["overlap_pixels"][m]==0 for r in selected) for m in cfg["models"]}}
    result={"status":"completed_exploratory_connected_label_region_diagnostic","groups":groups,
            "connectivity_sensitivity":sensitivity,
            "ordinary_control_protocol_preparation_resource_gate_passed":passed,
            "new_method_admitted":False,"new_training_admitted_by_this_diagnostic":False,
            "physical_building_instances_certified":False,"independent_truth_claim":False,
            "test_accessed":False,"training_started":False,
            "interpretation":"Post-hoc exploratory development diagnosis. Any-overlap and mask components are not complete-building detection accuracy or a causal explanation."}
    write_json(output/"region_records.json",records)
    write_json(output/"analysis.json",result)
    write_json(output/"status.json",result,replace=True)
    return result


def smoke():
    mask=np.zeros((32,32),dtype=np.uint8)
    mask[8:12,14:18]=1
    tiles=[(x,y,mask[y:y+16,x:x+16]) for y in [0,16] for x in [0,16]]
    restored=stitch(tiles,32,16)
    rows=region_records(restored,{"miss":np.zeros_like(mask),"hit":mask},connectivity=8,tile_size=16)
    assert len(rows)==1 and rows[0]["area"]==16 and rows[0]["touches_tile_grid"]
    assert rows[0]["overlap_pixels"]=={"miss":0,"hit":16}
    return {"smoke":"passed","synthetic_engineering_only":True,"model_inference":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","run","analyze","smoke"])
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    if args.command!="smoke" and args.config is None:
        parser.error("--config is required")
    result={"prepare":prepare,"run":infer,"analyze":analyze}.get(args.command,lambda _:smoke())(args.config)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
