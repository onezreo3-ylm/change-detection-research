"""Deterministic train-only sampling for the reviewed C-S / A-S controls."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import yaml
from torch.utils.data import Sampler

from pcdc.config import UniqueKeyLoader
from pcdc.data import file_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_sampling(proposal_path):
    path = Path(proposal_path).resolve()
    value = yaml.load(path.read_bytes(), Loader=UniqueKeyLoader)
    settings = value["sampling"]
    expected = {
        "train_positive_sources": 1046, "train_empty_sources": 9979,
        "source_batch_size": 4, "full_batches_per_epoch": 2756,
        "each_full_batch": {"positive": 1, "empty": 3},
        "final_batch": {"positive": 1, "empty": 0},
        "positive_draws_per_epoch": 2757, "empty_draws_per_epoch": 8268,
        "random_generator": "Python_random_Random_local_instances_version_2_string_seed",
        "global_runtime_rng_modified_by_sampling": False,
    }
    if value["seed"] != 42 or settings["seed"] != 42:
        raise ValueError("The reviewed sampler is seed42 only")
    if any(settings.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Sampling differs from the reviewed exact proposal")
    paths = {}
    for key in ("train_label_report", "captured_data_audit", "error_report", "reference_validation"):
        target = (path.parent / value["evidence"][key]).resolve()
        if file_sha256(target) != value["evidence"][key + "_sha256"]:
            raise ValueError(f"Sampling evidence changed: {key}")
        paths[key] = target
    report = json.loads(paths["train_label_report"].read_text(encoding="utf-8"))
    if report["test_accessed"] is not False:
        raise ValueError("Sampling labels must come from development data")
    train = report["train"]
    if (train["pairs"], train["positive_pairs"], train["empty_pairs"]) != (11025, 1046, 9979):
        raise ValueError("Sampling partition counts differ")
    rows = train["positive_manifest"]
    positives = [row["name"] for row in rows]
    if len(set(positives)) != 1046 or any(row["positive_pixels"] <= 0 for row in rows):
        raise ValueError("Invalid complete train-positive manifest")
    return value, paths, positives


@dataclass(frozen=True)
class EpochPlan:
    batches: tuple
    summary: dict


class StratifiedBatchSampler(Sampler):
    """Local RNGs, complete positive cycles, unique empty draws, fixed tail."""

    def __init__(self, names, positive_names, settings, *, epoch=1):
        self.names = tuple(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate source names")
        positive_names = tuple(positive_names)
        if len(set(positive_names)) != len(positive_names) or not set(positive_names) < set(self.names):
            raise ValueError("Positive names must be a unique proper source subset")
        positive_set = set(positive_names)
        self.positive = tuple(i for i, name in enumerate(self.names) if name in positive_set)
        self.empty = tuple(i for i, name in enumerate(self.names) if name not in positive_set)
        self.settings = dict(settings)
        if (len(self.positive) != settings["train_positive_sources"]
                or len(self.empty) != settings["train_empty_sources"]):
            raise ValueError("Dataset differs from the train label partition")
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Epoch must be a positive integer")
        self.epoch = epoch

    def __len__(self):
        return self.settings["full_batches_per_epoch"] + 1

    def plan(self):
        s, epoch = self.settings, self.epoch
        full, wanted = s["full_batches_per_epoch"], s["positive_draws_per_epoch"]
        if wanted != full + 1 or s["empty_draws_per_epoch"] != 3 * full:
            raise ValueError("Unsupported batch construction")
        draws, cycle = [], 0
        while len(draws) < wanted:
            current = list(self.positive)
            random.Random(s["positive_cycle_seed_template"].format(epoch=epoch, cycle=cycle)).shuffle(current)
            draws.extend(current)
            cycle += 1
        draws = draws[:wanted]
        empty = list(self.empty)
        random.Random(s["empty_seed_template"].format(epoch=epoch)).shuffle(empty)
        empty = empty[:s["empty_draws_per_epoch"]]
        batches = []
        for batch in range(full):
            row = [draws[batch], *empty[3 * batch:3 * batch + 3]]
            random.Random(s["within_batch_seed_template"].format(epoch=epoch, batch=batch)).shuffle(row)
            batches.append(tuple(row))
        random.Random(s["full_batch_order_seed_template"].format(epoch=epoch)).shuffle(batches)
        batches.append((draws[-1],))
        repeats = Counter(draws)
        name_batches = [[f"train/{self.names[i]}@x0_y0" for i in row] for row in batches]
        return EpochPlan(tuple(batches), {
            "epoch": epoch, "batches": len(batches),
            "source_draws": wanted + len(empty), "positive_draws": wanted,
            "empty_draws": len(empty), "unique_positive_sources": len(repeats),
            "unique_empty_sources": len(set(empty)),
            "positive_repeat_min": min(repeats.values()), "positive_repeat_max": max(repeats.values()),
            "source_batch_sha256": digest(name_batches),
        })

    def __iter__(self):
        return iter(self.plan().batches)


class ObservedLoader:
    """Check the actual loaded source batches against the deterministic plan."""

    def __init__(self, loader, sampler):
        self.loader, self.sampler = loader, sampler
        self.generator = loader.generator
        self.observed = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        plan = self.sampler.plan()
        names, batches, samples = [], 0, 0
        self.observed = None
        for position, batch in enumerate(self.loader):
            if position >= len(plan.batches):
                raise ValueError("The actual sampler stream contains extra batches")
            indices = plan.batches[position]
            expected = [f"train/{self.sampler.names[i]}@x0_y0" for i in indices]
            if list(batch["sample_id"]) != expected:
                raise ValueError("Actual source stream differs from the frozen batch plan")
            names.append(expected)
            batches += 1
            samples += len(expected)
            yield batch
        if batches != len(plan.batches) or samples != plan.summary["source_draws"]:
            raise ValueError("The actual sampler stream is incomplete")
        if digest(names) != plan.summary["source_batch_sha256"]:
            raise ValueError("Actual source batch hash differs")
        self.observed = dict(plan.summary, actual_source_stream_verified=True)
