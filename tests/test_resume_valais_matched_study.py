"""Resume/order engineering checks; all fixture results are synthetic only."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from pcdc.engine import read_checkpoint
import pcdc.robust_baselines as robust
from pcdc.smoke import create_fixture
from scripts.resume_valais_matched_study import OrderRecorder, SourceNames, names_from_manifest, replay_prefix, update_order
from test_robust_baselines import RobustFixture, TinyDetector


def metadata_fixture():
    manifest = {"split": "train", "tile_size": 64, "entries": [
                {"name": f"{i}.tif", "size": [64, 64]} for i in range(5)], "tiles": 5, "source_pairs": 5}
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(SourceNames(names_from_manifest(manifest)), batch_size=2,
                        shuffle=True, num_workers=0, generator=generator)
    digest, epoch_batches, history, rng = hashlib.sha256(), [], [], None
    for epoch in range(1, 4):
        batches = [list(batch) for batch in loader]
        epoch_batches.append(batches)
        stream = hashlib.sha256()
        for batch in batches:
            update_order(digest, batch)
            robust._update_stream_hash(stream, batch, ["clean"] * len(batch))
        history.append({"epoch": epoch, "train": {"exposure_stream_sha256": stream.hexdigest(),
                       "source_samples": 5, "pair_views": 10, "optimizer_steps": 3, "amp_skipped_steps": 0,
                       "exposure_counts": {"clean|l0|clean|none|none": 5}}})
        if epoch == 1:
            rng = generator.get_state().clone()
    checkpoint = {"epoch": 1, "manifests": {"train": manifest}, "config": {
                  "experiment": {"seed": 42}, "train": {"batch_size": 2, "epochs": 3}},
                  "history": history[:1], "rng": {"loader": rng}}
    reference = {"source_order_sha256": digest.hexdigest(), "source_pairs": 15, "batches": 9}
    return checkpoint, reference, epoch_batches


class MatchedResumeTests(unittest.TestCase):
    def test_train_metadata_names_include_tile_coordinates_and_exact_order(self):
        manifest = {"split": "train", "tile_size": 64, "entries": [
                    {"name": "a.tif", "size": [128, 64]}], "tiles": 2, "source_pairs": 1}
        self.assertEqual(names_from_manifest(manifest), ["train/a.tif@x0_y0", "train/a.tif@x64_y0"])
        manifest["split"] = "test"
        with self.assertRaisesRegex(ValueError, "train-only"):
            names_from_manifest(manifest)

    def test_replay_matches_rng_prefix_and_keeps_singleton_tail(self):
        checkpoint, reference, batches = metadata_fixture()
        prefix, plan = replay_prefix(checkpoint, reference)
        self.assertTrue(plan["saved_loader_rng_matches"])
        self.assertEqual(plan["first_resumed_batch_ids"], batches[1][0])
        tracker = OrderRecorder(prefix, plan)
        for epoch in batches[1:]:
            self.assertEqual([len(batch) for batch in epoch], [2, 2, 1])
            for batch in epoch:
                tracker.record(batch, ["clean"] * len(batch))
        self.assertEqual(tracker.finish(), reference)

    def test_saved_stream_tampering_is_rejected(self):
        checkpoint, reference, _ = metadata_fixture()
        checkpoint["history"][0]["train"]["exposure_stream_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "epoch 1"):
            replay_prefix(checkpoint, reference)

    def test_saved_rng_tampering_is_rejected(self):
        checkpoint, reference, _ = metadata_fixture()
        checkpoint["rng"]["loader"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "loader RNG"):
            replay_prefix(checkpoint, reference)

    def test_completed_reference_mismatch_is_rejected(self):
        checkpoint, reference, _ = metadata_fixture()
        reference["source_order_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "completed B1"):
            replay_prefix(checkpoint, reference)

    def test_partial_resume_cannot_claim_completion(self):
        checkpoint, reference, batches = metadata_fixture()
        prefix, plan = replay_prefix(checkpoint, reference)
        tracker = OrderRecorder(prefix, plan)
        tracker.record(batches[1][0], ["clean"] * len(batches[1][0]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            tracker.finish()

    def test_actual_resumed_order_changes_fail_at_epoch_boundary(self):
        checkpoint, reference, batches = metadata_fixture()
        prefix, plan = replay_prefix(checkpoint, reference)
        tracker = OrderRecorder(prefix, plan)
        changed = copy.deepcopy(batches[1])
        changed[0] = list(reversed(changed[0]))
        with self.assertRaisesRegex(ValueError, "epoch 2"):
            for batch in changed:
                tracker.record(batch, ["clean"] * len(batch))

    def test_perturbed_instance_or_mismatched_lengths_are_rejected(self):
        checkpoint, reference, batches = metadata_fixture()
        prefix, plan = replay_prefix(checkpoint, reference)
        for instances in (["clean"], ["clean", "perturbed"]):
            with self.subTest(instances=instances), self.assertRaisesRegex(ValueError, "instance"):
                OrderRecorder(prefix, plan).record(batches[1][0], instances)

    def test_original_trainer_resume_with_recorder_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RobustFixture(Path(directory), epochs=2)
            create_fixture(fixture.base)
            complete_cfg, _ = fixture.config("b0e", "complete")
            stopped_cfg, _ = fixture.config("b0e", "stopped")
            actual = hashlib.sha256()
            original_update = robust._update_stream_hash
            def record_complete(digest, names, instances):
                original_update(digest, names, instances)
                update_order(actual, names)
            with patch("pcdc.robust_baselines.build_model", side_effect=lambda *a, **k: TinyDetector()), \
                 patch("pcdc.robust_baselines._update_stream_hash", side_effect=record_complete):
                complete = robust.train_robust(complete_cfg)
            complete_checkpoint = read_checkpoint(complete_cfg.output_dir / "last.pt")
            calls = 0
            def stop_after_epoch_one():
                nonlocal calls
                calls += 1
                if calls == 6:
                    raise KeyboardInterrupt("synthetic boundary stop")
            with patch("pcdc.robust_baselines.build_model", side_effect=lambda *a, **k: TinyDetector()):
                with self.assertRaises(KeyboardInterrupt):
                    robust.train_robust(stopped_cfg, guard_callback=stop_after_epoch_one)
                checkpoint = read_checkpoint(stopped_cfg.output_dir / "last.pt")
                reference = {"source_order_sha256": actual.hexdigest(), "source_pairs": 8, "batches": 4}
                prefix, plan = replay_prefix(checkpoint, reference)
                tracker = OrderRecorder(prefix, plan)
                def record_resume(digest, names, instances):
                    original_update(digest, names, instances)
                    tracker.record(names, instances)
                with patch("pcdc.robust_baselines._update_stream_hash", side_effect=record_resume):
                    resumed = robust.train_robust(stopped_cfg, resume=True)
            resumed_checkpoint = read_checkpoint(stopped_cfg.output_dir / "last.pt")
            self.assertEqual(tracker.finish(), reference)
            self.assertEqual(resumed["best_val_f1"], complete["best_val_f1"])
            for name, value in resumed_checkpoint["model"].items():
                torch.testing.assert_close(value, complete_checkpoint["model"][name], atol=0, rtol=0)
            self.assertEqual(resumed_checkpoint["scheduler"], complete_checkpoint["scheduler"])
            self.assertTrue(torch.equal(resumed_checkpoint["rng"]["loader"], complete_checkpoint["rng"]["loader"]))


if __name__ == "__main__":
    unittest.main()
