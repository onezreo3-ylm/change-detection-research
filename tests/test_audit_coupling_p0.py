"""Independent P0 audit tests; synthetic saved counts are engineering fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.audit_coupling_p0 import (audit_records, check_record, verify_manifest_files,
                                       verify_review, sha256, MODELS, FAMILIES)
from scripts.coupling_diagnostic import four_cell_diagnostic
from scripts.evaluate_coupling_p0 import summarize


class AuditCouplingP0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.rows = [], []
        diagnostics = {}
        for foreground in (0, 1):
            target = torch.zeros(256, 256)
            target[0, 0] = foreground
            logits = target * 4 - 2
            diagnostics[foreground] = four_cell_diagnostic(logits,
                dict.fromkeys(('uu', 'uv', 'vu', 'vv'), logits), target)
        for index in range(64):
            foreground = int(index < 16)
            sample = dict(id=f'train/{index}.tif@x0_y0', name=f'{index}.tif',
                          foreground_pixels=foreground,
                          stratum='positive_1' if foreground else 'empty')
            sample['conditions'] = [dict(family=family, level=level,
                images=dict(au='0'*64, av='0'*64, bu='1'*64, bv='1'*64),
                marginal_hashes={cell: ['0'*64, '1'*64] for cell in ('uu', 'uv', 'vu', 'vv')})
                for family in FAMILIES for level in (1, 3, 5)]
            cls.samples.append(sample)
            for model in MODELS:
                for family in FAMILIES:
                    for level in (1, 3, 5):
                        cls.rows.append(dict(model=model, source_id=sample['id'],
                            family=family, level=level, stratum=sample['stratum'],
                            foreground_pixels=foreground, diagnostic=diagnostics[foreground],
                            marginal_check=dict(matched=True, a_instances_identical=True,
                                                b_instances_identical=True, fully_degenerate=True)))
        cls.manifest = {'samples': cls.samples, 'settings': {'max_seconds': 21600}}
        cls.report = dict(status='completed_train_p0_diagnostic', smoke_only=False,
            train_only=True, val_accessed=False, test_accessed=False, training=False,
            gradients_enabled=False, amp=False, records=2304, main_predictions=9408, predictions=9414,
            elapsed_seconds=100, budget_seconds=21600, maximum_observed_temperature_c=65,
            self_checks=[dict(model=m, repeat_bitwise_equal=True, temporal_swap_allclose=True, max_swap_abs_error=0.0)
                         for m in MODELS], **summarize(cls.rows))

    def test_full_cartesian_coverage_counts_and_gate_pass_audit(self):
        result = audit_records(self.manifest, self.report, self.rows)
        self.assertEqual(result['records'], 2304)
        self.assertFalse(result['independent_gate']['passed'])

    def test_duplicate_or_missing_record_fails(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaisesRegex(ValueError, 'coverage|duplicate'):
                audit_records(self.manifest, self.report, rows)

    def test_incomplete_run_fails_before_count_audit(self):
        report = dict(self.report, status='incomplete')
        with self.assertRaisesRegex(ValueError, 'fully completed'):
            audit_records(self.manifest, report, self.rows)

    def test_group_count_tampering_fails(self):
        row = copy.deepcopy(self.rows[0])
        row['diagnostic']['groups']['D']['counts']['tp'] += 1
        with self.assertRaisesRegex(ValueError, 'pooled four-cell'):
            check_record(row, self.samples[0])

    def test_transition_and_foreground_conservation_tampering_fails(self):
        row = copy.deepcopy(self.rows[0])
        row['diagnostic']['cells']['uu']['transitions']['tp_to_fn'] = 1
        with self.assertRaisesRegex(ValueError, 'transition identities'):
            check_record(row, self.samples[0])
        row = copy.deepcopy(self.rows[0])
        row['diagnostic']['cells']['uu']['counts']['tp'] += 1
        with self.assertRaisesRegex(ValueError, 'pixel conservation'):
            check_record(row, self.samples[0])

    def test_gate_or_report_aggregate_tampering_fails(self):
        report = copy.deepcopy(self.report)
        report['gate']['passed'] = True
        with self.assertRaisesRegex(ValueError, 'gate mismatch'):
            audit_records(self.manifest, report, self.rows)
        report = copy.deepcopy(self.report)
        report['aggregates']['b0']['all']['groups']['D']['counts']['fp'] += 1
        with self.assertRaisesRegex(ValueError, 'aggregate counts'):
            audit_records(self.manifest, report, self.rows)

    def test_all_paths_prevalidated_before_any_image_hash(self):
        with tempfile.TemporaryDirectory(prefix='coupling-audit-') as temp:
            root = Path(temp)
            sample = dict(id='train/a.tif@x0_y0', name='a.tif', files={
                key: {'path': str(root/'train'/folder/'a.tif'), 'sha256': '0'*64}
                for key, folder in [('image_a', 'A'), ('image_b', 'B'), ('mask', 'label')]})
            sample['files']['mask']['path'] = str(root/'test'/'label'/'a.tif')
            manifest = dict(split='train', data_root=str(root), samples=[sample])
            with patch('scripts.audit_coupling_p0.sha256') as sha:
                with self.assertRaisesRegex(ValueError, 'Non-train image path'):
                    verify_manifest_files(manifest)
                sha.assert_not_called()

    def test_listed_training_file_hash_change_rejected(self):
        with tempfile.TemporaryDirectory(prefix='coupling-audit-') as temp:
            root = Path(temp)
            sample = dict(id='train/a.tif@x0_y0', name='a.tif', files={
                key: {'path': str(root/'train'/folder/'a.tif'), 'sha256': '0'*64}
                for key, folder in [('image_a', 'A'), ('image_b', 'B'), ('mask', 'label')]})
            manifest = dict(split='train', data_root=str(root), samples=[sample], models={},
                            source_hashes={}, config_path=str(root/'config.yaml'), config_sha256='0'*64)
            with patch('scripts.audit_coupling_p0.sha256', return_value='1'*64):
                with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
                    verify_manifest_files(manifest)

    def test_amp_budget_or_nonfinite_symmetry_error_rejected(self):
        for key, value in [('amp', True), ('budget_seconds', 21000)]:
            with self.assertRaises(ValueError):
                audit_records(self.manifest, dict(self.report, **{key: value}), self.rows)
        report = copy.deepcopy(self.report)
        report['self_checks'][0]['max_swap_abs_error'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'self checks'):
            audit_records(self.manifest, report, self.rows)

    def test_review_chain_and_qa_hashes(self):
        with tempfile.TemporaryDirectory(prefix='coupling-audit-') as temp:
            root = Path(temp)
            (root/'previews').mkdir()
            image = root/'previews'/'fixture.png'
            image.write_bytes(b'synthetic fixture bytes')
            manifest = {'qa_images': [{'path': str(image), 'sha256': sha256(image)}]}
            manifest_path = root/'manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            review = dict(manifest_sha256=sha256(manifest_path),
                status='accepted_train_only_sampled_visibility', reviewer='fixture', limitations='fixture',
                reviewed_images=[dict(manifest['qa_images'][0], findings='synthetic test')])
            review_path = root/'visual_review.json'
            review_path.write_text(json.dumps(review), encoding='utf-8')
            report = {'review_sha256': sha256(review_path)}
            self.assertEqual(verify_review(manifest, report, manifest_path, review_path)['reviewed_qa_hashes_verified'], 1)
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'QA image SHA256'):
                verify_review(manifest, report, manifest_path, review_path)
            with self.assertRaisesRegex(ValueError, 'review SHA256'):
                verify_review(manifest, {'review_sha256': '0'*64}, manifest_path, review_path)


if __name__ == '__main__':
    unittest.main()
