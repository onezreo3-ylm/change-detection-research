"""P0 preparation safety fixtures; synthetic images are engineering only."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pcdc.perturbations import corrupt_rgb
from scripts.prepare_coupling_p0 import checked_path, draw_instance, qa_sheet, train_path


class PrepareCouplingP0Tests(unittest.TestCase):
    def corruptions(self):
        return {'brightness': {
            'down': [{'factor': .9 - .1*i} for i in range(5)],
            'up': [{'factor': 1.1 + .1*i} for i in range(5)]}}

    def test_train_path_rejects_traversal_folder_and_split(self):
        with tempfile.TemporaryDirectory(prefix='coupling-p0-') as temp:
            root = Path(temp)
            self.assertEqual(train_path(root, 'A', 'source.png'), root / 'train' / 'A' / 'source.png')
            for folder, name in [('test', 'x.png'), ('A', '../x.png'), ('B', '..\\x.png'),
                                 ('label', 'foo/bar.png'), ('A', str(root / 'x.png')), ('A', '..')]:
                with self.subTest(folder=folder, name=name), self.assertRaises(ValueError):
                    train_path(root, folder, name)
            with self.assertRaisesRegex(ValueError, 'Non-train'):
                train_path(root / 'test', 'A', 'source.png')

    def test_sha_check_refuses_changed_config_dependency(self):
        with tempfile.TemporaryDirectory(prefix='coupling-p0-') as temp:
            root = Path(temp)
            dependency = root / 'audit.json'
            dependency.write_text('{"a":1}', encoding='utf-8')
            values = {'audit': 'audit.json', 'audit_sha256': hashlib.sha256(dependency.read_bytes()).hexdigest()}
            self.assertEqual(checked_path(root / 'config.yaml', values, 'audit'), dependency)
            dependency.write_text('{"a":2}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                checked_path(root / 'config.yaml', values, 'audit')

    def test_direction_draw_is_deterministic_same_level_and_keeps_duplicates(self):
        table = self.corruptions()
        original = copy.deepcopy(table)
        duplicate, different = False, False
        for index in range(40):
            sid = f'train/{index}'
            u = draw_instance(table, 'brightness', 3, 42, sid, 'u')
            v = draw_instance(table, 'brightness', 3, 42, sid, 'v')
            self.assertEqual(u, draw_instance(table, 'brightness', 3, 42, sid, 'u'))
            self.assertNotEqual(u['seed'], v['seed'])
            self.assertEqual(u['parameters'], table['brightness'][u['direction']][2])
            self.assertEqual(v['parameters'], table['brightness'][v['direction']][2])
            duplicate |= u['direction'] == v['direction']
            different |= u['direction'] != v['direction']
        self.assertTrue(duplicate, 'duplicate draws must not be forced opposite or redrawn')
        self.assertTrue(different, 'variants must not be forced identical')
        self.assertEqual(table, original)

    def test_out_of_scope_family_and_level_rejected(self):
        for family, level in [('brightness', 2), ('noise', 3), ('brightness', 0)]:
            with self.assertRaises(ValueError):
                draw_instance(self.corruptions(), family, level, 42, 'train/x', 'u')

    def test_qa_tiles_preserve_exact_original_and_corrupted_pixels(self):
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        b = np.flip(a, axis=0).copy()
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[30:70, 20:40] = 1
        directions = ['down', 'up']
        table = self.corruptions()
        with tempfile.TemporaryDirectory(prefix='coupling-p0-') as temp:
            output = Path(temp) / 'qa.png'
            with patch.object(Image.Image, 'resize', side_effect=AssertionError('QA must never resize')):
                report = qa_sheet((a, b), mask, 'brightness', directions, table, 5, output, 'fixture')
            pixels = np.asarray(Image.open(output))
            self.assertEqual(report['pixels'], [768, 834])
            self.assertTrue(report['original_resolution'])
            self.assertFalse(report['resized'])
            self.assertEqual(report['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            for row, raw in enumerate((a, b)):
                y = row * 278 + 22
                np.testing.assert_array_equal(pixels[y:y+256, :256], raw)
                for col, direction in enumerate(directions, 1):
                    expected = corrupt_rgb(raw, 'brightness', table['brightness'][direction][4], seed=0)
                    np.testing.assert_array_equal(pixels[y:y+256, col*256:(col+1)*256], expected)
            np.testing.assert_array_equal(pixels[578:834, :256, 0], mask * 255)


if __name__ == '__main__':
    unittest.main()
