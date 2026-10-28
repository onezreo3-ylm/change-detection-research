import unittest

import numpy as np

from scripts.audit_whu_candidate_remote import target_semantics, tile_position


class WhuCandidatePreflightTests(unittest.TestCase):
    def test_tile_position_uses_source_mosaic_columns(self):
        self.assertEqual(tile_position(0, 42), (0, 0))
        self.assertEqual(tile_position(42, 42), (1, 0))
        self.assertEqual(tile_position(100, 42), (2, 16))
        with self.assertRaises(ValueError):
            tile_position(-1, 42)

    def test_change_target_is_not_assumed_to_equal_xor(self):
        old = np.array([[1, 1], [0, 0]], dtype=bool)
        new = np.array([[1, 0], [1, 0]], dtype=bool)
        target = np.array([[0, 0], [1, 0]], dtype=bool)
        stats = target_semantics(old, new, target)
        self.assertEqual(stats["target_positive"], 1)
        self.assertEqual(stats["year_xor_positive"], 2)
        self.assertEqual(stats["target_on_added"], 1)
        self.assertEqual(stats["target_on_removed"], 0)
        self.assertFalse(stats["target_equals_year_xor"])


if __name__ == "__main__":
    unittest.main()
