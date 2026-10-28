import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.analyze_valais_class_balance import empty_batch_probability, summarize_labels


class ValaisClassBalanceTests(unittest.TestCase):
    def test_empty_batch_probability_without_replacement(self):
        self.assertAlmostEqual(empty_batch_probability(2, 4, 2), 1 / 6)
        self.assertEqual(empty_batch_probability(1, 4, 2), 0)
        with self.assertRaises(ValueError):
            empty_batch_probability(5, 4, 2)

    def test_summary_counts_only_strict_binary_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = np.zeros((256, 256), dtype=np.uint8)
            changed = empty.copy()
            changed[0, 0] = 1
            Image.fromarray(empty).save(root / "1.tif")
            Image.fromarray(changed).save(root / "2.tif")
            result = summarize_labels(root, 2, 2)
            self.assertEqual(result["positive_pairs"], 1)
            self.assertEqual(result["positive_pixels"], 1)
            self.assertEqual(result["natural_empty_batch_probability"], 0)
            changed[0, 1] = 2
            Image.fromarray(changed).save(root / "2.tif")
            with self.assertRaises(RuntimeError):
                summarize_labels(root, 2, 2)


if __name__ == "__main__":
    unittest.main()
