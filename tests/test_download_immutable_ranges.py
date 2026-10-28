import unittest

from scripts.download_immutable_ranges import byte_ranges


class DownloadImmutableRangesTests(unittest.TestCase):
    def test_chunks_cover_all_bytes_without_overlap(self):
        self.assertEqual(byte_ranges(10, 4), [(0, 3), (4, 7), (8, 9)])
        self.assertEqual(byte_ranges(8, 4), [(0, 3), (4, 7)])
        with self.assertRaises(ValueError):
            byte_ranges(0, 4)


if __name__ == "__main__":
    unittest.main()
