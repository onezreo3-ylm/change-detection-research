from types import SimpleNamespace
import unittest

from scripts.prepare_valais_train_qa import select_tiles


class ValaisTrainQaTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_train_only(self):
        names = [f"{index}.tif" for index in range(20)]
        records = [SimpleNamespace(name=name) for name in names]
        dataset = SimpleNamespace(split="train", records=records,
                                  tiles=[(index, 0, 0) for index in range(20)])
        positive = set(names[:8])
        first = select_tiles(dataset, positive, seed=810207,
                             positive_count=4, empty_count=4)
        second = select_tiles(dataset, positive, seed=810207,
                              positive_count=4, empty_count=4)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 8)
        self.assertTrue(all(names[first[i]] in positive for i in range(0, 8, 2)))
        self.assertTrue(all(names[first[i]] not in positive for i in range(1, 8, 2)))
        dataset.split = "val"
        with self.assertRaises(ValueError):
            select_tiles(dataset, positive, seed=810207,
                         positive_count=4, empty_count=4)


if __name__ == "__main__":
    unittest.main()
