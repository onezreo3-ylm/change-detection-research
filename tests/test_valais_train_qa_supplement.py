from types import SimpleNamespace
import unittest

from scripts.supplement_valais_train_qa import balanced_names


class ValaisTrainQaSupplementTests(unittest.TestCase):
    def test_excludes_first_batch_and_balances_direction_parity(self):
        names = [f"{index}.tif" for index in range(24)]
        dataset = SimpleNamespace(split="train", records=[SimpleNamespace(name=name)
                                                            for name in names])
        positive = set(names[:12])
        excluded = {names[0], names[1], names[12], names[13]}
        ordered = balanced_names(dataset, positive, excluded, seed=810208)
        self.assertEqual(len(set(ordered)), 8)
        self.assertFalse(set(ordered) & excluded)
        classes = [name in positive for name in ordered]
        self.assertEqual(classes, [True, False, True, False,
                                   False, True, False, True])
        self.assertEqual(sum(classes[::2]), 2)
        self.assertEqual(sum(classes[1::2]), 2)
        dataset.split = "val"
        with self.assertRaises(ValueError):
            balanced_names(dataset, positive, excluded, seed=810208)


if __name__ == "__main__":
    unittest.main()
