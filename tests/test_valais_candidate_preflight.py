import unittest

from scripts.audit_valais_candidate_remote import split_members
from scripts.audit_whu_candidate_remote import ZipEntry


def entry(name):
    return ZipEntry(name, 8, 0, 1, 1, 0)


class ValaisCandidatePreflightTests(unittest.TestCase):
    def test_split_members_require_three_matched_identifier_sets(self):
        entries = {
            name: entry(name)
            for name in (
                "dev/2017/1.tif",
                "dev/2023/1.tif",
                "dev/labels/1.tif",
            )
        }
        groups = split_members(entries, "dev")
        self.assertEqual(set(groups["2017"]), {"1"})
        del entries["dev/labels/1.tif"]
        with self.assertRaises(RuntimeError):
            split_members(entries, "dev")


if __name__ == "__main__":
    unittest.main()
