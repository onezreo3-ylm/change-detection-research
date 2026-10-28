import os
import unittest

from scripts.run_valais_clean_guarded import creation_time_ticks


@unittest.skipUnless(os.name == "nt", "Windows process binding only")
class ValaisGuardLauncherTests(unittest.TestCase):
    def test_creation_tick_identity_is_stable_for_current_process(self):
        first = creation_time_ticks(os.getpid())
        second = creation_time_ticks(os.getpid())
        self.assertGreater(first, 0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
