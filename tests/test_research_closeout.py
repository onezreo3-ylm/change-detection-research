"""Synthetic-only checks for the metadata-only research closeout verifier."""

from pathlib import Path
import tempfile
import unittest

from scripts.verify_research_closeout import (read_json, read_yaml, safe_path, sha,
                                              synthetic_smoke, verify_files)


class ResearchCloseoutTests(unittest.TestCase):
    def test_synthetic_smoke(self) -> None:
        self.assertEqual(synthetic_smoke()["status"], "passed")

    def test_verifies_then_rejects_tampered_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.json"
            path.write_text('{"synthetic":true}\n', encoding="utf-8")
            digest = sha(path)
            self.assertEqual(verify_files(root, {"record.json": digest}), 1)
            path.write_text('{"synthetic":false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                verify_files(root, {"record.json": digest})

    def test_path_escape_and_absolute_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe.json").write_text("{}", encoding="utf-8")
            self.assertEqual(safe_path(root, "safe.json"), root / "safe.json")
            for invalid in ("../safe.json", "/safe.json", "..\\safe.json", ""):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    safe_path(root, invalid)

    def test_noncanonical_hash_and_duplicate_keys_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid SHA256"):
                verify_files(root, {"record.json": "A" * 64})
            path.write_text('{"x":1,"x":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                read_json(path)
            yaml_path = root / "config.yaml"
            yaml_path.write_text("x: 1\nx: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate/nonstring YAML key"):
                read_yaml(yaml_path)


if __name__ == "__main__":
    unittest.main()
