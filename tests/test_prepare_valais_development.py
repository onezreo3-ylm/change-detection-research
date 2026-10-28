import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import yaml

from pcdc.config import load_config
from scripts.prepare_valais_development import run


def tiff(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="TIFF")
    return buffer.getvalue()


class PrepareValaisDevelopmentTests(unittest.TestCase):
    def test_new_dataset_config_is_valid_and_test_access_disabled(self):
        config = load_config("configs/valais_clean_development_seed42.yaml")
        self.assertFalse(config.run_test_after_training)
        self.assertEqual(config.values["data"]["label_values"], [0, 1])
        self.assertEqual(config.values["data"]["tile_size"], 256)

    def test_materializer_uses_only_verified_train_val(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "archives"
            archives.mkdir()
            spec = {"dataset_revision": "fixture", "splits": {}}
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            label = np.zeros((256, 256), dtype=np.uint8)
            for split in ("train", "val"):
                archive = archives / f"{split}.zip"
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr(f"{split}/2017/1.tif", tiff(image))
                    zf.writestr(f"{split}/2023/1.tif", tiff(image))
                    zf.writestr(f"{split}/labels/1.tif", tiff(label))
                    zf.writestr(f"{split}/test_imgs_filtered_1000_2.pkl", b"untrusted")
                spec["splits"][split] = {
                    "size_bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "expected_pairs": 1,
                }
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(spec), encoding="utf-8")
            output = root / "prepared"
            report = run(config, archives, output)
            self.assertFalse(report["official_test_archive_accessed"])
            self.assertTrue((output / "train" / "A" / "1.tif").is_file())
            self.assertTrue((output / "val" / "label" / "1.tif").is_file())
            self.assertFalse((output / "test").exists())
            with self.assertRaises(FileExistsError):
                run(config, archives, output)


if __name__ == "__main__":
    unittest.main()
