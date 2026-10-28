import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from scripts.audit_valais_local import audit_split


def _tiff(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="TIFF")
    return buffer.getvalue()


class ValaisLocalAuditTests(unittest.TestCase):
    def test_audits_decoded_pair_and_binary_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "val.zip"
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            label = np.zeros((256, 256), dtype=np.uint8)
            label[1:3, 1:3] = 1
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("val/2017/1.tif", _tiff(rgb))
                rgb[0, 0, 0] = 9
                zf.writestr("val/2023/1.tif", _tiff(rgb))
                zf.writestr("val/labels/1.tif", _tiff(label))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            result = audit_split(archive, "val", archive.stat().st_size, digest)
            self.assertEqual(result["pairs"], 1)
            self.assertEqual(result["positive_pixels"], 4)
            self.assertEqual(result["label_values"], [0, 1])
            with self.assertRaises(ValueError):
                audit_split(archive, "test", archive.stat().st_size, digest)


if __name__ == "__main__":
    unittest.main()
