import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.recover_b0e_native_json import native_json_bytes
from scripts.recover_b0e_analysis_serialization import require_same_prefix


class NativeJsonRecoveryTests(unittest.TestCase):
    def test_real_original_windows_writer_partial_is_a_byte_exact_prefix(self):
        value={'metric':.123456789,'primary':{'source_paired_lower_positive':np.bool_(True)}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'failed.json'
            with self.assertRaises(TypeError):
                with p.open('w',encoding='utf-8',newline='\r\n') as f:json.dump(value,f,indent=2,allow_nan=False)
            partial=p.read_bytes()
            self.assertIn(b'\r\n',partial)
            require_same_prefix(partial,native_json_bytes(value))
            with self.assertRaises(ValueError):require_same_prefix(partial,native_json_bytes(value).replace(b'\r\n',b'\n'))

    def test_embedded_newline_and_floats_are_preserved_without_double_CR(self):
        value={'text':'a\nb','number':.987654321,'check':np.bool_(False)}
        encoded=native_json_bytes(value)
        self.assertNotIn(b'\r\r\n',encoded)
        self.assertEqual(json.loads(encoded),{'text':'a\nb','number':.987654321,'check':False})

    def test_nonfinite_or_unrelated_types_still_fail(self):
        with self.assertRaises(ValueError):native_json_bytes({'x':float('nan')})
        with self.assertRaises(TypeError):native_json_bytes({'x':object()})


if __name__=='__main__':unittest.main()
