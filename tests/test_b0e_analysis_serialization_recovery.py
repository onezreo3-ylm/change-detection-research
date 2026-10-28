import json
import unittest

import numpy as np

from scripts import analyze_b0e_replication as original
from scripts.recover_b0e_analysis_serialization import (
    normalize_numpy_booleans,require_invalid_partial,require_same_prefix,strict_bytes,
)


class SerializationRecoveryTests(unittest.TestCase):
    def test_only_numpy_booleans_convert_without_numeric_changes(self):
        value={'checks':[np.bool_(True),np.bool_(False)],'float':.123456789,'int':17}
        r=normalize_numpy_booleans(value)
        self.assertIs(r['checks'][0],True);self.assertIs(r['checks'][1],False)
        self.assertEqual(r['float'],value['float']);self.assertIs(type(r['int']),int)
        self.assertEqual(json.loads(strict_bytes(value)),r)

    def test_actual_frozen_primary_gate_serializes_without_changing_decision(self):
        records={s:{'b0e':dict(corruption_mf1=.8,clean_f1=.9,pooled_recall=.8,background_fpr=.01),
                    'b1':dict(corruption_mf1=.82,clean_f1=.9,pooled_recall=.82,background_fpr=.009)} for s in original.SEEDS}
        gate=original.primary_gate(records,np.array([.01,.02,.03]),original.load_config()['rules'])
        self.assertIsInstance(gate['checks']['source_paired_lower_positive'],np.bool_)
        result=json.loads(strict_bytes(gate))
        self.assertEqual(result['decision'],gate['decision'])
        self.assertEqual(result['mean_deltas'],gate['mean_deltas'])
        self.assertEqual(result['source_paired_interval'],gate['source_paired_interval'])
        self.assertTrue(result['all_passed'])

    def test_unknown_types_and_nonfinite_values_remain_rejected(self):
        for value in (dict(x=np.int64(3)),dict(x=object())):
            with self.assertRaises(TypeError):strict_bytes(value)
        for value in (float('nan'),float('inf'),np.float64(float('nan'))):
            with self.assertRaises(ValueError):strict_bytes(dict(x=value))

    def test_valid_analysis_or_unrelated_partial_cannot_be_replaced(self):
        with self.assertRaises(ValueError):require_invalid_partial(b'{"status":"completed"}')
        with self.assertRaises(ValueError):require_invalid_partial(b'{"other": ')
        require_invalid_partial(b'{"source_paired_lower_positive": ')

    def test_entire_existing_numeric_prefix_must_be_byte_identical(self):
        require_same_prefix(b'{"x": 0.123,',b'{"x": 0.123, "b": true}')
        with self.assertRaises(ValueError):require_same_prefix(b'{"x": 0.123,',b'{"x": 0.124, "b": true}')


if __name__=='__main__':unittest.main()
