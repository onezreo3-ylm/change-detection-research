import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from scripts import tiling_probe_supervision as s


class TilingSupervisionTests(unittest.TestCase):
    def ledger(self, directory, entries=None, total=None, cap=172800):
        entries = entries or []
        path = Path(directory) / "ledger.json"
        path.write_text(json.dumps({"entries": entries, "total_charged_seconds": sum(e['charged_seconds'] for e in entries) if total is None else total,
                                    "batch_upper_bound_seconds": cap}), encoding="utf-8")
        return {"ledger_path": str(path), "pilot_wall_limit_seconds": 900,
                "main_wall_limit_seconds": 1800, "whole_probe_limit_seconds": 3600}

    def test_owner_guard_module_imports_without_torch(self):
        r = subprocess.run([sys.executable, "-c", "import sys; import scripts.tiling_probe_supervision; assert 'torch' not in sys.modules"],
                           cwd=s.ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_preserves_shared_cap_and_charges_existing_probe_spend(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.ledger(d, [{"kind": "tiling_probe_gpu_pilot", "charged_seconds": 3590, "status": "failed_preserve"}])
            _, limit = s.reserve(cfg, "main")
            self.assertEqual(limit, 10)
        with tempfile.TemporaryDirectory() as d:
            cfg = self.ledger(d, total=172750)
            _, limit = s.reserve(cfg, "pilot")
            self.assertEqual(limit, 50)

    def test_no_running_reservation_rerun_training_or_budget_reset(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self.ledger(d)
            s.reserve(cfg, "pilot")
            with self.assertRaises(RuntimeError): s.reserve(cfg, "pilot")
            with self.assertRaises(RuntimeError): s.reserve(cfg, "main")
            with self.assertRaises(ValueError): s.reserve(cfg, "train")
        with tempfile.TemporaryDirectory() as d:
            cfg = self.ledger(d, [{"id": "tiling_probe_gpu_pilot_20261025", "charged_seconds": 5, "status": "completed"}])
            with self.assertRaises(RuntimeError): s.reserve(cfg, "pilot")

    def test_runtime_rejects_weakened_temperature_or_longer_cap(self):
        path = s.ROOT / "configs/levir_tiling_runtime_20261025.yaml"
        s.load_runtime(path)
        original = yaml.safe_load(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "runtime.yaml"
            for key, value in [("immediate_pause_c", 90), ("poll_seconds", 10), ("main_wall_limit_seconds", 3600), ("automatic_retry", True)]:
                cfg = copy.deepcopy(original); cfg['study_config'] = str(s.ROOT / 'configs/levir_tiling_context_probe_20261025.yaml'); cfg[key] = value
                p.write_text(yaml.safe_dump(cfg), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError): s.load_runtime(p)

    def test_gpu_identity_device_mapping_and_sensor_timeout(self):
        _, cfg = s.load_runtime(s.ROOT / "configs/levir_tiling_runtime_20261025.yaml")
        with patch.dict(os.environ, {}, clear=True), patch.object(s.subprocess, 'run', return_value=SimpleNamespace(stdout=cfg['gpu_uuid']+',50,0,1000\n')) as run:
            self.assertEqual(s.gpu_state(cfg)['temperature_c'], 50)
            self.assertEqual(run.call_args.kwargs['timeout'], 1)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), self.assertRaises(ValueError): s.gpu_state(cfg)
        with patch.dict(os.environ, {}, clear=True), patch.object(s.subprocess, 'run', return_value=SimpleNamespace(stdout='GPU-other,50,0,1000\n')), self.assertRaises(ValueError):
            s.gpu_state(cfg)

    def test_worker_fails_before_cuda_when_bound_supervisors_are_dead(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {'study_output': d, 'startup_limit_seconds': 1}
            session = Path(d) / 'pilot_session'; session.mkdir()
            proof = {'worker': {'process_id': os.getpid(), 'creation_time_ticks': 1}, 'owner': {}, 'guard': {}, 'watchdog': {}}
            class DeadHandle:
                def __init__(self, settings): pass
                def is_alive(self): return False
                def close(self): pass
            with patch.object(s, 'load_runtime', return_value=(Path(d)/'runtime.yaml',cfg)), patch.object(s, 'attachment', return_value=proof), patch.object(s.shared,'creation_ticks',return_value=1), patch.object(s,'WindowsProcess',DeadHandle):
                with self.assertRaises(RuntimeError): s.worker('unused','pilot')
            self.assertEqual(s.read(session/'worker_result.json')['status'],'failed_preserve')

    def test_independent_guard_stops_bound_worker_on_dead_owner(self):
        with tempfile.TemporaryDirectory() as d:
            session=Path(d)/'pilot_session'; session.mkdir()
            _,cfg=s.load_runtime(s.ROOT/'configs/levir_tiling_runtime_20261025.yaml');cfg['study_output']=d
            proof={r:{'role':r} for r in ['worker','owner','guard','watchdog']}
            proof['guard'].update(process_id=os.getpid(),creation_time_ticks=1)
            handles={}
            class FakeHandle:
                def __init__(self, settings): self.role=settings['role'];self.active=self.role!='owner';self.stopped=False;handles[self.role]=self
                def is_alive(self): return self.active
                def stop(self): self.stopped=True;self.active=False
                def close(self): pass
            with patch.object(s,'load_runtime',return_value=(Path(d)/'runtime.yaml',cfg)),patch.object(s,'attachment',return_value=proof),patch.object(s.shared,'creation_ticks',return_value=1),patch.object(s,'WindowsProcess',FakeHandle):
                result=s.guard_loop('unused','pilot','guard')
            self.assertTrue(handles['worker'].stopped)
            self.assertEqual(result['status'],'stopped_worker_preserve')

    def test_failed_test_evidence_cannot_seal_real_execution(self):
        with tempfile.TemporaryDirectory() as d:
            checks=Path(d)/'checks.json';checks.write_text(json.dumps({'passed':False}))
            smoke=Path(d)/'smoke.json';smoke.write_text(json.dumps({'status':'passed','synthetic_only':True}))
            with self.assertRaises(ValueError):s.seal(s.ROOT/'configs/levir_tiling_runtime_20261025.yaml',checks,smoke)


if __name__=='__main__':unittest.main()
