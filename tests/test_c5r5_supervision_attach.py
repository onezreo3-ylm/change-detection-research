from datetime import datetime,timezone,timedelta
import unittest
from scripts.c5r5_supervision_attach import validate_attachment,remaining_seconds


class CpuAttachmentTests(unittest.TestCase):
    def fixture(self):
        return {'owned_active':False},{'owned_active':True,'identity_matches':True,'pid':12,'creation_ticks':34},{
            'status':'running','stage':'c5_training','completed_roles':[],'child_pid':12,'child_creation_ticks':34}

    def test_only_a_live_existing_c5_with_a_gone_owner_can_be_attached(self):
        validate_attachment(*self.fixture())
        values=list(self.fixture());values[0]['owned_active']=True
        with self.assertRaises(ValueError):validate_attachment(*values)

    def test_reused_pid_dead_worker_or_r5_stage_cannot_be_attached(self):
        for field,value in [('identity_matches',False),('owned_active',False),('creation_ticks',35)]:
            values=list(self.fixture());values[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_attachment(*values)
        values=list(self.fixture());values[2]['stage']='r5_training'
        with self.assertRaises(ValueError):validate_attachment(*values)

    def test_time_budget_keeps_engineering_elapsed_and_unlogged_age(self):
        now=datetime(2026,10,21,tzinfo=timezone.utc)
        progress={'elapsed_seconds':1000,'updated_utc':(now-timedelta(seconds=12)).isoformat()}
        self.assertEqual(remaining_seconds(progress,300,now),86400-300-1000-12)

    def test_exhausted_time_or_future_clock_is_not_reset(self):
        now=datetime(2026,10,21,tzinfo=timezone.utc)
        with self.assertRaises(RuntimeError):remaining_seconds({'elapsed_seconds':86300,'updated_utc':now.isoformat()},300,now)
        with self.assertRaises(ValueError):remaining_seconds({'elapsed_seconds':1,'updated_utc':(now+timedelta(seconds=5)).isoformat()},300,now)
