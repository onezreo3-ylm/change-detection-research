"""Engineering-only simulations of the operational monitor; never research data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.temperature_guard import TemperaturePolicy, monitor


def settings(root):
    return {'run_dir': str(root), 'process_id': 123, 'poll_seconds': 10,
            'confirmation_seconds': 5, 'confirmed_pause_c': 85,
            'immediate_pause_c': 86, 'confirmation_samples': 3, 'max_sensor_failures': 3}


class TemperatureGuardTests(unittest.TestCase):
    def test_normal_temperature_and_old_76_rule_do_not_pause(self):
        policy = TemperaturePolicy(settings('.'))
        for temperature in (44, 75, 76, 80, 84):
            self.assertEqual(policy.observe(temperature), 'ok')

    def test_three_confirmations_reset_when_cool(self):
        policy = TemperaturePolicy(settings('.'))
        self.assertEqual([policy.observe(t) for t in (85, 85, 84, 85, 85, 85)],
                         ['confirm', 'confirm', 'ok', 'confirm', 'confirm', 'temperature_pause'])

    def test_86_pauses_on_first_sample(self):
        self.assertEqual(TemperaturePolicy(settings('.')).observe(86), 'temperature_pause')

    def test_sensor_failures_fail_closed_and_reset(self):
        policy = TemperaturePolicy(settings('.'))
        self.assertEqual([policy.observe(t) for t in (None, None, 70, None, None, None)],
                         ['sensor_retry', 'sensor_retry', 'ok', 'sensor_retry', 'sensor_retry', 'sensor_pause'])

    def test_monitor_confirms_pauses_once_and_cleans_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock()
            process.is_alive.return_value = True
            pause = Mock(return_value={'epoch': 41})
            sleeper = Mock()
            result = monitor(settings(root), process, read=Mock(side_effect=[75, 85, 85, 85]),
                             sleep=sleeper, pause=pause)
            self.assertEqual(result['status'], 'paused_temperature_guard')
            pause.assert_called_once_with(process, settings(root), 'paused_temperature_guard', 85)
            self.assertEqual([call.args[0] for call in sleeper.call_args_list], [10, 5, 5])
            self.assertFalse((root / 'temperature_guard.lock').exists())
            self.assertEqual(json.loads((root / 'temperature_guard_state.json').read_text())['temperature_c'], 85)

    def test_sensor_error_stops_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = Mock()
            process.is_alive.return_value = True
            pause = Mock(return_value={})
            result = monitor(settings(Path(temporary)), process, read=Mock(side_effect=RuntimeError('sensor unavailable')),
                             sleep=Mock(), pause=pause)
            self.assertEqual(result['status'], 'paused_temperature_unavailable')
            pause.assert_called_once()

    def test_exited_training_is_not_stopped_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'status.json').write_text('{"status":"completed"}', encoding='utf-8')
            process = Mock()
            process.is_alive.side_effect = [True, False]
            pause = Mock()
            result = monitor(settings(root), process, read=Mock(return_value=86), sleep=Mock(), pause=pause)
            pause.assert_not_called()
            self.assertEqual(result['status'], 'training_process_exited')
            self.assertEqual(json.loads((root / 'status.json').read_text())['status'], 'completed')

    def test_duplicate_guard_never_touches_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'temperature_guard.lock').write_text('existing guard', encoding='utf-8')
            process, pause = Mock(), Mock()
            with self.assertRaises(FileExistsError):
                monitor(settings(root), process, read=Mock(return_value=86), sleep=Mock(), pause=pause)
            process.is_alive.assert_not_called()
            pause.assert_not_called()


if __name__ == '__main__':
    unittest.main()
