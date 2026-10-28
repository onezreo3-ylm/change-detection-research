"""Local, process-bound GPU polling; no model changes or chat wakeups.

All paths and thresholds are in YAML. A persistent Windows process handle plus
creation time, executable and command-line checks prevent PID-reuse mistakes.
This is a software pause policy, not a hard temperature cap or CPU monitoring.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.guard-tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def load_settings(path: Path) -> dict:
    with path.open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    required = {'run_dir', 'process_id', 'creation_time_ticks', 'executable',
                'command_line', 'powershell', 'nvidia_smi', 'gpu_index',
                'poll_seconds', 'confirmation_seconds', 'confirmed_pause_c',
                'immediate_pause_c', 'confirmation_samples', 'max_sensor_failures'}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError('Unexpected or missing temperature guard settings')
    for name in ('process_id', 'creation_time_ticks', 'gpu_index', 'poll_seconds',
                 'confirmation_seconds', 'confirmed_pause_c', 'immediate_pause_c',
                 'confirmation_samples', 'max_sensor_failures'):
        value = settings[name]
        if type(value) is not int or value < (0 if name == 'gpu_index' else 1):
            raise ValueError(f'{name} must be an explicitly nonnegative/positive integer')
    if not settings['confirmed_pause_c'] < settings['immediate_pause_c'] < 120:
        raise ValueError('Expected confirmed_pause_c < immediate_pause_c < 120')
    for name in ('run_dir', 'executable', 'powershell', 'nvidia_smi'):
        candidate = Path(settings[name])
        if not candidate.is_absolute() or not candidate.exists():
            raise ValueError(f'{name} must be an existing absolute path')
    if not Path(settings['run_dir']).is_dir() or not settings['command_line'].strip():
        raise ValueError('A run directory and exact command line are required')
    return settings


class TemperaturePolicy:
    def __init__(self, settings: dict):
        self.settings = settings
        self.hot_samples = 0
        self.sensor_failures = 0

    def observe(self, temperature: int | None) -> str:
        if temperature is None:
            self.hot_samples = 0
            self.sensor_failures += 1
            return ('sensor_pause' if self.sensor_failures >= self.settings['max_sensor_failures']
                    else 'sensor_retry')
        if not math.isfinite(temperature) or not 0 <= temperature < 120:
            raise ValueError('Invalid GPU core temperature')
        self.sensor_failures = 0
        if temperature >= self.settings['immediate_pause_c']:
            return 'temperature_pause'
        self.hot_samples = (self.hot_samples + 1
                            if temperature >= self.settings['confirmed_pause_c'] else 0)
        if self.hot_samples >= self.settings['confirmation_samples']:
            return 'temperature_pause'
        return 'confirm' if self.hot_samples else 'ok'


class WindowsProcess:
    def __init__(self, settings: dict):
        if os.name != 'nt':
            raise RuntimeError('This operational helper is Windows-only')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.api.GetProcessTimes.restype = wintypes.BOOL
        self.api.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                       wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.api.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.WaitForSingleObject.restype = wintypes.DWORD
        self.api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateProcess.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.OpenProcess(0x00100000 | 0x1000 | 0x0001, False, settings['process_id'])
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not self.api.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            buffer = ctypes.create_unicode_buffer(32768)
            length = wintypes.DWORD(len(buffer))
            if not self.api.QueryFullProcessImageNameW(self.handle, 0, buffer, ctypes.byref(length)):
                raise ctypes.WinError(ctypes.get_last_error())
            if ticks != settings['creation_time_ticks'] or (
                    os.path.normcase(buffer.value) != os.path.normcase(settings['executable'])):
                raise ValueError('Training process identity changed; refusing to attach')
            command = (f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={settings['process_id']}'; "
                       '$p.CommandLine | ConvertTo-Json -Compress')
            observed = subprocess.run([settings['powershell'], '-NoProfile', '-NonInteractive',
                                       '-Command', command], capture_output=True, text=True,
                                      check=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            if json.loads(observed.stdout) != settings['command_line']:
                raise ValueError('Training command line changed; refusing to attach')
        except BaseException:
            self.close()
            raise

    def is_alive(self) -> bool:
        result = self.api.WaitForSingleObject(self.handle, 0)
        if result == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 258

    def stop(self) -> None:
        if self.is_alive() and not self.api.TerminateProcess(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        if self.api.WaitForSingleObject(self.handle, 10000) != 0:
            raise RuntimeError('Training process did not exit after pause request')

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def read_temperature(settings: dict) -> int:
    result = subprocess.run([settings['nvidia_smi'], '-i', str(settings['gpu_index']),
                             '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    value = int(result.stdout.strip())
    if not 0 <= value < 120:
        raise ValueError('GPU temperature outside the supported sensor range')
    return value


def pause_training(process, settings: dict, reason: str, temperature: int | None) -> dict:
    process.stop()
    # Imported only AFTER stopping; map_location=cpu never runs a model on the GPU.
    import torch
    run_dir = Path(settings['run_dir'])
    status = {'status': reason, 'kind': 'research', 'paused_utc': utc_now(),
              'temperature_c': temperature, 'manual_resume_required': True,
              'reason': 'Local temperature guard stopped the bound process; incomplete epoch discarded.'}
    try:
        checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1 or checkpoint.get('kind') != 'research':
            raise ValueError('Unexpected checkpoint format/kind')
        status.update(epoch=checkpoint['epoch'], epochs=checkpoint['config']['train']['epochs'],
                      best_epoch=checkpoint['best_epoch'], best_val_f1=checkpoint['best_val_f1'])
    except Exception as error:
        status['checkpoint_verification_error'] = f'{type(error).__name__}: {error}'
    write_json(run_dir / 'status.json', status)
    return status


def monitor(settings: dict, process, *, read=read_temperature, sleep=time.sleep,
            pause=pause_training) -> dict:
    run_dir = Path(settings['run_dir'])
    lock = run_dir / 'temperature_guard.lock'
    # A second guard must fail before it can touch the training process or state.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    policy = TemperaturePolicy(settings)
    state = {'status': 'monitoring', 'guard_pid': os.getpid(), 'training_pid': settings['process_id'],
             'started_utc': utc_now(), 'settings': settings,
             'guard_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        while process.is_alive():
            temperature = None
            sensor_error = None
            try:
                temperature = read(settings)
            except Exception as error:
                sensor_error = f'{type(error).__name__}: {error}'
            decision = policy.observe(temperature)
            state.update(last_sample_utc=utc_now(), temperature_c=temperature,
                         sensor_error=sensor_error, decision=decision,
                         hot_samples=policy.hot_samples, sensor_failures=policy.sensor_failures)
            if decision in ('temperature_pause', 'sensor_pause'):
                reason = ('paused_temperature_guard' if decision == 'temperature_pause'
                          else 'paused_temperature_unavailable')
                # Avoid overwriting a completed run if it exited during the sensor query.
                if process.is_alive():
                    state['training_pause'] = pause(process, settings, reason, temperature)
                    state['status'] = reason
                else:
                    state['status'] = 'training_process_exited'
                write_json(run_dir / 'temperature_guard_state.json', state)
                return state
            write_json(run_dir / 'temperature_guard_state.json', state)
            sleep(settings['confirmation_seconds'] if decision == 'confirm' else settings['poll_seconds'])
        state.update(status='training_process_exited', ended_utc=utc_now())
        write_json(run_dir / 'temperature_guard_state.json', state)
        return state
    except Exception as error:
        state.update(status='guard_error', error=f'{type(error).__name__}: {error}')
        if process.is_alive():
            state['training_pause'] = pause(process, settings, 'paused_temperature_guard_error', None)
        write_json(run_dir / 'temperature_guard_state.json', state)
        raise
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Read-only identity/sensor smoke; never stops training')
    args = parser.parse_args()
    settings = load_settings(args.config)
    process = WindowsProcess(settings)
    try:
        if args.check:
            print(json.dumps({'check': 'passed', 'pid': settings['process_id'],
                              'process_alive': process.is_alive(), 'temperature_c': read_temperature(settings)}))
        else:
            print(json.dumps(monitor(settings, process), ensure_ascii=False))
    finally:
        process.close()


if __name__ == '__main__':
    main()
