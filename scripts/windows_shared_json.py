"""Atomic JSON with a cross-process Windows mutex around read/replace.

This filesystem rejects replacement while a reader holds the destination open,
including a delete-sharing handle. Serialize short snapshots with the kernel;
never reuse stale data or relax a supervisor heartbeat deadline.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import hashlib
import os
from pathlib import Path
import time


def sharing_operation(operation, *, timeout=0.25):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return operation()
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", 5) not in (5,32,33) or time.monotonic() >= deadline:
                raise
            # Bounded filesystem contention handling, never a worker/GPU retry.
            time.sleep(0.01)


@contextmanager
def snapshot_mutex(path):
    path = Path(path)
    if os.name != "nt":
        yield
        return
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateMutexW.argtypes = [wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR]
    api.CreateMutexW.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.ReleaseMutex.argtypes = [wintypes.HANDLE]
    api.ReleaseMutex.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    name = "Local\\PCDCResearchJSON_" + hashlib.sha256(os.path.normcase(str(path.resolve())).encode()).hexdigest()
    handle = api.CreateMutexW(None,False,name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        waited = api.WaitForSingleObject(handle,1000)
        if waited not in (0,128):
            if waited == 258:
                raise TimeoutError("JSON snapshot mutex exceeded one second: " + str(path))
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = True
        yield
    finally:
        if acquired:
            api.ReleaseMutex(handle)
        api.CloseHandle(handle)


@contextmanager
def shared_text_reader(path):
    with snapshot_mutex(path):
        stream = sharing_operation(lambda: Path(path).open("r",encoding="utf-8-sig"))
        with stream:
            yield stream


def read_json(path):
    with shared_text_reader(path) as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exactly one writer owns each heartbeat/ledger file. Keep temp identity
    # separate between processes; failures leave the completed old snapshot.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with snapshot_mutex(path):
        sharing_operation(lambda: os.replace(temporary, path))
