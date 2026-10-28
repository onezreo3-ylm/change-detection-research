"""Query process identity and termination together; an open handle is not liveness."""
from __future__ import annotations

import ctypes
from ctypes import wintypes


def classify_process(pid, actual_ticks, expected_ticks, wait_result, exit_code):
    if wait_result not in (0, 258):
        raise OSError("Process wait query failed; liveness is unknown")
    identity_matches = actual_ticks == expected_ticks
    active = wait_result == 258 and exit_code == 259
    return {"pid": pid, "creation_ticks": actual_ticks, "identity_matches": identity_matches,
            "exit_code": exit_code, "wait_result": wait_result,
            "active": active, "owned_active": active and identity_matches}


def process_snapshot(pid, expected_ticks):
    if type(pid) is not int or pid <= 0 or type(expected_ticks) is not int:
        raise ValueError("Explicit PID and creation identity required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error in (87, 1168):
            return {"pid": pid, "missing": True, "owned_active": False, "active": False}
        raise ctypes.WinError(error)
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        result = kernel.WaitForSingleObject(handle, 0)
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return classify_process(pid, ticks, expected_ticks, result, code.value)
    finally:
        kernel.CloseHandle(handle)
