"""Block file and network access while a student function runs.

Student code may only read data through the ``DataView`` it is handed, which
never returns anything dated after ``as_of``. Reading files or calling the
network directly would bypass that, so those calls raise ``LookAheadGuardError``
while a student function is running. ``DataView`` itself is allowed through.

Load anything you trained offline at import time (top of your module), not
inside ``exposure()`` / ``value()`` / ``filter_candidates()``. Date-indexed
inputs you build yourself (for example news sentiment scores) go in
``student_data/<name>.parquet`` and are read with ``data.extra(name)``.
"""
from __future__ import annotations

import builtins
import io
import socket
import sqlite3
import threading
from contextlib import contextmanager

import pandas as pd


class LookAheadGuardError(RuntimeError):
    pass


_state = threading.local()


def _allowed() -> bool:
    return getattr(_state, "allow_depth", 0) > 0 or getattr(_state, "guard_depth", 0) == 0


@contextmanager
def allow_io():
    """Used by DataView: its own reads are point-in-time and may touch disk."""
    _state.allow_depth = getattr(_state, "allow_depth", 0) + 1
    try:
        yield
    finally:
        _state.allow_depth -= 1


def _blocked(name, original):
    def wrapper(*args, **kwargs):
        if not _allowed():
            raise LookAheadGuardError(
                f"{name}() was called inside a student function. Read data through the "
                f"DataView argument (data.close(), data.fundamentals(), data.extra(), ...) "
                f"or load offline files at import time.")
        return original(*args, **kwargs)
    return wrapper


_TARGETS = [
    (builtins, "open"), (io, "open"), (sqlite3, "connect"), (socket.socket, "connect"),
    (pd, "read_parquet"), (pd, "read_csv"), (pd, "read_excel"), (pd, "read_json"),
    (pd, "read_sql"), (pd, "read_pickle"), (pd, "read_feather"),
]
_lock = threading.Lock()
_installed = False


def _install():
    global _installed
    with _lock:
        if _installed:
            return
        for owner, attr in _TARGETS:
            original = getattr(owner, attr)
            setattr(owner, attr, _blocked(f"{getattr(owner, '__name__', owner)}.{attr}", original))
        _installed = True


@contextmanager
def student_call():
    """Wrap every call into student code."""
    _install()
    _state.guard_depth = getattr(_state, "guard_depth", 0) + 1
    try:
        yield
    finally:
        _state.guard_depth -= 1
