from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time

_RECORD = Path(".runtime/gpu-sampler.json")
_LOCK = Path(".runtime/gpu-sampler.lock")


class SamplerRecordError(RuntimeError):
    pass


@dataclass(frozen=True)
class SamplerRecord:
    schema: int
    pid: int
    process_start: str
    command_sha256: str


def _paths(path: Path | None) -> tuple[Path, Path]:
    record = path or _RECORD
    if record.name != _RECORD.name or record.parent.name != _RECORD.parent.name:
        raise SamplerRecordError("the sampler record must be .runtime/gpu-sampler.json")
    return record, record.with_name(_LOCK.name)


def _runtime(record: Path) -> None:
    runtime = record.parent
    if runtime.is_symlink():
        raise SamplerRecordError(".runtime must be a real private directory")
    try:
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = runtime.lstat()
    except OSError as error:
        raise SamplerRecordError(".runtime cannot be opened safely") from error
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
        raise SamplerRecordError(".runtime must be a mode-700 directory")


@contextmanager
def _locked(record: Path, lock: Path):
    _runtime(record)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as error:
        raise SamplerRecordError("the sampler lock cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise SamplerRecordError("the sampler lock must be a mode-600 regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SamplerRecordError("another sampler lifecycle operation is active") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _ps(pid: int, field: str) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", f"{field}="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SamplerRecordError("the sampler process identity could not be inspected") from error
    value = result.stdout.strip()
    if result.returncode and not value:
        return None
    if result.returncode or not value:
        raise SamplerRecordError("the sampler process identity could not be inspected")
    return value


def _identity_or_none(pid: int) -> tuple[str, str] | None:
    if type(pid) is not int or pid <= 1:
        raise SamplerRecordError("refusing an unsafe sampler PID")
    state = _ps(pid, "state")
    if state is None or state.startswith("Z"):
        return None
    process_start = _ps(pid, "lstart")
    if process_start is None:
        return None
    command = _ps(pid, "command")
    confirmed_start = _ps(pid, "lstart")
    confirmed_state = _ps(pid, "state")
    if command is None or confirmed_start is None or confirmed_state is None or confirmed_state.startswith("Z"):
        return None
    if confirmed_start != process_start:
        raise SamplerRecordError("the sampler process identity changed during inspection")
    return process_start, hashlib.sha256(command.encode()).hexdigest()


def _validate(record: SamplerRecord) -> None:
    if (
        type(record.schema) is not int
        or record.schema != 1
        or type(record.pid) is not int
        or record.pid <= 1
        or not isinstance(record.process_start, str)
        or not record.process_start
        or len(record.process_start) > 128
        or any(ord(character) < 32 for character in record.process_start)
        or not isinstance(record.command_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", record.command_sha256)
    ):
        raise SamplerRecordError("the sampler ownership record is invalid")


def _read(record_path: Path) -> SamplerRecord:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(record_path, flags)
    except FileNotFoundError as error:
        raise SamplerRecordError("the sampler ownership record is missing") from error
    except OSError as error:
        raise SamplerRecordError("the sampler ownership record cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise SamplerRecordError("the sampler ownership record must be a mode-600 regular file")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            payload = json.load(stream)
        if not isinstance(payload, dict) or set(payload) != set(SamplerRecord.__dataclass_fields__):
            raise ValueError
        result = SamplerRecord(**payload)
        _validate(result)
        return result
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SamplerRecordError("the sampler ownership record is unreadable or invalid") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write(record_path: Path, record: SamplerRecord) -> None:
    _validate(record)
    if record_path.exists() or record_path.is_symlink():
        raise SamplerRecordError("a sampler ownership record already exists")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=record_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(asdict(record), stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, record_path)
        directory = os.open(record_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise SamplerRecordError("a sampler ownership record already exists") from error
    except OSError as error:
        raise SamplerRecordError("the sampler ownership record cannot be written safely") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _matches_or_none(record: SamplerRecord) -> bool:
    identity = _identity_or_none(record.pid)
    if identity is None:
        return False
    if identity != (record.process_start, record.command_sha256):
        raise SamplerRecordError("the live sampler process does not match its ownership record")
    return True


def _remove(record_path: Path) -> None:
    record_path.unlink(missing_ok=True)


def create_sampler_record(pid: int, *, path: Path | None = None) -> SamplerRecord:
    record_path, lock_path = _paths(path)
    with _locked(record_path, lock_path):
        identity = _identity_or_none(pid)
        if identity is None:
            raise SamplerRecordError("the sampler process is no longer running")
        record = SamplerRecord(1, pid, *identity)
        _write(record_path, record)
        return record


def verify_sampler_record(*, path: Path | None = None) -> SamplerRecord:
    record_path, lock_path = _paths(path)
    with _locked(record_path, lock_path):
        record = _read(record_path)
        if not _matches_or_none(record):
            _remove(record_path)
            raise SamplerRecordError("the recorded sampler process is no longer running")
        return record


def _wait(record: SamplerRecord, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if not _matches_or_none(record):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _signal(record: SamplerRecord, signal_number: int) -> bool:
    if not _matches_or_none(record):
        return False
    try:
        os.kill(record.pid, signal_number)
    except ProcessLookupError:
        if not _matches_or_none(record):
            return False
        raise SamplerRecordError("the verified sampler process could not be signaled") from None
    except OSError as error:
        raise SamplerRecordError("the verified sampler process could not be signaled") from error
    return True


def stop_sampler(*, timeout_seconds: float = 10.0, path: Path | None = None) -> bool:
    if (
        not isinstance(timeout_seconds, int | float)
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or not 0 <= timeout_seconds <= 300
    ):
        raise ValueError("timeout_seconds must be finite and between 0 and 300")
    record_path, lock_path = _paths(path)
    with _locked(record_path, lock_path):
        if not record_path.exists() and not record_path.is_symlink():
            return False
        record = _read(record_path)
        if not _matches_or_none(record):
            _remove(record_path)
            return False
        if not _signal(record, signal.SIGTERM) or _wait(record, timeout_seconds):
            _remove(record_path)
            return True
        if not _signal(record, signal.SIGKILL) or _wait(record, timeout_seconds):
            _remove(record_path)
            return True
        raise SamplerRecordError("the verified sampler process did not stop; ownership was retained")
