from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_GATE = """import json, os, sys
fd = int(sys.argv[1])
with os.fdopen(fd, "rb") as stream:
    command = json.load(stream)
if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
    raise SystemExit(125)
os.execvpe(command[0], command, os.environ)
"""


class RecordError(RuntimeError):
    pass


class StaleProcessError(RecordError):
    pass


@dataclass(frozen=True)
class ProcessRecord:
    schema: int
    pid: int
    start_ticks: int
    command_sha256: str
    profile: str
    port: int
    source_sha: str
    log_path: str


def _validate_record(record: ProcessRecord) -> None:
    if record.schema != 1 or record.pid <= 1 or record.start_ticks < 1:
        raise RecordError("invalid process record identity")
    if not re.fullmatch(r"[0-9a-f]{64}", record.command_sha256):
        raise RecordError("invalid process command hash")
    if record.profile not in {"pi0_aloha_sim", "pi05_aloha_base"} or not 1 <= record.port <= 65535:
        raise RecordError("invalid process profile or port")
    if not _SHA.fullmatch(record.source_sha):
        raise RecordError("invalid process source SHA")
    log_path = Path(record.log_path)
    if log_path.is_absolute() or ".." in log_path.parts or not record.log_path.startswith(".runtime/"):
        raise RecordError("invalid process log path")


def _identity(pid: int, proc_root: Path) -> tuple[int, str]:
    if pid <= 1:
        raise RecordError("refusing unsafe PID")
    process_dir = proc_root / str(pid)
    try:
        raw_stat = (process_dir / "stat").read_text(encoding="utf-8")
        command = (process_dir / "cmdline").read_bytes()
    except FileNotFoundError as error:
        raise StaleProcessError("recorded process no longer exists") from error
    marker = raw_stat.rfind(") ")
    fields = raw_stat[marker + 2 :].split() if marker >= 0 else []
    if len(fields) <= 19 or not fields[19].isdigit() or not command:
        raise RecordError("cannot validate recorded process identity")
    return int(fields[19]), hashlib.sha256(command).hexdigest()


def create_record(
    path: Path,
    *,
    pid: int,
    profile: str,
    port: int,
    source_sha: str,
    log_path: str,
    proc_root: Path = Path("/proc"),
) -> ProcessRecord:
    if path.exists() or path.is_symlink():
        raise RecordError("process record already exists")
    start_ticks, command_sha256 = _identity(pid, proc_root)
    record = ProcessRecord(1, pid, start_ticks, command_sha256, profile, port, source_sha, log_path)
    _validate_record(record)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        os.chmod(temporary, 0o600)
        json.dump(asdict(record), stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def read_record(path: Path) -> ProcessRecord:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RecordError("process record is missing") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecordError("process record must be a regular file")
    if metadata.st_mode & 0o077:
        raise RecordError("process record permissions must be 0600")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != set(ProcessRecord.__dataclass_fields__):
            raise ValueError
        record = ProcessRecord(**data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RecordError("process record is malformed") from error
    try:
        _validate_record(record)
    except TypeError as error:
        raise RecordError("process record is malformed") from error
    return record


def verify_record(path: Path, proc_root: Path = Path("/proc")) -> ProcessRecord:
    record = read_record(path)
    start_ticks, command_sha256 = _identity(record.pid, proc_root)
    if (start_ticks, command_sha256) != (record.start_ticks, record.command_sha256):
        raise RecordError("recorded PID identity does not match; refusing to signal")
    return record


def hold_record(
    path: Path,
    *,
    profile: str,
    port: int,
    source_sha: str,
    proc_root: Path = Path("/proc"),
    poll_seconds: float = 1.0,
) -> None:
    expected = verify_record(path, proc_root)
    if (expected.profile, expected.port, expected.source_sha) != (profile, port, source_sha):
        raise RecordError("process record does not match the requested holder")
    while True:
        try:
            current = verify_record(path, proc_root)
        except StaleProcessError:
            return
        except RecordError:
            if not path.exists() and not path.is_symlink():
                return
            raise
        if current != expected:
            raise RecordError("process record changed while held")
        time.sleep(poll_seconds)


def signal_record(
    path: Path,
    signal_number: int,
    *,
    proc_root: Path = Path("/proc"),
    pidfd_open: Callable[[int], int] | None = None,
    pidfd_send_signal: Callable[[int, int], None] | None = None,
    close: Callable[[int], None] = os.close,
) -> int:
    if signal_number not in {signal.SIGTERM, signal.SIGKILL}:
        raise RecordError("unsupported process signal")
    record = read_record(path)
    open_pidfd = pidfd_open or getattr(os, "pidfd_open", None)
    send_signal = pidfd_send_signal or getattr(signal, "pidfd_send_signal", None)
    if open_pidfd is None or send_signal is None:
        raise RecordError("pidfd signaling is required to prevent PID-reuse races")
    try:
        pidfd = open_pidfd(record.pid)
    except OSError as error:
        raise StaleProcessError("recorded process no longer exists") from error
    try:
        verify_record(path, proc_root)
        send_signal(pidfd, signal_number)
    finally:
        close(pidfd)
    return record.pid


def launch_recorded_process(
    path: Path,
    *,
    command: list[str],
    expected_command_prefix: tuple[str, str],
    profile: str,
    port: int,
    source_sha: str,
    log_path: str,
    proc_root: Path = Path("/proc"),
) -> int:
    if not command or len(expected_command_prefix) != 2 or not all(expected_command_prefix):
        raise RecordError("launch command and identity prefix are required")
    open_pidfd = getattr(os, "pidfd_open", None)
    send_signal = getattr(signal, "pidfd_send_signal", None)
    if open_pidfd is None or send_signal is None:
        raise RecordError("pidfd process launch is required")

    _validate_record(ProcessRecord(1, 2, 1, "0" * 64, profile, port, source_sha, log_path))
    log = path.parent.parent / log_path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    read_gate, write_gate = os.pipe()
    try:
        with os.fdopen(descriptor, "ab") as stream:
            process = subprocess.Popen(
                [sys.executable, "-c", _GATE, str(read_gate)],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                pass_fds=(read_gate,),
            )
    finally:
        os.close(read_gate)

    pidfd = -1
    try:
        pidfd = open_pidfd(process.pid)
        initial_start, _ = _identity(process.pid, proc_root)
    except Exception:
        os.close(write_gate)
        process.wait(timeout=5)
        if pidfd >= 0:
            os.close(pidfd)
        raise

    try:
        with os.fdopen(write_gate, "w", encoding="utf-8") as stream:
            json.dump(command, stream)
        write_gate = -1
        matched = False
        for _ in range(50):
            if process.poll() is not None:
                raise RecordError("policy server exited during launch")
            start_ticks, _ = _identity(process.pid, proc_root)
            command_line = (proc_root / str(process.pid) / "cmdline").read_bytes().split(b"\0")
            expected = [value.encode() for value in expected_command_prefix]
            if start_ticks == initial_start and command_line[:2] == expected:
                matched = True
                break
            time.sleep(0.1)
        if not matched:
            raise RecordError("policy server command did not stabilize")
        record = create_record(
            path,
            pid=process.pid,
            profile=profile,
            port=port,
            source_sha=source_sha,
            log_path=log_path,
            proc_root=proc_root,
        )
        if record.start_ticks != initial_start:
            path.unlink(missing_ok=True)
            raise RecordError("launched PID identity changed")
    except Exception:
        try:
            send_signal(pidfd, signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            send_signal(pidfd, signal.SIGKILL)
            process.wait(timeout=10)
        raise
    finally:
        if write_gate >= 0:
            os.close(write_gate)
        os.close(pidfd)
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage an owned policy-server process record.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("path", type=Path)
    create.add_argument("pid", type=int)
    create.add_argument("profile")
    create.add_argument("port", type=int)
    create.add_argument("source_sha")
    create.add_argument("log_path")
    verify = subparsers.add_parser("verify")
    verify.add_argument("path", type=Path)
    field = subparsers.add_parser("field")
    field.add_argument("path", type=Path)
    field.add_argument("name", choices=("pid", "port", "profile", "source_sha", "log_path"))
    signal_parser = subparsers.add_parser("signal")
    signal_parser.add_argument("path", type=Path)
    signal_parser.add_argument("name", choices=("TERM", "KILL"))
    launch = subparsers.add_parser("launch")
    launch.add_argument("path", type=Path)
    launch.add_argument("profile")
    launch.add_argument("port", type=int)
    launch.add_argument("source_sha")
    launch.add_argument("log_path")
    launch.add_argument("expected_executable")
    launch.add_argument("expected_script")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    hold = subparsers.add_parser("hold")
    hold.add_argument("path", type=Path)
    hold.add_argument("profile")
    hold.add_argument("port", type=int)
    hold.add_argument("source_sha")
    args = parser.parse_args()
    try:
        if args.command == "create":
            create_record(
                args.path,
                pid=args.pid,
                profile=args.profile,
                port=args.port,
                source_sha=args.source_sha,
                log_path=args.log_path,
            )
        elif args.command == "verify":
            print(verify_record(args.path).pid)
        elif args.command == "field":
            print(getattr(read_record(args.path), args.name))
        elif args.command == "signal":
            number = signal.SIGTERM if args.name == "TERM" else signal.SIGKILL
            print(signal_record(args.path, number))
        elif args.command == "hold":
            hold_record(
                args.path,
                profile=args.profile,
                port=args.port,
                source_sha=args.source_sha,
            )
        else:
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            print(
                launch_recorded_process(
                    args.path,
                    command=command,
                    expected_command_prefix=(args.expected_executable, args.expected_script),
                    profile=args.profile,
                    port=args.port,
                    source_sha=args.source_sha,
                    log_path=args.log_path,
                )
            )
    except StaleProcessError as error:
        parser.exit(3, f"{error}\n")
    except RecordError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
