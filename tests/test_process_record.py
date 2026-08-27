import json
import os
from pathlib import Path
import signal
import sys

import pytest

from tools.remote_aloha.process_record import RecordError
from tools.remote_aloha.process_record import StaleProcessError
from tools.remote_aloha.process_record import create_record
from tools.remote_aloha.process_record import launch_recorded_process
from tools.remote_aloha.process_record import read_record
from tools.remote_aloha.process_record import signal_record
from tools.remote_aloha.process_record import verify_record

PID = 4242
SHA = "a" * 40


def _write_process(proc_root: Path, *, start_ticks: int = 12345, command: bytes = b"python\0serve_policy.py\0") -> None:
    process = proc_root / str(PID)
    process.mkdir(parents=True, exist_ok=True)
    fields = ["S", *("0" for _ in range(18)), str(start_ticks), *("0" for _ in range(8))]
    (process / "stat").write_text(f"{PID} (worker ) with spaces) {' '.join(fields)}\n", encoding="utf-8")
    (process / "cmdline").write_bytes(command)


def _record(tmp_path: Path) -> tuple[Path, Path]:
    proc_root = tmp_path / "proc"
    _write_process(proc_root)
    path = tmp_path / ".runtime" / "server.json"
    create_record(
        path,
        pid=PID,
        profile="pi0_aloha_sim",
        port=8000,
        source_sha=SHA,
        log_path=".runtime/server.log",
        proc_root=proc_root,
    )
    return path, proc_root


def test_process_record_is_atomic_private_and_round_trips(tmp_path: Path) -> None:
    path, proc_root = _record(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert verify_record(path, proc_root).pid == PID
    with pytest.raises(RecordError, match="already exists"):
        create_record(
            path,
            pid=PID,
            profile="pi0_aloha_sim",
            port=8000,
            source_sha=SHA,
            log_path=".runtime/server.log",
            proc_root=proc_root,
        )


def test_identity_mismatch_and_stale_process_are_rejected(tmp_path: Path) -> None:
    path, proc_root = _record(tmp_path)
    _write_process(proc_root, start_ticks=99999)
    with pytest.raises(RecordError, match="does not match"):
        verify_record(path, proc_root)
    _write_process(proc_root, command=b"unrelated\0process\0")
    with pytest.raises(RecordError, match="does not match"):
        verify_record(path, proc_root)
    for child in (proc_root / str(PID)).iterdir():
        child.unlink()
    (proc_root / str(PID)).rmdir()
    with pytest.raises(StaleProcessError, match="no longer exists"):
        verify_record(path, proc_root)


def test_signal_gate_calls_kill_only_after_full_identity_match(tmp_path: Path) -> None:
    path, proc_root = _record(tmp_path)
    calls = []
    opened = []
    closed = []
    assert (
        signal_record(
            path,
            signal.SIGTERM,
            proc_root=proc_root,
            pidfd_open=lambda pid: opened.append(pid) or 99,
            pidfd_send_signal=lambda *args: calls.append(args),
            close=closed.append,
        )
        == PID
    )
    assert opened == [PID]
    assert calls == [(99, signal.SIGTERM)]
    assert closed == [99]
    _write_process(proc_root, start_ticks=99999)
    with pytest.raises(RecordError, match="does not match"):
        signal_record(
            path,
            signal.SIGKILL,
            proc_root=proc_root,
            pidfd_open=lambda pid: 100,
            pidfd_send_signal=lambda *args: calls.append(args),
            close=closed.append,
        )
    assert calls == [(99, signal.SIGTERM)]
    assert closed == [99, 100]
    with pytest.raises(RecordError, match="unsupported"):
        signal_record(path, signal.SIGHUP, proc_root=proc_root)


@pytest.mark.parametrize(
    "mutation",
    [
        {"pid": 1},
        {"start_ticks": 0},
        {"command_sha256": "bad"},
        {"profile": "x;id"},
        {"port": 0},
        {"source_sha": "bad"},
        {"log_path": "/tmp/server.log"},
        {"schema": 2},
    ],
)
def test_malformed_process_records_are_rejected(tmp_path: Path, mutation: dict[str, object]) -> None:
    path, _ = _record(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(mutation)
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(RecordError, match="invalid"):
        read_record(path)


def test_symlink_and_permissive_record_are_rejected(tmp_path: Path) -> None:
    path, _ = _record(tmp_path)
    path.chmod(0o644)
    with pytest.raises(RecordError, match="0600"):
        read_record(path)
    target = tmp_path / "target"
    target.write_text("{}", encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(RecordError, match="regular file"):
        read_record(path)


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="requires Linux pidfd")
def test_pidfd_gated_launch_creates_an_owned_record(tmp_path: Path) -> None:
    path = tmp_path / ".runtime" / "server.json"
    script = tmp_path / "policy-test.py"
    script.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    wrapper = "import os,sys,time; time.sleep(.3); os.execv(sys.argv[1], sys.argv[1:])"
    pid = launch_recorded_process(
        path,
        command=[sys.executable, "-c", wrapper, sys.executable, str(script)],
        expected_command_prefix=(sys.executable, str(script)),
        profile="pi0_aloha_sim",
        port=8000,
        source_sha=SHA,
        log_path=".runtime/server.log",
    )
    try:
        assert verify_record(path).pid == pid
        signal_record(path, signal.SIGTERM)
        os.waitpid(pid, 0)
    finally:
        try:
            signal_record(path, signal.SIGKILL)
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError, StaleProcessError):
            pass
