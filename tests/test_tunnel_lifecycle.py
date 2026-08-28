from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.remote_aloha import connection_check
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.remote import RemoteError


def test_tunnel_argv_is_one_explicit_loopback_forward() -> None:
    config = RemoteConfig(local_policy_port=8123, policy_port=8456)
    argv = connection_check.build_tunnel_argv(config, Path(".runtime/test.sock"))
    assert argv[:4] == ["ssh", "-T", "-N", "-f"]
    assert argv[-3:] == ["-L", "127.0.0.1:8123:127.0.0.1:8456", "robot-gpu"]
    assert "ExitOnForwardFailure=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert "ForwardAgent=no" in argv
    assert "ForwardX11=no" in argv
    assert "PermitLocalCommand=no" in argv
    assert "ClearAllForwardings=yes" not in argv
    assert "-g" not in argv


def test_alias_validation_is_quiet_and_rejects_configured_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        connection_check,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "hostname private-target\n", ""),
    )
    connection_check._validate_alias("robot-gpu", 10)  # noqa: SLF001

    monkeypatch.setattr(
        connection_check,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "hostname private-target\nlocalforward 9000 127.0.0.1:9000\n", ""
        ),
    )
    with pytest.raises(RemoteError, match="permits only its exact"):
        connection_check._validate_alias("robot-gpu", 10)  # noqa: SLF001

    monkeypatch.setattr(
        connection_check,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "hostname robot-gpu\n", ""),
    )
    with pytest.raises(RemoteError, match="not configured"):
        connection_check._validate_alias("robot-gpu", 10)  # noqa: SLF001


def _record(tmp_path: Path) -> connection_check.TunnelRecord:
    control = tmp_path / ".runtime" / "tunnel.sock"
    command = f"ssh: {control.resolve()} [mux]"
    return connection_check.TunnelRecord(
        schema=1,
        pid=4242,
        process_start="Thu Aug 27 12:00:00 2026",
        command_sha256=hashlib.sha256(command.encode()).hexdigest(),
        ssh_alias="robot-gpu",
        local_host="127.0.0.1",
        local_port=8000,
        remote_host="127.0.0.1",
        remote_port=8000,
        source_sha="a" * 40,
        control_socket=".runtime/tunnel.sock",
    )


def test_private_record_round_trip_and_malformed_state_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    record = _record(tmp_path)
    connection_check._write_record(record)  # noqa: SLF001
    assert connection_check._read_record() == record  # noqa: SLF001
    assert Path(".runtime/tunnel.json").stat().st_mode & 0o777 == 0o600
    Path(".runtime/tunnel.json").chmod(0o644)
    with pytest.raises(RemoteError, match="unreadable or invalid"):
        connection_check._read_record()  # noqa: SLF001

    Path(".runtime/tunnel.json").unlink()
    target = tmp_path / "target"
    target.write_text(json.dumps(asdict(record)), encoding="utf-8")
    Path(".runtime/tunnel.json").symlink_to(target)
    with pytest.raises(RemoteError, match="unreadable or invalid"):
        connection_check._read_record()  # noqa: SLF001


def test_start_rejects_local_port_collision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(connection_check, "_candidate_sha", lambda: "a" * 40)
    monkeypatch.setattr(connection_check, "_validate_alias", lambda *args: None)

    class OccupiedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            raise OSError("occupied")

    monkeypatch.setattr(connection_check.socket, "socket", lambda *args: OccupiedSocket())
    with pytest.raises(RemoteError, match="already occupied"):
        connection_check.start(RemoteConfig(local_policy_port=8123))


def test_lifecycle_lock_rejects_concurrent_operation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    def flock(_descriptor: int, operation: int) -> None:
        if operation & connection_check.fcntl.LOCK_EX:
            raise BlockingIOError

    monkeypatch.setattr(connection_check.fcntl, "flock", flock)
    with (
        pytest.raises(RemoteError, match="another tunnel lifecycle"),
        connection_check._lifecycle_lock(),  # noqa: SLF001
    ):
        raise AssertionError("the contended lock must not be entered")


def test_stale_record_cleanup_and_stop_twice_signal_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    connection_check._write_record(_record(tmp_path))  # noqa: SLF001
    calls = []
    monkeypatch.setattr(
        connection_check, "_control", lambda *args: calls.append(args) or subprocess.CompletedProcess([], 1)
    )
    monkeypatch.setattr(connection_check, "_prove_stopped", lambda *args: True)
    monkeypatch.setattr(connection_check, "_local_port_is_free", lambda *args: True)
    config = RemoteConfig()
    connection_check.stop(config)
    connection_check.stop(config)
    assert calls == []
    assert not Path(".runtime/tunnel.json").exists()


def test_stale_record_is_retained_when_stopped_state_is_uncertain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    connection_check._write_record(_record(tmp_path))  # noqa: SLF001
    monkeypatch.setattr(connection_check, "_prove_stopped", lambda *args: False)
    with pytest.raises(RemoteError, match="may still be active"):
        connection_check.stop(RemoteConfig())
    assert Path(".runtime/tunnel.json").exists()


def test_stop_rejects_unowned_port_or_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(connection_check, "_local_port_is_free", lambda *args: False)
    with pytest.raises(RemoteError, match="occupied without an owned"):
        connection_check.stop(RemoteConfig())

    monkeypatch.setattr(connection_check, "_local_port_is_free", lambda *args: True)
    control = Path(".runtime/tunnel.sock")
    control.write_text("not owned", encoding="utf-8")
    with pytest.raises(RemoteError, match="incomplete"):
        connection_check.stop(RemoteConfig())
    assert control.exists()


def test_unresponsive_owned_state_is_retained_when_stop_cannot_be_proved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    record = _record(tmp_path)
    connection_check._write_record(record)  # noqa: SLF001
    Path(".runtime/tunnel.sock").write_text("test stand-in", encoding="utf-8")
    monkeypatch.setattr(connection_check, "_validate_socket", lambda: None)
    monkeypatch.setattr(connection_check, "_control", lambda *args: subprocess.CompletedProcess([], 1, "", ""))
    monkeypatch.setattr(connection_check, "_prove_stopped", lambda *args: False)
    with pytest.raises(RemoteError, match="may still be active"):
        connection_check.stop(RemoteConfig())
    assert Path(".runtime/tunnel.json").exists()
    assert Path(".runtime/tunnel.sock").exists()


def test_active_stop_verifies_identity_before_shutdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    record = _record(tmp_path)
    connection_check._write_record(record)  # noqa: SLF001
    Path(".runtime/tunnel.sock").write_text("test stand-in", encoding="utf-8")
    monkeypatch.setattr(connection_check, "_validate_socket", lambda: None)
    monkeypatch.setattr(connection_check, "_control", lambda *args: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(connection_check, "_verify", lambda: record)
    stopped = []
    monkeypatch.setattr(connection_check, "_shutdown_verified", lambda *args: stopped.append(args))
    connection_check.stop(RemoteConfig())
    assert stopped == [(record, 10)]


def test_failed_start_cleans_only_the_recorded_tunnel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(connection_check, "_candidate_sha", lambda: "a" * 40)
    monkeypatch.setattr(connection_check, "_validate_alias", lambda *args: None)
    monkeypatch.setattr(connection_check, "_local_port_is_free", lambda *args: True)
    monkeypatch.setattr(connection_check, "_run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(connection_check, "_validate_socket", lambda: None)
    monkeypatch.setattr(connection_check, "_control_pid", lambda *args: 4242)
    command = f"ssh: {(tmp_path / '.runtime' / 'tunnel.sock').resolve()} [mux]"
    monkeypatch.setattr(
        connection_check,
        "_process_identity",
        lambda *args: ("Thu Aug 27 12:00:00 2026", command, hashlib.sha256(command.encode()).hexdigest()),
    )
    monkeypatch.setattr(connection_check, "_validate_listener", lambda *args: None)
    monkeypatch.setattr(connection_check, "_health", lambda *args: (_ for _ in ()).throw(RemoteError("health failed")))
    cleaned = []

    def cleanup(record: connection_check.TunnelRecord, timeout: int) -> None:
        cleaned.append((record, timeout))
        Path(".runtime/tunnel.json").unlink()

    monkeypatch.setattr(connection_check, "_shutdown_verified", cleanup)
    with pytest.raises(RemoteError, match="health failed"):
        connection_check.start(RemoteConfig())
    assert len(cleaned) == 1
    assert not Path(".runtime/tunnel.json").exists()


@pytest.mark.parametrize("outcome", ["nonzero", "interrupt"])
def test_launch_failure_captures_and_cleans_a_live_control_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(connection_check, "_candidate_sha", lambda: "a" * 40)
    monkeypatch.setattr(connection_check, "_validate_alias", lambda *args: None)
    monkeypatch.setattr(connection_check, "_local_port_is_free", lambda *args: True)

    def launch(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(".runtime/tunnel.sock").write_text("test stand-in", encoding="utf-8")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess([], 1, "", "")

    monkeypatch.setattr(connection_check, "_run", launch)
    monkeypatch.setattr(connection_check, "_validate_socket", lambda: None)
    monkeypatch.setattr(connection_check, "_control_pid", lambda *args: 4242)
    command = f"ssh: {(tmp_path / '.runtime' / 'tunnel.sock').resolve()} [mux]"
    monkeypatch.setattr(
        connection_check,
        "_process_identity",
        lambda *args: ("Thu Aug 27 12:00:00 2026", command, hashlib.sha256(command.encode()).hexdigest()),
    )
    cleaned = []

    def cleanup(record: connection_check.TunnelRecord, timeout: int) -> None:
        cleaned.append((record, timeout))
        Path(".runtime/tunnel.json").unlink()
        Path(".runtime/tunnel.sock").unlink()

    monkeypatch.setattr(connection_check, "_shutdown_verified", cleanup)
    expected_error = KeyboardInterrupt if outcome == "interrupt" else RemoteError
    with pytest.raises(expected_error):
        connection_check.start(RemoteConfig())
    assert len(cleaned) == 1
    assert not Path(".runtime/tunnel.json").exists()
    assert not Path(".runtime/tunnel.sock").exists()


def test_duplicate_start_is_rejected_before_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    connection_check._write_record(_record(tmp_path))  # noqa: SLF001
    monkeypatch.setattr(
        connection_check,
        "_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SSH must not be launched")),
    )
    with pytest.raises(RemoteError, match="already exists"):
        connection_check.start(RemoteConfig())


def test_verify_rejects_changed_process_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    connection_check._runtime()  # noqa: SLF001
    record = _record(tmp_path)
    connection_check._write_record(record)  # noqa: SLF001
    monkeypatch.setattr(connection_check, "_validate_socket", lambda: None)
    monkeypatch.setattr(connection_check, "_control_pid", lambda *args: record.pid)
    monkeypatch.setattr(
        connection_check,
        "_process_identity",
        lambda *args: (record.process_start, "ssh: changed [mux]", "b" * 64),
    )
    with pytest.raises(RemoteError, match="identity"):
        connection_check._verify()  # noqa: SLF001
