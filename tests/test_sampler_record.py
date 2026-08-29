import hashlib
import json
import os
from pathlib import Path
import signal

import pytest

from tools.remote_aloha.sampler_record import SamplerRecordError
from tools.remote_aloha.sampler_record import create_sampler_record
from tools.remote_aloha.sampler_record import stop_sampler
from tools.remote_aloha.sampler_record import verify_sampler_record


def _path(tmp_path: Path) -> Path:
    return tmp_path / ".runtime" / "gpu-sampler.json"


def test_record_is_private_atomic_and_stop_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _path(tmp_path)
    alive = True
    identity = ("Thu Aug 27 10:23:51 2026", hashlib.sha256(b"ssh robot-gpu").hexdigest())
    monkeypatch.setattr("tools.remote_aloha.sampler_record._identity_or_none", lambda _: identity if alive else None)

    def terminate(_pid: int, number: int) -> None:
        nonlocal alive
        assert number == signal.SIGTERM
        alive = False

    monkeypatch.setattr(os, "kill", terminate)
    record = create_sampler_record(4242, path=path)
    assert verify_sampler_record(path=path) == record
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.with_name("gpu-sampler.lock").stat().st_mode & 0o777 == 0o600
    assert stop_sampler(timeout_seconds=2, path=path)
    assert not path.exists()
    assert not stop_sampler(path=path)


def test_stale_record_is_cleared_without_signaling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _path(tmp_path)
    identities = iter([("start", "a" * 64), None])
    monkeypatch.setattr("tools.remote_aloha.sampler_record._identity_or_none", lambda _: next(identities))
    create_sampler_record(4242, path=path)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, number: signals.append((pid, number)))
    assert not stop_sampler(path=path)
    assert not path.exists()
    assert signals == []


def test_mismatched_live_process_is_rejected_and_retained(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _path(tmp_path)
    identities = iter([("start", "a" * 64), ("start", "b" * 64)])
    monkeypatch.setattr("tools.remote_aloha.sampler_record._identity_or_none", lambda _: next(identities))
    create_sampler_record(4242, path=path)
    with pytest.raises(SamplerRecordError, match="does not match"):
        stop_sampler(path=path)
    assert path.exists()


def test_term_then_kill_only_while_identity_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = _path(tmp_path)
    path.parent.mkdir(mode=0o700)
    path.write_text(
        json.dumps({"schema": 1, "pid": 4242, "process_start": "start", "command_sha256": "a" * 64}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    identities = iter(
        [
            ("start", "a" * 64),
            ("start", "a" * 64),
            ("start", "a" * 64),
            ("start", "a" * 64),
            None,
        ]
    )
    monkeypatch.setattr("tools.remote_aloha.sampler_record._identity_or_none", lambda _: next(identities))
    sent: list[int] = []
    monkeypatch.setattr(os, "kill", lambda _pid, number: sent.append(number))
    assert stop_sampler(timeout_seconds=0, path=path)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert not path.exists()


def test_identity_change_after_term_prevents_kill_and_retains_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _path(tmp_path)
    path.parent.mkdir(mode=0o700)
    path.write_text(
        json.dumps({"schema": 1, "pid": 4242, "process_start": "start", "command_sha256": "a" * 64}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    identities = iter([("start", "a" * 64), ("start", "a" * 64), ("other", "b" * 64)])
    monkeypatch.setattr("tools.remote_aloha.sampler_record._identity_or_none", lambda _: next(identities))
    sent: list[int] = []
    monkeypatch.setattr(os, "kill", lambda _pid, number: sent.append(number))
    with pytest.raises(SamplerRecordError, match="does not match"):
        stop_sampler(timeout_seconds=0, path=path)
    assert sent == [signal.SIGTERM]
    assert path.exists()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1, 301])
def test_stop_rejects_unbounded_timeout(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        stop_sampler(timeout_seconds=timeout, path=_path(tmp_path))
