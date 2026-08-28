from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import urllib.request

from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.policy_smoke import run_policy_smoke
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.remote import _candidate_sha

_RECORD = Path(".runtime/tunnel.json")
_CONTROL_SOCKET = Path(".runtime/tunnel.sock")
_LOCK = Path(".runtime/tunnel.lock")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class TunnelRecord:
    schema: int
    pid: int
    process_start: str
    command_sha256: str
    ssh_alias: str
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int
    source_sha: str
    control_socket: str


def _run(arguments: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(arguments, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RemoteError("the local SSH tunnel command could not complete") from error


def _runtime() -> None:
    path = _RECORD.parent
    if path.is_symlink():
        raise RemoteError(".runtime must be a real private directory")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
        raise RemoteError(".runtime must be a mode-700 directory")


@contextmanager
def _lifecycle_lock():
    _runtime()
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(_LOCK, flags, 0o600)
    except OSError as error:
        raise RemoteError("the tunnel lifecycle lock cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise RemoteError("the tunnel lifecycle lock must be a mode-600 regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RemoteError("another tunnel lifecycle operation is active") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_alias(alias: str, timeout: int) -> None:
    result = _run(["ssh", "-G", alias], timeout=timeout)
    if result.returncode:
        raise RemoteError("the private robot-gpu SSH alias is not configured")
    values: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(" ")
        values.setdefault(key, []).append(value)
    if not values.get("hostname") or values["hostname"] == [alias]:
        raise RemoteError("the private robot-gpu SSH alias is not configured")
    if any(values.get(key) for key in ("localforward", "remoteforward", "dynamicforward")):
        raise RemoteError(
            "remove SSH-config forwarding from robot-gpu; the project permits only its exact local forward"
        )


def build_tunnel_argv(config: RemoteConfig, control_socket: Path = _CONTROL_SOCKET) -> list[str]:
    forward = f"{config.local_policy_host}:{config.local_policy_port}:" f"{config.policy_host}:{config.policy_port}"
    return [
        "ssh",
        "-T",
        "-N",
        "-f",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        f"ConnectTimeout={config.ssh_connect_timeout_seconds}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=1",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ControlMaster=yes",
        "-o",
        "ControlPersist=no",
        "-S",
        str(control_socket.resolve()),
        "-L",
        forward,
        config.ssh_alias,
    ]


def _control(alias: str, operation: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return _run(
        ["ssh", "-S", str(_CONTROL_SOCKET.resolve()), "-O", operation, alias],
        timeout=timeout,
    )


def _control_pid(alias: str, timeout: int) -> int:
    result = _control(alias, "check", timeout)
    match = re.search(r"Master running \(pid=(\d+)\)", result.stdout + result.stderr)
    if result.returncode or match is None or int(match.group(1)) <= 1:
        raise RemoteError("the recorded SSH control master is not running")
    return int(match.group(1))


def _process_identity_or_none(pid: int, timeout: int) -> tuple[str, str, str] | None:
    start = _run(["ps", "-p", str(pid), "-o", "lstart="], timeout=timeout)
    command = _run(["ps", "-ww", "-p", str(pid), "-o", "command="], timeout=timeout)
    process_start = start.stdout.strip()
    process_command = command.stdout.strip()
    if start.returncode and command.returncode and not process_start and not process_command:
        return None
    if start.returncode or command.returncode or not process_start or not process_command:
        raise RemoteError("the SSH control-master identity cannot be verified")
    return process_start, process_command, hashlib.sha256(process_command.encode()).hexdigest()


def _process_identity(pid: int, timeout: int) -> tuple[str, str, str]:
    identity = _process_identity_or_none(pid, timeout)
    if identity is None:
        raise RemoteError("the SSH control master is no longer running")
    return identity


def _validate_socket() -> None:
    try:
        metadata = _CONTROL_SOCKET.lstat()
    except FileNotFoundError as error:
        raise RemoteError("the SSH control socket is missing") from error
    if not stat.S_ISSOCK(metadata.st_mode) or metadata.st_mode & 0o077:
        raise RemoteError("the SSH control socket must be a private Unix socket")


def _validate_listener(pid: int, config: RemoteConfig) -> None:
    lsof = shutil.which("lsof")
    if lsof is None:
        raise RemoteError("lsof is required to validate the Mac tunnel listener")
    result = _run(
        [
            lsof,
            "-nP",
            "-a",
            "-p",
            str(pid),
            f"-iTCP:{config.local_policy_port}",
            "-sTCP:LISTEN",
            "-Fpn",
        ],
        timeout=config.ssh_connect_timeout_seconds,
    )
    names = [line[1:] for line in result.stdout.splitlines() if line.startswith("n")]
    expected = f"{config.local_policy_host}:{config.local_policy_port}"
    if result.returncode or not names or set(names) != {expected}:
        raise RemoteError("the Mac tunnel listener is not owned and bound only to IPv4 loopback")


def _health(config: RemoteConfig) -> None:
    try:
        with urllib.request.urlopen(
            f"http://{config.local_policy_host}:{config.local_policy_port}/healthz",
            timeout=config.policy_connect_timeout_seconds,
        ) as response:
            if response.status != 200 or response.read() != b"OK\n":
                raise RemoteError("the tunneled policy health response is invalid")
    except (OSError, TimeoutError) as error:
        raise RemoteError("the Mac cannot reach policy health through the SSH tunnel") from error


def _validate_record(record: TunnelRecord) -> None:
    if (
        record.schema != 1
        or record.pid <= 1
        or not record.process_start
        or not re.fullmatch(r"[0-9a-f]{64}", record.command_sha256)
        or not _ALIAS.fullmatch(record.ssh_alias)
        or record.local_host != "127.0.0.1"
        or record.remote_host != "127.0.0.1"
        or not 1 <= record.local_port <= 65535
        or not 1 <= record.remote_port <= 65535
        or not _SHA.fullmatch(record.source_sha)
        or record.control_socket != str(_CONTROL_SOCKET)
    ):
        raise RemoteError("the tunnel ownership record is invalid")


def _read_record() -> TunnelRecord:
    try:
        metadata = _RECORD.lstat()
        payload = json.loads(_RECORD.read_text(encoding="utf-8"))
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError
        if not isinstance(payload, dict) or set(payload) != set(TunnelRecord.__dataclass_fields__):
            raise ValueError
        record = TunnelRecord(**payload)
        _validate_record(record)
        return record
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RemoteError("the tunnel ownership record is unreadable or invalid") from error


def _write_record(record: TunnelRecord) -> None:
    _validate_record(record)
    if _RECORD.exists() or _RECORD.is_symlink():
        raise RemoteError("a tunnel ownership record already exists")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=_RECORD.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(asdict(record), stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, _RECORD)
    except OSError as error:
        raise RemoteError("the tunnel ownership record cannot be written safely") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _record_config(record: TunnelRecord) -> RemoteConfig:
    return RemoteConfig(
        ssh_alias=record.ssh_alias,
        local_policy_host=record.local_host,
        local_policy_port=record.local_port,
        policy_host=record.remote_host,
        policy_port=record.remote_port,
    )


def _verify_identity(record: TunnelRecord) -> int:
    _validate_socket()
    pid = _control_pid(record.ssh_alias, 10)
    process_start, process_command, command_hash = _process_identity(pid, 10)
    if (
        pid != record.pid
        or process_start != record.process_start
        or command_hash != record.command_sha256
        or str(_CONTROL_SOCKET.resolve()) not in process_command
        or " -g " in f" {process_command} "
    ):
        raise RemoteError("the SSH control-master identity does not match its ownership record")
    return pid


def _verify(config: RemoteConfig | None = None) -> TunnelRecord:
    _runtime()
    record = _read_record()
    pid = _verify_identity(record)
    actual = config or _record_config(record)
    if config is not None and (
        config.ssh_alias,
        config.local_policy_host,
        config.local_policy_port,
        config.policy_host,
        config.policy_port,
    ) != (
        record.ssh_alias,
        record.local_host,
        record.local_port,
        record.remote_host,
        record.remote_port,
    ):
        raise RemoteError("the running tunnel does not match the requested configuration")
    _validate_listener(pid, actual)
    return record


def _remove_control_socket() -> None:
    if not _CONTROL_SOCKET.exists() and not _CONTROL_SOCKET.is_symlink():
        return
    metadata = _CONTROL_SOCKET.lstat()
    if not stat.S_ISSOCK(metadata.st_mode):
        raise RemoteError("refusing to remove a non-socket tunnel control path")
    _CONTROL_SOCKET.unlink()


def _local_port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _prove_stopped(record: TunnelRecord) -> bool:
    identity = _process_identity_or_none(record.pid, 2)
    if identity is not None and (identity[0], identity[2]) == (record.process_start, record.command_sha256):
        return False
    return _local_port_is_free(record.local_host, record.local_port)


def _wait_for_stopped(record: TunnelRecord, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _control(record.ssh_alias, "check", 2).returncode != 0 and _prove_stopped(record):
            return
        time.sleep(0.1)
    raise RemoteError("the SSH control master or its listener did not stop; ownership state was retained")


def _shutdown_verified(record: TunnelRecord, timeout: int) -> None:
    _verify_identity(record)
    result = _control(record.ssh_alias, "exit", timeout)
    if result.returncode:
        raise RemoteError("the verified SSH control master did not accept shutdown")
    _wait_for_stopped(record, timeout)
    _remove_control_socket()
    if _RECORD.exists() and not _RECORD.is_symlink():
        _RECORD.unlink()


def _capture_record(config: RemoteConfig, candidate: str) -> TunnelRecord:
    _validate_socket()
    pid = _control_pid(config.ssh_alias, config.ssh_connect_timeout_seconds)
    process_start, _, command_hash = _process_identity(pid, config.ssh_connect_timeout_seconds)
    record = TunnelRecord(
        schema=1,
        pid=pid,
        process_start=process_start,
        command_sha256=command_hash,
        ssh_alias=config.ssh_alias,
        local_host=config.local_policy_host,
        local_port=config.local_policy_port,
        remote_host=config.policy_host,
        remote_port=config.policy_port,
        source_sha=candidate,
        control_socket=str(_CONTROL_SOCKET),
    )
    _write_record(record)
    return record


def _record_for_cleanup(config: RemoteConfig, candidate: str) -> TunnelRecord | None:
    if _RECORD.exists() or _RECORD.is_symlink():
        record = _read_record()
        if (
            record.ssh_alias,
            record.local_host,
            record.local_port,
            record.remote_host,
            record.remote_port,
            record.source_sha,
        ) != (
            config.ssh_alias,
            config.local_policy_host,
            config.local_policy_port,
            config.policy_host,
            config.policy_port,
            candidate,
        ):
            raise RemoteError("the interrupted tunnel ownership record does not match this launch")
        return record
    if _CONTROL_SOCKET.exists() or _CONTROL_SOCKET.is_symlink():
        return _capture_record(config, candidate)
    return None


def _start_locked(config: RemoteConfig) -> None:
    if _RECORD.exists() or _RECORD.is_symlink() or _CONTROL_SOCKET.exists() or _CONTROL_SOCKET.is_symlink():
        raise RemoteError("a tunnel record or control socket already exists; run make stop")
    candidate = _candidate_sha()
    _validate_alias(config.ssh_alias, config.ssh_connect_timeout_seconds)
    if not _local_port_is_free(config.local_policy_host, config.local_policy_port):
        raise RemoteError("the configured Mac loopback port is already occupied")
    arguments = build_tunnel_argv(config)
    record = None
    try:
        result = _run(arguments, timeout=config.ssh_connect_timeout_seconds + 10)
        if result.returncode:
            raise RemoteError("the SSH local forward could not start")
        record = _capture_record(config, candidate)
        _validate_listener(record.pid, config)
        _health(config)
    except BaseException:
        try:
            record = record or _record_for_cleanup(config, candidate)
        except BaseException as ownership_error:
            raise RemoteError(
                "tunnel startup failed before ownership could be verified; control state was retained"
            ) from ownership_error
        if record is None:
            raise
        try:
            _shutdown_verified(record, config.ssh_connect_timeout_seconds)
        except BaseException as cleanup_error:
            raise RemoteError(
                "tunnel startup failed and verified cleanup could not finish; ownership state was retained"
            ) from cleanup_error
        raise
    print("Mac loopback SSH tunnel is ready.")


def start(config: RemoteConfig) -> None:
    with _lifecycle_lock():
        _start_locked(config)


def check(config: RemoteConfig) -> None:
    _verify(config)
    _health(config)
    print("Mac loopback SSH tunnel is healthy.")


def _stop_locked(config: RemoteConfig) -> None:
    record_exists = _RECORD.exists() or _RECORD.is_symlink()
    socket_exists = _CONTROL_SOCKET.exists() or _CONTROL_SOCKET.is_symlink()
    if not record_exists and not socket_exists:
        if not _local_port_is_free(config.local_policy_host, config.local_policy_port):
            raise RemoteError("the Mac policy port is occupied without an owned tunnel record")
        print("Mac SSH tunnel is already stopped.")
        return
    if record_exists and not socket_exists:
        record = _read_record()
        if not _prove_stopped(record):
            raise RemoteError("the tunnel control socket is missing but its process or listener may still be active")
        _RECORD.unlink()
        print("Removed a stale tunnel record; no process was signaled.")
        return
    if not record_exists:
        raise RemoteError("tunnel state is incomplete; refusing an unverified stop")
    record = _read_record()
    _validate_socket()
    if _control(record.ssh_alias, "check", 2).returncode != 0:
        if not _prove_stopped(record):
            raise RemoteError(
                "the tunnel control socket is unresponsive but its process or listener may still be active"
            )
        _remove_control_socket()
        _RECORD.unlink()
        print("Removed stale tunnel state; no process was signaled.")
        return
    record = _verify()
    _shutdown_verified(record, 10)
    print("Owned Mac SSH tunnel stopped.")


def stop(config: RemoteConfig | None = None) -> None:
    with _lifecycle_lock():
        _stop_locked(config or load_remote_config())


def smoke(config: RemoteConfig) -> None:
    record = _verify(config)
    candidate = _candidate_sha()
    if record.source_sha != candidate:
        raise RemoteError("the running tunnel source SHA differs from the current candidate")
    _health(config)
    try:
        summary = run_policy_smoke(
            profile_name=config.policy_profile.name,
            backend=config.policy_backend,
            host=config.local_policy_host,
            port=config.local_policy_port,
            source_sha=candidate,
            connect_timeout=config.policy_connect_timeout_seconds,
            metadata_timeout=config.policy_metadata_timeout_seconds,
            inference_timeout=config.policy_inference_timeout_seconds,
            close_timeout=config.policy_close_timeout_seconds,
        )
    except Exception as error:
        raise RemoteError("tunneled policy smoke failed; inspect private runtime logs") from error
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
    evidence = Path("outputs") / "phase03" / timestamp
    evidence.mkdir(mode=0o700, parents=True)
    path = evidence / f"policy-smoke-{config.policy_profile.name}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True)
        stream.write("\n")
    print(json.dumps({**summary, "evidence": str(path)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the Phase 3 Mac SSH tunnel and tunneled smoke test.")
    parser.add_argument("command", choices=("start", "check", "stop", "smoke"))
    args = parser.parse_args()
    config = load_remote_config()
    try:
        if args.command == "start":
            start(config)
        elif args.command == "check":
            check(config)
        elif args.command == "stop":
            stop(config)
        else:
            smoke(config)
    except RemoteError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
