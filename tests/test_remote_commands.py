import base64
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools.remote_aloha import connection_check
from tools.remote_aloha import remote as remote_module
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.remote import GpuSampler
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.remote import RemoteSession
from tools.remote_aloha.remote import RemoteTarget
from tools.remote_aloha.remote import _candidate_sha
from tools.remote_aloha.remote import _doctor_script
from tools.remote_aloha.remote import _gpu_sampler_command
from tools.remote_aloha.remote import _gpu_sampler_copy_script
from tools.remote_aloha.remote import _gpu_sampler_ready_script
from tools.remote_aloha.remote import _gpu_sampler_stop_script
from tools.remote_aloha.remote import _read_launch_receipt
from tools.remote_aloha.remote import _remote_script_command
from tools.remote_aloha.remote import _setup_script
from tools.remote_aloha.remote import _write_launch_receipt
from tools.remote_aloha.remote import build_wsl_command
from tools.remote_aloha.remote import classify_route
from tools.remote_aloha.remote import route as check_route
from tools.remote_aloha.remote import select_ubuntu_distro
from tools.remote_aloha.remote import smoke
from tools.remote_aloha.remote import ssh_argv
from tools.remote_aloha.remote import stop
from tools.remote_aloha.remote import windows_listener_addresses_are_private


def _decode_powershell(command: str) -> str:
    return base64.b64decode(command.rsplit(" ", 1)[1]).decode("utf-16le")


def test_ssh_argv_is_bounded_and_fail_closed() -> None:
    argv = ssh_argv(RemoteConfig())
    assert argv[:2] == ["ssh", "-T"]
    assert argv[-1] == "robot-gpu"
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert "ConnectionAttempts=1" in argv
    assert "ConnectTimeout=10" in argv
    assert "ClearAllForwardings=yes" in argv
    with pytest.raises(ValueError, match="unsafe SSH alias"):
        ssh_argv(RemoteConfig(ssh_alias="user@host"))


def test_wsl_command_keeps_distro_and_payload_out_of_outer_windows_quoting() -> None:
    distro = "Ubuntu Dev's"
    command = build_wsl_command("powershell", distro)
    assert distro not in command
    decoded = _decode_powershell(command)
    assert distro not in decoded
    encoded_distro = decoded.split("FromBase64String('", 1)[1].split("')", 1)[0]
    assert base64.b64decode(encoded_distro).decode() == distro
    assert "[Console]::In.ReadToEnd()" in decoded
    assert "$payload | wsl.exe --distribution $distro --exec bash -c \"tr -d '\\r' | bash -s --\"" in decoded
    assert build_wsl_command("bash", "") == "bash -s --"
    assert build_wsl_command("bash", "", 19) == "timeout --signal=TERM --kill-after=30s 19s bash -s --"
    for route_name in ("powershell", "cmd"):
        assert (
            "--exec bash -c \"tr -d '\\r' | timeout --signal=TERM --kill-after=30s 19s bash -s --\""
            in _decode_powershell(build_wsl_command(route_name, "Ubuntu", 19))
        )
    with pytest.raises(ValueError, match="explicit WSL distro"):
        build_wsl_command("cmd", "")
    with pytest.raises(ValueError, match="unsupported"):
        build_wsl_command("fish", "Ubuntu")


def test_remote_script_values_are_base64_data_not_shell_code() -> None:
    value = "/srv/open pi's;$(id)"
    config = RemoteConfig(remote_dir=value)
    script = _remote_script_command(config, "example.sh", [value])
    assert value not in script
    assert "eval" not in script
    assert '\ncd "$repo"\nexec ' in script
    encoded = base64.b64encode(value.encode()).decode("ascii")
    assert script.count(encoded) == 2


def test_default_tilde_remote_path_resolves_inside_wsl(tmp_path: Path) -> None:
    executable = tmp_path / "src" / "openpi" / "scripts" / "example.sh"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/usr/bin/env bash\nprintf '%s' \"$1\"\n", encoding="utf-8")
    executable.chmod(0o755)
    script = _remote_script_command(RemoteConfig(), "example.sh", ["round trip"])
    result = subprocess.run(
        ["bash"],
        input=script,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "round trip"


def test_default_tilde_setup_path_resolves_inside_wsl(tmp_path: Path) -> None:
    prefix = _setup_script(RemoteConfig(), "a" * 40, "codex/06-hardening-docs").split("lifecycle_state=", 1)[0]
    result = subprocess.run(
        ["bash"],
        input=prefix + "printf '%s' \"$remote_dir\"\n",
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(tmp_path / "src" / "openpi")


def test_candidate_rejects_an_older_phase_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_module, "_git", lambda *args: "codex/05-observability" if args[0] == "branch" else "a" * 40
    )
    with pytest.raises(RemoteError, match="codex/06-hardening-docs"):
        _candidate_sha()


@pytest.mark.parametrize(
    "script",
    [
        _doctor_script(8000),
        _setup_script(RemoteConfig(), "a" * 40, "codex/06-hardening-docs"),
        _remote_script_command(RemoteConfig(), "example.sh", ["one", "two"]),
        _gpu_sampler_command(
            RemoteConfig(),
            ".runtime/gpu-metrics-" + "b" * 32 + "-pi0_aloha_sim.jsonl",
            "b" * 32,
            "4242",
            "a" * 40,
        ),
        _gpu_sampler_ready_script(RemoteConfig(), ".runtime/gpu-metrics-" + "b" * 32 + "-pi0_aloha_sim.jsonl"),
        _gpu_sampler_copy_script(
            RemoteConfig(),
            ".runtime/gpu-metrics-" + "b" * 32 + "-pi0_aloha_sim.jsonl",
            ".runtime/server-abc.log",
        ),
        _gpu_sampler_stop_script(RemoteConfig(), expected_profile="pi0_aloha_sim", expected_source_sha="a" * 40),
    ],
)
def test_generated_wsl_bash_is_valid(script: str) -> None:
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_doctor_checks_evdev_build_prerequisites() -> None:
    script = _doctor_script(8000)
    assert "base64 cc curl flock git realpath ss timeout" in script
    assert "GNU time" in script
    assert "linux-input-headers" in script


def test_setup_migrates_only_the_known_pre_rename_origin() -> None:
    script = _setup_script(RemoteConfig(), "a" * 40, "codex/06-hardening-docs")
    assert "legacy_repo_url" in script
    assert 'remote set-url origin "$repo_url"' in script
    assert [line for line in script.splitlines() if line.startswith("progress '")] == [
        "progress 'validating workspace and storage'",
        "progress 'syncing the exact source candidate'",
        "progress 'synchronizing pinned submodules'",
        "progress 'synchronizing the locked Python environment'",
        "progress 'validating the RTX 3090 runtime'",
        "progress 'setup complete'",
    ]


def test_setup_streams_the_exact_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = "a" * 40
    session = RemoteSession(RemoteConfig())
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    captured = {}
    monkeypatch.setattr(remote_module, "_read_launch_receipt", lambda: None)
    monkeypatch.setattr(remote_module, "_candidate", lambda: (candidate, "codex/push-pi-scenarios"))
    monkeypatch.setattr(remote_module, "doctor", lambda actual_session: target)

    def run_wsl(*args: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return f"__ALOHA_PROJECT_SHA__={candidate}\n__ALOHA_SETUP__=passed"

    monkeypatch.setattr(session, "run_wsl", run_wsl)
    remote_module.setup(session)
    assert captured == {
        "timeout": session.config.server_startup_timeout_seconds + 45,
        "label": "setup-pc",
        "command_timeout": session.config.server_startup_timeout_seconds,
        "stream": True,
    }


def test_doctor_accepts_selected_ubuntu_2404_and_requires_uv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    uv = "uv 0.8.13"

    def run_wsl(*args: object, **kwargs: object) -> str:
        return "\n".join(
            [
                "__ALOHA_OS_ID__=ubuntu",
                "__ALOHA_OS_VERSION__=24.04",
                "__ALOHA_WSL2__=yes",
                "__ALOHA_ARCH__=x86_64",
                "__ALOHA_GPU_NAME__=NVIDIA GeForce RTX 3090",
                "__ALOHA_RAM_TOTAL_KIB__=32866932",
                "__ALOHA_RAM_AVAILABLE_KIB__=31981904",
                "__ALOHA_TOOLS__=ready",
                f"__ALOHA_UV__={uv}",
            ]
        )

    monkeypatch.setattr(RemoteSession, "run_wsl", run_wsl)
    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04"))
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    assert remote_module.doctor(session, target) == target
    summary = json.loads(capsys.readouterr().out)
    assert summary["ram_total_kib"] == 32866932
    assert summary["ram_available_kib"] == 31981904
    assert summary["automatic_conversion_restore_mode"] == "full-float32"
    with pytest.raises(RemoteError, match="select it explicitly"):
        remote_module.doctor(RemoteSession(RemoteConfig()), target)
    uv = "missing"
    with pytest.raises(RemoteError, match="curl -LsSf https://astral.sh/uv/install.sh"):
        remote_module.doctor(session, target)


def test_doctor_rejects_invalid_wsl_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"total": "", "available": "12000000"}

    def run_wsl(*args: object, **kwargs: object) -> str:
        return "\n".join(
            [
                "__ALOHA_OS_ID__=ubuntu",
                "__ALOHA_OS_VERSION__=24.04",
                "__ALOHA_WSL2__=yes",
                "__ALOHA_ARCH__=x86_64",
                "__ALOHA_GPU_NAME__=NVIDIA GeForce RTX 3090",
                f"__ALOHA_RAM_TOTAL_KIB__={values['total']}",
                f"__ALOHA_RAM_AVAILABLE_KIB__={values['available']}",
                "__ALOHA_TOOLS__=ready",
                "__ALOHA_UV__=uv 0.12.6",
            ]
        )

    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04"))
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    monkeypatch.setattr(session, "run_wsl", run_wsl)
    with pytest.raises(RemoteError, match="could not be measured"):
        remote_module.doctor(session, target)
    values.update(total="12000000", available="12000001")
    with pytest.raises(RemoteError, match="could not be measured"):
        remote_module.doctor(session, target)


def test_shell_route_and_distro_selection_are_unambiguous() -> None:
    outputs = {"bash": (1, ""), "powershell": (0, "__ALOHA_ROUTE_POWERSHELL__"), "cmd": (1, "")}
    assert classify_route(outputs) == "powershell"
    with pytest.raises(RemoteError, match="uniquely"):
        classify_route({"bash": (0, "__ALOHA_ROUTE_BASH__"), "cmd": (0, "__ALOHA_ROUTE_CMD__")})
    assert select_ubuntu_distro({"Ubuntu-22.04": "ubuntu"}, "") == "Ubuntu-22.04"
    assert select_ubuntu_distro({"Ubuntu A": "ubuntu", "Debian": "debian"}, "Ubuntu A") == "Ubuntu A"
    with pytest.raises(RemoteError, match="exactly one"):
        select_ubuntu_distro({"Ubuntu A": "ubuntu", "Ubuntu B": "ubuntu"}, "")
    with pytest.raises(RemoteError, match="not a discovered"):
        select_ubuntu_distro({"Debian": "debian"}, "Ubuntu")


def test_windows_route_gate_checks_owned_wsl_server_and_rejects_wildcard_listener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    candidate = "a" * 40
    config = RemoteConfig(wsl_distro="Ubuntu-24.04", policy_backend="pytorch")
    receipt = {
        "profile": config.policy_profile.name,
        "backend": config.policy_backend,
        "port": config.policy_port,
        "remote_dir": config.remote_dir,
        "route": "cmd",
        "source_sha": candidate,
        "ssh_alias": config.ssh_alias,
        "wsl_distro": config.wsl_distro,
    }
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: candidate)
    monkeypatch.setattr(remote_module, "_read_launch_receipt", lambda: receipt)
    session = RemoteSession(config)
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: "__ALOHA_SERVER__=ready")
    captured = {}

    def ssh(command: str, **kwargs: object) -> tuple[int, str]:
        captured.update({"command": command, **kwargs})
        listeners = base64.b64encode(b'["127.0.0.1"]')
        return 0, f"__ALOHA_WINDOWS_WSL_ROUTE__=ready\n__ALOHA_WINDOWS_LISTENERS__={listeners.decode()}"

    monkeypatch.setattr(session, "ssh", ssh)
    check_route(session)
    script = _decode_powershell(str(captured["command"]))
    assert "Invoke-WebRequest" in script
    assert "127.0.0.1:8000/healthz" in script
    assert "Get-NetTCPConnection" in script
    assert "ConvertTo-Json" in script
    assert windows_listener_addresses_are_private([])
    assert windows_listener_addresses_are_private(["127.0.0.1", "::1"])
    assert not windows_listener_addresses_are_private(["192.0.2.1"])


def test_windows_route_gate_rejects_a_concrete_nonloopback_listener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    candidate = "a" * 40
    config = RemoteConfig(wsl_distro="Ubuntu-24.04", policy_backend="pytorch")
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: candidate)
    monkeypatch.setattr(
        remote_module,
        "_read_launch_receipt",
        lambda: {
            "profile": config.policy_profile.name,
            "backend": config.policy_backend,
            "port": config.policy_port,
            "remote_dir": config.remote_dir,
            "route": "cmd",
            "source_sha": candidate,
            "ssh_alias": config.ssh_alias,
            "wsl_distro": config.wsl_distro,
        },
    )
    session = RemoteSession(config)
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: "__ALOHA_SERVER__=ready")
    encoded = base64.b64encode(b'["192.0.2.1"]')
    monkeypatch.setattr(
        session,
        "ssh",
        lambda *args, **kwargs: (
            0,
            f"__ALOHA_WINDOWS_WSL_ROUTE__=ready\n__ALOHA_WINDOWS_LISTENERS__={encoded.decode()}",
        ),
    )
    with pytest.raises(RemoteError, match="cannot safely reach"):
        check_route(session)


def test_windows_distro_list_removes_wsl_utf16_nuls(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = base64.b64encode("Ubuntu-24.04".encode("utf-16le")).decode()
    monkeypatch.setattr(RemoteSession, "ssh", lambda *args, **kwargs: (0, f"{encoded}\nAA==\n"))
    assert RemoteSession(RemoteConfig())._windows_distros() == ["Ubuntu-24.04"]  # noqa: SLF001


def test_remote_session_uses_argv_stdin_and_total_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = RemoteSession(RemoteConfig())
    session.evidence_dir = tmp_path
    assert session.ssh("bash -s --", input_text="printf ok\n", timeout=17, label="test") == (0, "ok")
    assert isinstance(captured["argv"], list)
    assert captured["input"] == "printf ok\n"
    assert captured["timeout"] == 17
    assert "shell" not in captured
    assert (tmp_path / "01-test.log").stat().st_mode & 0o777 == 0o600


def test_remote_session_streams_once_and_keeps_private_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(remote_module, "ssh_argv", lambda config: ["bash", "-c"])
    session = RemoteSession(RemoteConfig())
    session.evidence_dir = tmp_path
    status, output = session.ssh(
        r"printf '[setup] validating workspace and storage\n[setup] \342'; "
        r"sleep 0.35; printf '\202\254 private-path token\nprivate-out\n'; "
        r"printf '[server] warning\nprivate-err\n' >&2",
        timeout=5,
        label="setup-pc",
        stream=True,
    )
    captured = capsys.readouterr()
    assert status == 0
    assert output == "[setup] validating workspace and storage\n[setup] € private-path token\nprivate-out"
    assert captured.out == ""
    assert captured.err == "[setup] validating workspace and storage\n"
    evidence = tmp_path / "01-setup-pc.log"
    assert evidence.stat().st_mode & 0o777 == 0o600
    assert "[setup] € private-path token" in evidence.read_text(encoding="utf-8")


def test_remote_server_stream_reconstructs_only_valid_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(remote_module, "ssh_argv", lambda config: ["bash", "-c"])
    session = RemoteSession(RemoteConfig())
    session.evidence_dir = tmp_path
    command = "\n".join(
        [
            "printf '[server] loading profile=pi05_aloha_base backend=pytorch; a temporary RAM increase is expected\\n'",
            "printf '[server] still loading; elapsed=10s\\n'",
            "printf '[server] still loading; elapsed=999999s\\n'",
            f"printf '[server] still loading; elapsed={'9' * 5000}s\\n'",
            "printf '[server] warning\\n'",
            "printf '[setup] setup complete\\n'",
        ]
    )
    session.ssh(command, timeout=5, label="server-start", stream=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "[server] loading profile=pi05_aloha_base backend=pytorch; a temporary RAM increase is expected",
        "[server] still loading; elapsed=10s",
    ]
    assert "999999" in (tmp_path / "01-server-start.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["timeout", "interrupted-communicate", "interrupted-clock"])
def test_remote_stream_failure_kills_child_and_preserves_private_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    class Process:
        def __init__(self) -> None:
            self.args = ["ssh"]
            self.returncode = None
            self.killed = False
            self.first = True

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def communicate(self, **kwargs: object) -> tuple[bytes, bytes]:
            if mode == "interrupted-communicate" and self.first:
                self.first = False
                raise KeyboardInterrupt
            return b"private-out", b"private-err"

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    session = RemoteSession(RemoteConfig())
    session.evidence_dir = tmp_path
    if mode.startswith("interrupted"):
        if mode == "interrupted-clock":
            times = iter((0.0,))

            def monotonic() -> float:
                try:
                    return next(times)
                except StopIteration as error:
                    raise KeyboardInterrupt from error

            monkeypatch.setattr(remote_module.time, "monotonic", monotonic)
        with pytest.raises(KeyboardInterrupt):
            session.ssh("probe", timeout=1, label="interrupted", stream=True)
        label = "interrupted"
    else:
        times = iter((0.0, 2.0))
        monkeypatch.setattr(remote_module.time, "monotonic", lambda: next(times))
        with pytest.raises(RemoteError, match="total deadline") as error:
            session.ssh("probe", timeout=1, label="timeout", stream=True)
        assert "private" not in str(error.value)
        label = "timeout"
    assert process.killed
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    evidence = tmp_path / f"01-{label}.log"
    assert evidence.stat().st_mode & 0o777 == 0o600
    assert evidence.read_text(encoding="utf-8").count("private") == 2


def test_remote_session_maps_outer_timeout_without_leaking_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("ssh", 5, output="private", stderr="private")

    monkeypatch.setattr(subprocess, "run", timeout)
    session = RemoteSession(RemoteConfig())
    session.evidence_dir = tmp_path
    with pytest.raises(RemoteError, match="total deadline") as error:
        session.ssh("probe", timeout=5, label="timeout")
    assert "private" not in str(error.value)
    evidence = tmp_path / "01-timeout.log"
    assert evidence.stat().st_mode & 0o777 == 0o600
    assert evidence.read_text(encoding="utf-8").count("private") == 2


def test_transport_failure_is_classified_even_during_shell_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def host_key_failure(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 255, "", "Host key verification failed.")

    monkeypatch.setattr(subprocess, "run", host_key_failure)
    session = RemoteSession(RemoteConfig())
    session.evidence_dir = tmp_path
    with pytest.raises(RemoteError, match="fingerprint-verification"):
        session.ssh("probe", timeout=5, label="route", check=False)


def test_launch_receipt_is_private_and_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config = RemoteConfig()
    _write_launch_receipt(config, "a" * 40, RemoteTarget("powershell", "Ubuntu-22.04"))
    receipt = Path(".runtime/phase2-launch.json")
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert _read_launch_receipt() == {
        "backend": "pytorch",
        "profile": "pi05_aloha_base",
        "port": 8000,
        "remote_dir": "~/src/openpi",
        "route": "powershell",
        "source_sha": "a" * 40,
        "ssh_alias": "robot-gpu",
        "wsl_distro": "Ubuntu-22.04",
    }
    receipt.chmod(0o644)
    with pytest.raises(RemoteError, match="invalid"):
        _read_launch_receipt()


@pytest.mark.parametrize("failure", ["branch", "dirty", "origin", "missing", "stale", "permissive", "symlink"])
def test_candidate_gate_rejects_every_unpublished_or_unscanned_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40

    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return sha
        branch = "codex/06-hardening-docs"
        if arguments == ("branch", "--show-current"):
            return "wrong" if failure == "branch" else branch
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return " M file" if failure == "dirty" else ""
        if arguments == ("rev-parse", f"origin/{branch}"):
            return "b" * 40 if failure == "origin" else sha
        raise AssertionError(arguments)

    monkeypatch.setattr(remote_module, "_git", fake_git)
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    runtime.chmod(0o700)
    receipt = runtime / "secret-scan.sha"
    if failure != "missing":
        receipt.write_text(("b" * 40 if failure == "stale" else sha) + "\n", encoding="utf-8")
        receipt.chmod(0o644 if failure == "permissive" else 0o600)
    if failure == "symlink":
        target = runtime / "target"
        receipt.replace(target)
        receipt.symlink_to(target)
    with pytest.raises(RemoteError):
        _candidate_sha()


@pytest.mark.parametrize("branch", ["codex/06-hardening-docs", "codex/push-pi-scenarios"])
def test_candidate_gate_accepts_exact_clean_pushed_scan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, branch: str
) -> None:
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    answers = {
        ("rev-parse", "HEAD"): sha,
        ("branch", "--show-current"): branch,
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("rev-parse", f"origin/{branch}"): sha,
    }
    monkeypatch.setattr(remote_module, "_git", lambda *arguments: answers[arguments])
    receipt = tmp_path / ".runtime" / "secret-scan.sha"
    receipt.parent.mkdir()
    receipt.parent.chmod(0o700)
    receipt.write_text(sha + "\n", encoding="utf-8")
    receipt.chmod(0o600)
    assert _candidate_sha() == sha


def test_server_passes_jax_memory_fraction_to_wsl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: "a" * 40)
    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04", jax_mem_fraction="0.85", policy_backend="jax"))
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    monkeypatch.setattr(session, "discover_target", lambda: target)
    events = []
    streamed = []

    def fake_run_wsl(*args: object, **kwargs: object) -> str:
        events.append(("server", str(args[1])))
        streamed.append(kwargs.get("stream"))
        return ""

    monkeypatch.setattr(session, "run_wsl", fake_run_wsl)
    monkeypatch.setattr(
        connection_check, "start", lambda config, actual_target: events.append(("holder", actual_target))
    )
    monkeypatch.setattr(remote_module, "route", lambda actual_session: events.append(("route", actual_session)))
    remote_module.server(session)
    script = events[0][1]
    fraction = base64.b64encode(b"0.85").decode()
    backend = base64.b64encode(b"jax").decode()
    candidate = base64.b64encode(("a" * 40).encode()).decode()
    assert f'arg1="$(printf %s {backend} | base64 -d)"' in script
    assert f'arg6="$(printf %s {fraction} | base64 -d)"' in script
    assert f'arg7="$(printf %s {candidate} | base64 -d)"' in script
    assert [event[0] for event in events] == ["server", "holder", "route"]
    assert streamed == [True]


def test_server_failure_stops_remote_before_tunnel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: "a" * 40)
    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04"))
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    monkeypatch.setattr(session, "discover_target", lambda: target)
    events = []
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: events.append("server") or "")

    def fail_holder(*args: object) -> None:
        events.append("holder")
        raise RemoteError("holder failed")

    monkeypatch.setattr(connection_check, "start", fail_holder)
    monkeypatch.setattr(remote_module, "stop", lambda actual_session: events.append("remote-stop"))
    monkeypatch.setattr(connection_check, "stop", lambda config: events.append("tunnel-stop"))
    with pytest.raises(RemoteError, match="holder failed"):
        remote_module.server(session)
    assert events == ["server", "holder", "remote-stop", "tunnel-stop"]


def test_convert_uses_one_bounded_allowlisted_remote_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: "a" * 40)
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04"))
    monkeypatch.setattr(remote_module, "doctor", lambda actual_session: target)
    captured = {}

    def fake_run_wsl(actual_target: RemoteTarget, script: str, **kwargs: object) -> str:
        captured.update({"target": actual_target, "script": script, **kwargs})
        return "\n".join(
            [
                "__ALOHA_CONVERSION__=passed",
                "__ALOHA_CONVERSION_PARTIAL__=absent",
                "__ALOHA_PROFILE__=pi05_aloha_base",
                f"__ALOHA_PROJECT_SHA__={'a' * 40}",
                "__ALOHA_CONVERSION_RESTORE_MODE__=partial-bfloat16",
                "__ALOHA_AVAILABLE_RAM_KIB__=12000000",
                f"__ALOHA_MODEL_HASH__={'b' * 64}",
                "__ALOHA_PROBE_MAX_RSS_KIB__=100",
                "__ALOHA_FULL_MAX_RSS_KIB__=200",
                "__ALOHA_GPU_PEAK_MIB__=300",
                "__ALOHA_GPU_SAMPLES__=4",
                "__ALOHA_REMOTE_EVIDENCE__=.runtime/conversion/20260827T120000Z-123",
            ]
        )

    monkeypatch.setattr(session, "run_wsl", fake_run_wsl)
    remote_module.convert(session)
    assert captured["target"] == target
    assert captured["command_timeout"] == 7200
    assert captured["timeout"] == 7275
    assert "convert_policy_checkpoint.sh" in str(captured["script"])
    assert "bash -c" not in str(captured["script"])
    auto_mode = base64.b64encode(b"auto").decode()
    assert f'arg4="$(printf %s {auto_mode} | base64 -d)"' in str(captured["script"])


def test_convert_rejects_nonpositive_resource_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: "a" * 40)
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04"))
    monkeypatch.setattr(remote_module, "doctor", lambda actual_session: target)
    output = "\n".join(
        [
            "__ALOHA_CONVERSION__=passed",
            "__ALOHA_CONVERSION_PARTIAL__=absent",
            "__ALOHA_PROFILE__=pi05_aloha_base",
            f"__ALOHA_PROJECT_SHA__={'a' * 40}",
            "__ALOHA_CONVERSION_RESTORE_MODE__=partial-bfloat16",
            "__ALOHA_AVAILABLE_RAM_KIB__=12000000",
            f"__ALOHA_MODEL_HASH__={'b' * 64}",
            "__ALOHA_PROBE_MAX_RSS_KIB__=100",
            "__ALOHA_FULL_MAX_RSS_KIB__=200",
            "__ALOHA_GPU_PEAK_MIB__=300",
            "__ALOHA_GPU_SAMPLES__=0",
            "__ALOHA_REMOTE_EVIDENCE__=.runtime/conversion/20260827T120000Z-123",
        ]
    )
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: output)

    with pytest.raises(RemoteError, match="complete validated evidence"):
        remote_module.convert(session)


def test_convert_auto_requires_mode_matching_available_ram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: "a" * 40)
    session = RemoteSession(RemoteConfig(wsl_distro="Ubuntu-24.04"))
    monkeypatch.setattr(remote_module, "doctor", lambda actual_session: RemoteTarget("powershell", "Ubuntu-24.04"))
    output = "\n".join(
        [
            "__ALOHA_CONVERSION__=passed",
            "__ALOHA_CONVERSION_PARTIAL__=absent",
            "__ALOHA_PROFILE__=pi05_aloha_base",
            f"__ALOHA_PROJECT_SHA__={'a' * 40}",
            "__ALOHA_CONVERSION_RESTORE_MODE__=full-float32",
            "__ALOHA_AVAILABLE_RAM_KIB__=12000000",
            f"__ALOHA_MODEL_HASH__={'b' * 64}",
            "__ALOHA_PROBE_MAX_RSS_KIB__=0",
            "__ALOHA_FULL_MAX_RSS_KIB__=200",
            "__ALOHA_GPU_PEAK_MIB__=300",
            "__ALOHA_GPU_SAMPLES__=4",
            "__ALOHA_REMOTE_EVIDENCE__=.runtime/conversion/20260827T120000Z-123",
        ]
    )
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: output)
    with pytest.raises(RemoteError, match="complete validated evidence"):
        remote_module.convert(session)


def test_convert_explicit_full_mode_overrides_low_ram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: "a" * 40)
    config = RemoteConfig(wsl_distro="Ubuntu-24.04", conversion_restore_mode="full-float32")
    session = RemoteSession(config)
    monkeypatch.setattr(remote_module, "doctor", lambda actual_session: RemoteTarget("powershell", "Ubuntu-24.04"))
    output = "\n".join(
        [
            "__ALOHA_CONVERSION__=passed",
            "__ALOHA_CONVERSION_PARTIAL__=absent",
            "__ALOHA_PROFILE__=pi05_aloha_base",
            f"__ALOHA_PROJECT_SHA__={'a' * 40}",
            "__ALOHA_CONVERSION_RESTORE_MODE__=full-float32",
            "__ALOHA_AVAILABLE_RAM_KIB__=12000000",
            f"__ALOHA_MODEL_HASH__={'b' * 64}",
            "__ALOHA_PROBE_MAX_RSS_KIB__=0",
            "__ALOHA_FULL_MAX_RSS_KIB__=200",
            "__ALOHA_GPU_PEAK_MIB__=300",
            "__ALOHA_GPU_SAMPLES__=4",
            "__ALOHA_REMOTE_EVIDENCE__=.runtime/conversion/20260827T120000Z-123",
        ]
    )
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: output)
    remote_module.convert(session)


def test_stop_uses_the_original_receipt_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    launched = RemoteConfig(ssh_alias="original-gpu", remote_dir="/srv/original", policy_port=8123)
    target = RemoteTarget("powershell", "Ubuntu-22.04")
    _write_launch_receipt(launched, "a" * 40, target)
    session = RemoteSession(RemoteConfig(ssh_alias="changed-gpu", remote_dir="/srv/changed", policy_port=9000))
    captured = {}

    def fake_run_wsl(actual_target: RemoteTarget, script: str, **kwargs: object) -> str:
        captured.update({"target": actual_target, "script": script, **kwargs})
        return "__ALOHA_GPU_SAMPLER_STOPPED__=absent" if kwargs["label"] == "gpu-sampler-stop" else ""

    monkeypatch.setattr(session, "run_wsl", fake_run_wsl)
    monkeypatch.setattr(session, "discover_target", lambda: pytest.fail("receipt target should be used"))
    stop(session)
    assert session.config.ssh_alias == "original-gpu"
    assert captured["target"] == target
    assert "/srv/original" not in str(captured["script"])
    assert base64.b64encode(b"/srv/original").decode() in str(captured["script"])
    assert not Path(".runtime/phase2-launch.json").exists()


def test_gpu_sampler_checks_exact_mac_owner_and_stops_through_record(monkeypatch, tmp_path: Path) -> None:
    class Process:
        pid = 4242
        waited = False

        def poll(self):
            return None

        def wait(self, timeout: int):
            self.waited = True
            return -15

    process = Process()
    session = RemoteSession(RemoteConfig())

    def run_wsl(target: RemoteTarget, script: str, **kwargs: object) -> str:
        if kwargs["label"] == "gpu-sampler-stop":
            return "__ALOHA_GPU_SAMPLER_STOPPED__=stopped"
        if kwargs["label"] == "gpu-metrics-copy":
            return "__ALOHA_GPU_METRICS__=" + base64.b64encode(b"metrics\n").decode() + "\n__ALOHA_SERVER_LOG__="
        return "\n".join(
            [
                "__ALOHA_CLOCK_UTC__=2026-08-28T08:00:00.000Z",
                "__ALOHA_CLOCK_MONOTONIC_NS__=1",
            ]
        )

    monkeypatch.setattr(session, "run_wsl", run_wsl)
    monkeypatch.setattr(remote_module, "verify_sampler_record", lambda: SimpleNamespace(pid=4242))
    stopped = []
    monkeypatch.setattr(remote_module, "stop_sampler", lambda **kwargs: stopped.append(kwargs) or True)
    sampler = GpuSampler(
        session,
        RemoteTarget("powershell", "Ubuntu-24.04"),
        process,
        io.BytesIO(),
        io.BytesIO(),
        "b" * 32,
        "pi0_aloha_sim",
        "a" * 40,
        ".runtime/gpu.jsonl",
        ".runtime/server.log",
        {},
    )
    sampler.check()
    assert sampler.stop(tmp_path).read_bytes() == b"metrics\n"
    assert stopped == [{"timeout_seconds": 30}]
    assert process.waited


def test_gpu_sampler_check_rejects_another_recorded_process(monkeypatch) -> None:
    process = SimpleNamespace(pid=4242, poll=lambda: None)
    monkeypatch.setattr(remote_module, "verify_sampler_record", lambda: SimpleNamespace(pid=9999))
    sampler = GpuSampler(
        RemoteSession(RemoteConfig()),
        RemoteTarget("powershell", "Ubuntu-24.04"),
        process,
        io.BytesIO(),
        io.BytesIO(),
        "b" * 32,
        "pi0_aloha_sim",
        "a" * 40,
        ".runtime/gpu.jsonl",
        ".runtime/server.log",
        {},
    )
    with pytest.raises(RemoteError, match="another process"):
        sampler.check()


def test_gpu_sampler_remote_stop_failure_still_cleans_local_owner(monkeypatch, tmp_path: Path) -> None:
    class Process:
        pid = 4242
        waited = False

        def poll(self):
            return None

        def wait(self, timeout: int):
            self.waited = True
            return -15

    process = Process()
    session = RemoteSession(RemoteConfig())
    monkeypatch.setattr(session, "run_wsl", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("remote")))
    stopped = []
    monkeypatch.setattr(remote_module, "stop_sampler", lambda **kwargs: stopped.append(kwargs) or True)
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    sampler = GpuSampler(
        session,
        RemoteTarget("powershell", "Ubuntu-24.04"),
        process,
        stdout,
        stderr,
        "b" * 32,
        "pi0_aloha_sim",
        "a" * 40,
        ".runtime/gpu.jsonl",
        ".runtime/server.log",
        {},
    )
    with pytest.raises(TimeoutError, match="remote"):
        sampler.stop(tmp_path)
    assert stopped == [{"timeout_seconds": 30}]
    assert process.waited
    assert stdout.closed
    assert stderr.closed


def test_smoke_requires_a_second_session_survival_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    candidate = "a" * 40
    monkeypatch.setattr(remote_module, "_candidate_sha", lambda: candidate)
    config = RemoteConfig(wsl_distro="Ubuntu-24.04", policy_backend="pytorch")
    target = RemoteTarget("powershell", "Ubuntu-24.04")
    _write_launch_receipt(config, candidate, target)
    session = RemoteSession(config)
    scripts = []

    def fake_run_wsl(actual_target: RemoteTarget, script: str, **kwargs: object) -> str:
        assert actual_target == target
        scripts.append(script)
        return "__ALOHA_SERVER__=ready" if len(scripts) == 2 else ""

    monkeypatch.setattr(session, "run_wsl", fake_run_wsl)
    smoke(session)
    assert len(scripts) == 2
    assert "smoke_policy.sh" in scripts[0]
    assert "check_policy_server.sh" in scripts[1]
