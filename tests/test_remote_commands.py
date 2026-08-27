import base64
from pathlib import Path
import subprocess

import pytest

from tools.remote_aloha import remote as remote_module
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.remote import RemoteSession
from tools.remote_aloha.remote import RemoteTarget
from tools.remote_aloha.remote import _candidate_sha
from tools.remote_aloha.remote import _doctor_script
from tools.remote_aloha.remote import _read_launch_receipt
from tools.remote_aloha.remote import _remote_script_command
from tools.remote_aloha.remote import _setup_script
from tools.remote_aloha.remote import _write_launch_receipt
from tools.remote_aloha.remote import build_wsl_command
from tools.remote_aloha.remote import classify_route
from tools.remote_aloha.remote import select_ubuntu_distro
from tools.remote_aloha.remote import ssh_argv
from tools.remote_aloha.remote import stop


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
    assert "$payload | wsl.exe --distribution $distro --exec bash -s --" in decoded
    assert build_wsl_command("bash", "") == "bash -s --"
    assert build_wsl_command("bash", "", 19) == "timeout --signal=TERM --kill-after=30s 19s bash -s --"
    for route in ("powershell", "cmd"):
        assert "--exec timeout --signal=TERM --kill-after=30s 19s bash -s --" in _decode_powershell(
            build_wsl_command(route, "Ubuntu", 19)
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
    prefix = _setup_script(RemoteConfig(), "a" * 40).split("lifecycle_state=", 1)[0]
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


@pytest.mark.parametrize(
    "script",
    [
        _doctor_script(8000),
        _setup_script(RemoteConfig(), "a" * 40),
        _remote_script_command(RemoteConfig(), "example.sh", ["one", "two"]),
    ],
)
def test_generated_wsl_bash_is_valid(script: str) -> None:
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


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
        "profile": "pi0_aloha_sim",
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
        if arguments == ("branch", "--show-current"):
            return "wrong" if failure == "branch" else remote_module.PHASE_BRANCH
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return " M file" if failure == "dirty" else ""
        if arguments == ("rev-parse", f"origin/{remote_module.PHASE_BRANCH}"):
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


def test_candidate_gate_accepts_exact_clean_pushed_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    sha = "a" * 40
    answers = {
        ("rev-parse", "HEAD"): sha,
        ("branch", "--show-current"): remote_module.PHASE_BRANCH,
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("rev-parse", f"origin/{remote_module.PHASE_BRANCH}"): sha,
    }
    monkeypatch.setattr(remote_module, "_git", lambda *arguments: answers[arguments])
    receipt = tmp_path / ".runtime" / "secret-scan.sha"
    receipt.parent.mkdir()
    receipt.parent.chmod(0o700)
    receipt.write_text(sha + "\n", encoding="utf-8")
    receipt.chmod(0o600)
    assert _candidate_sha() == sha


def test_stop_uses_the_original_receipt_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    launched = RemoteConfig(ssh_alias="original-gpu", remote_dir="/srv/original", policy_port=8123)
    target = RemoteTarget("powershell", "Ubuntu-22.04")
    _write_launch_receipt(launched, "a" * 40, target)
    session = RemoteSession(RemoteConfig(ssh_alias="changed-gpu", remote_dir="/srv/changed", policy_port=9000))
    captured = {}

    def fake_run_wsl(actual_target: RemoteTarget, script: str, **kwargs: object) -> str:
        captured.update({"target": actual_target, "script": script, **kwargs})
        return ""

    monkeypatch.setattr(session, "run_wsl", fake_run_wsl)
    monkeypatch.setattr(session, "discover_target", lambda: pytest.fail("receipt target should be used"))
    stop(session)
    assert session.config.ssh_alias == "original-gpu"
    assert captured["target"] == target
    assert "/srv/original" not in str(captured["script"])
    assert base64.b64encode(b"/srv/original").decode() in str(captured["script"])
    assert not Path(".runtime/phase2-launch.json").exists()
