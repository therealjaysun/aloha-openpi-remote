from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.config import load_remote_config

PUBLIC_REPO = "https://github.com/therealjaysun/pi-robotics.git"
PHASE_BRANCH = "codex/03-secure-connectivity"
UPSTREAM_SHA = "215abfb217dbac7d5f1273282331b9b1866c0479"
_ROUTES = {"bash", "powershell", "cmd"}
_SCAN_RECEIPT = Path(".runtime/secret-scan.sha")
_LAUNCH_RECEIPT = Path(".runtime/phase2-launch.json")


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteTarget:
    route: str
    distro: str


def ssh_argv(config: RemoteConfig) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.ssh_alias):
        raise ValueError("unsafe SSH alias")
    timeout = config.ssh_connect_timeout_seconds
    if not 1 <= timeout <= 300:
        raise ValueError("unsafe SSH connect timeout")
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=1",
        "-o",
        "ClearAllForwardings=yes",
        config.ssh_alias,
    ]


def encode_powershell(script: str) -> str:
    return base64.b64encode(script.encode("utf-16le")).decode("ascii")


def powershell_command(script: str) -> str:
    return f"powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand {encode_powershell(script)}"


def build_wsl_command(route: str, distro: str, command_timeout: int | None = None) -> str:
    if route not in _ROUTES:
        raise ValueError(f"unsupported remote route: {route}")
    if command_timeout is not None and not 1 <= command_timeout <= 7200:
        raise ValueError("unsafe WSL command timeout")
    linux_command = "bash -s --"
    if command_timeout is not None:
        linux_command = f"timeout --signal=TERM --kill-after=30s {command_timeout}s bash -s --"
    if route == "bash":
        return linux_command
    if not distro or distro.startswith("-") or any(ord(character) < 32 for character in distro):
        raise ValueError("a safe explicit WSL distro is required for a Windows route")
    encoded_distro = base64.b64encode(distro.encode()).decode("ascii")
    launcher = rf"""
$ErrorActionPreference = 'Stop'
$distro = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_distro}'))
$payload = [Console]::In.ReadToEnd()
$payload | wsl.exe --distribution $distro --exec bash -c "tr -d '\r' | {linux_command}"
exit $LASTEXITCODE
""".strip()
    return powershell_command(launcher)


def classify_route(outputs: dict[str, tuple[int, str]]) -> str:
    markers = {
        "bash": "__ALOHA_ROUTE_BASH__",
        "powershell": "__ALOHA_ROUTE_POWERSHELL__",
        "cmd": "__ALOHA_ROUTE_CMD__",
    }
    matches = [route for route, marker in markers.items() if outputs.get(route) == (0, marker)]
    if len(matches) != 1:
        raise RemoteError("could not uniquely detect the remote shell route")
    return matches[0]


def select_ubuntu_distro(discovered: dict[str, str], configured: str) -> str:
    if configured:
        if discovered.get(configured) != "ubuntu":
            raise RemoteError("OPENPI_WSL_DISTRO is not a discovered Ubuntu WSL distro")
        return configured
    ubuntu = [name for name, os_id in discovered.items() if os_id == "ubuntu"]
    if len(ubuntu) != 1:
        raise RemoteError("set OPENPI_WSL_DISTRO because exactly one Ubuntu distro was not discovered")
    return ubuntu[0]


def windows_listener_addresses_are_private(addresses: object) -> bool:
    return isinstance(addresses, list) and all(
        isinstance(address, str) and address in {"127.0.0.1", "::1"} for address in addresses
    )


def _encoded_assignment(name: str, value: str) -> str:
    encoded = base64.b64encode(value.encode()).decode("ascii")
    return f'{name}="$(printf %s {encoded} | base64 -d)"'


def _markers(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.replace("\x00", "").replace("\r", "").splitlines():
        match = re.fullmatch(r"__ALOHA_([A-Z0-9_]+)__=(.*)", line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


class RemoteSession:
    def __init__(self, config: RemoteConfig) -> None:
        self.config = config
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")  # noqa: UP017 (Python 3.10)
        self.evidence_dir = Path("outputs") / "phase03" / timestamp
        self._counter = 0

    def _save_raw(self, label: str, result: subprocess.CompletedProcess[str]) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError("unsafe evidence label")
        self.evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.evidence_dir.chmod(0o700)
        self._counter += 1
        path = self.evidence_dir / f"{self._counter:02d}-{label}.log"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"exit={result.returncode}\nSTDOUT\n{result.stdout or ''}\nSTDERR\n{result.stderr or ''}\n")

    def ssh(
        self,
        command: str,
        *,
        input_text: str = "",
        timeout: int,
        label: str,
        check: bool = True,
    ) -> tuple[int, str]:
        try:
            result = subprocess.run(
                [*ssh_argv(self.config), command],
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
            self._save_raw(label, subprocess.CompletedProcess(error.cmd, -1, stdout, stderr))
            raise RemoteError(f"remote {label} exceeded its total deadline") from error
        self._save_raw(label, result)
        output = result.stdout.replace("\x00", "").replace("\r", "").strip()
        if result.returncode:
            error = result.stderr.lower()
            if "host key verification failed" in error:
                detail = "SSH host trust is missing or changed; complete the fingerprint-verification gate"
            elif "permission denied" in error:
                detail = "SSH authentication failed; verify the private robot-gpu alias and key"
            elif "could not resolve" in error or "name or service not known" in error:
                detail = "the private robot-gpu destination could not be resolved"
            elif "connection refused" in error:
                detail = "the PC SSH service refused the connection"
            elif "operation timed out" in error or "connection timed out" in error:
                detail = "the PC SSH connection timed out"
            else:
                detail = ""
            if detail:
                raise RemoteError(detail)
            if check:
                raise RemoteError(f"remote {label} failed; inspect the ignored mode-600 evidence log")
        return result.returncode, output

    def detect_route(self) -> str:
        try:
            result = subprocess.run(
                ["ssh", "-G", self.config.ssh_alias],
                capture_output=True,
                text=True,
                timeout=self.config.ssh_connect_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RemoteError("the private SSH alias could not be validated") from error
        hostname = next(
            (line.split(maxsplit=1)[1] for line in result.stdout.splitlines() if line.startswith("hostname ")),
            "",
        )
        if result.returncode or not hostname or hostname == self.config.ssh_alias:
            raise RemoteError("configure the private robot-gpu SSH alias before remote work")
        timeout = self.config.ssh_connect_timeout_seconds + 10
        probes = {
            "bash": "printf '%s\\n' __ALOHA_ROUTE_BASH__",
            "powershell": "Write-Output __ALOHA_ROUTE_POWERSHELL__",
            "cmd": "@echo __ALOHA_ROUTE_CMD__",
        }
        outputs = {
            route: self.ssh(command, timeout=timeout, label=f"route-{route}", check=False)
            for route, command in probes.items()
        }
        return classify_route(outputs)

    def run_wsl(
        self,
        target: RemoteTarget,
        script: str,
        *,
        timeout: int,
        label: str,
        command_timeout: int | None = None,
    ) -> str:
        _, output = self.ssh(
            build_wsl_command(target.route, target.distro, command_timeout),
            input_text=script,
            timeout=timeout,
            label=label,
        )
        return output

    def _windows_distros(self) -> list[str]:
        script = """
$ErrorActionPreference = 'Stop'
& wsl.exe --list --quiet | ForEach-Object {
    $name = $_.Trim()
    if ($name) { [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($name)) }
}
exit $LASTEXITCODE
""".strip()
        _, output = self.ssh(
            powershell_command(script),
            timeout=self.config.ssh_connect_timeout_seconds + 20,
            label="wsl-list",
        )
        names = []
        for line in output.splitlines():
            try:
                name = base64.b64decode(line, validate=True).decode().replace("\x00", "")
            except (ValueError, UnicodeDecodeError) as error:
                raise RemoteError("WSL returned an unreadable distro list") from error
            if name:
                names.append(name)
        return names

    def discover_target(self) -> RemoteTarget:
        route = self.detect_route()
        if route == "bash":
            output = self.run_wsl(
                RemoteTarget(route, ""),
                "printf '__ALOHA_DISTRO__='; printf %s \"${WSL_DISTRO_NAME:-}\" | base64; printf '\\n'\n",
                timeout=self.config.ssh_connect_timeout_seconds + 10,
                label="wsl-direct-distro",
            )
            encoded = _markers(output).get("DISTRO", "")
            try:
                distro = base64.b64decode(encoded, validate=True).decode()
            except (ValueError, UnicodeDecodeError) as error:
                raise RemoteError("direct WSL shell did not identify its distro") from error
            if not distro or (self.config.wsl_distro and distro != self.config.wsl_distro):
                raise RemoteError("direct WSL distro does not match OPENPI_WSL_DISTRO")
            target = RemoteTarget(route, distro)
            os_id = _markers(
                self.run_wsl(
                    target,
                    ". /etc/os-release; printf '__ALOHA_OS_ID__=%s\\n' \"$ID\"\n",
                    timeout=self.config.ssh_connect_timeout_seconds + 10,
                    label="wsl-direct-os",
                )
            ).get("OS_ID")
            if os_id != "ubuntu":
                raise RemoteError("the direct WSL shell is not Ubuntu")
            return target

        discovered: dict[str, str] = {}
        names = self._windows_distros()
        if self.config.wsl_distro:
            names = [name for name in names if name == self.config.wsl_distro]
        for index, name in enumerate(names, start=1):
            target = RemoteTarget(route, name)
            try:
                result = self.run_wsl(
                    target,
                    "set -e; . /etc/os-release; printf '__ALOHA_OS_ID__=%s\\n' \"$ID\"\n",
                    timeout=self.config.ssh_connect_timeout_seconds + 15,
                    label=f"wsl-os-{index}",
                )
            except RemoteError:
                continue
            discovered[name] = _markers(result).get("OS_ID", "")
        return RemoteTarget(route, select_ubuntu_distro(discovered, self.config.wsl_distro))


def _doctor_script(port: int) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
. /etc/os-release
case "$(uname -r)" in *[Mm]icrosoft*WSL2*|*[Mm]icrosoft-standard-WSL2*) wsl2=yes ;; *) wsl2=no ;; esac
smi="$(command -v nvidia-smi || true)"
[[ -n "$smi" ]] || [[ ! -x /usr/lib/wsl/lib/nvidia-smi ]] || smi=/usr/lib/wsl/lib/nvidia-smi
[[ -n "$smi" ]] || {{ echo 'nvidia-smi is unavailable in WSL' >&2; exit 1; }}
gpu_line="$($smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | grep -m1 '3090' || true)"
[[ -n "$gpu_line" ]] || {{ echo 'RTX 3090 was not detected' >&2; exit 1; }}
IFS=, read -r gpu_name gpu_memory gpu_driver <<<"$gpu_line"
port_state=free
ss -H -ltn 'sport = :{port}' | grep -q . && port_state=listening
printf '__ALOHA_OS_ID__=%s\n' "$ID"
printf '__ALOHA_OS_VERSION__=%s\n' "$VERSION_ID"
printf '__ALOHA_WSL2__=%s\n' "$wsl2"
printf '__ALOHA_ARCH__=%s\n' "$(uname -m)"
printf '__ALOHA_GPU_NAME__=%s\n' "${{gpu_name# }}"
printf '__ALOHA_GPU_MEMORY_MIB__=%s\n' "${{gpu_memory// /}}"
printf '__ALOHA_DRIVER__=%s\n' "${{gpu_driver// /}}"
printf '__ALOHA_DISK_FREE_KIB__=%s\n' "$(df -Pk "$HOME" | awk 'NR==2 {{print $4}}')"
printf '__ALOHA_RAM_AVAILABLE_KIB__=%s\n' "$(awk '$1 == \"MemAvailable:\" {{print $2; exit}}' /proc/meminfo)"
printf '__ALOHA_PORT__=%s\n' "$port_state"
printf '__ALOHA_PYTHON__=%s\n' "$(python3 --version 2>&1 || printf missing)"
printf '__ALOHA_UV__=%s\n' "$(uv --version 2>/dev/null || printf missing)"
missing=()
for tool in base64 cc curl flock git realpath ss timeout; do command -v "$tool" >/dev/null || missing+=("$tool"); done
[[ -x /usr/bin/time ]] && /usr/bin/time --version 2>&1 | grep -qi 'GNU time' || missing+=(gnu-time)
[[ -r /usr/include/linux/input.h && -r /usr/include/linux/input-event-codes.h ]] || missing+=(linux-input-headers)
printf '__ALOHA_TOOLS__=%s\n' "${{missing[*]:-ready}}"
"""


def doctor(session: RemoteSession, target: RemoteTarget | None = None) -> RemoteTarget:
    target = target or session.discover_target()
    output = session.run_wsl(
        target,
        _doctor_script(session.config.policy_port),
        timeout=session.config.ssh_connect_timeout_seconds + 30,
        label="doctor-pc",
    )
    facts = _markers(output)
    version = facts.get("OS_VERSION")
    if facts.get("OS_ID") != "ubuntu" or version not in {"22.04", "24.04"}:
        raise RemoteError("the selected WSL distro must be Ubuntu 22.04 or 24.04")
    if version == "24.04" and session.config.wsl_distro != target.distro:
        raise RemoteError("Ubuntu 24.04 is experimental; select it explicitly with OPENPI_WSL_DISTRO")
    if facts.get("WSL2") != "yes" or facts.get("ARCH") != "x86_64":
        raise RemoteError("the selected distro must be x86_64 WSL2")
    if "3090" not in facts.get("GPU_NAME", ""):
        raise RemoteError("RTX 3090 validation failed")
    if facts.get("TOOLS") != "ready":
        raise RemoteError(f"required WSL tools are missing: {facts.get('TOOLS', 'unknown')}")
    if facts.get("UV") == "missing":
        raise RemoteError("uv is missing in the selected WSL distro")
    available_ram_kib = facts.get("RAM_AVAILABLE_KIB", "")
    if not available_ram_kib.isdigit() or int(available_ram_kib) <= 0:
        raise RemoteError("WSL available system RAM could not be measured")
    summary = {
        "os": f"Ubuntu {version} WSL2",
        "openpi_support": "upstream" if version == "22.04" else "experimental",
        "architecture": facts["ARCH"],
        "gpu": facts["GPU_NAME"],
        "gpu_memory_mib": facts.get("GPU_MEMORY_MIB"),
        "driver": facts.get("DRIVER"),
        "disk_free_kib": facts.get("DISK_FREE_KIB"),
        "ram_available_kib": int(available_ram_kib),
        "automatic_conversion_restore_mode": (
            "partial-bfloat16" if int(available_ram_kib) < 16 * 1024 * 1024 else "full-float32"
        ),
        "policy_port": facts.get("PORT"),
        "python": facts.get("PYTHON"),
        "uv": facts.get("UV"),
    }
    print(json.dumps(summary, sort_keys=True))
    return target


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RemoteError("could not validate the local remote-test candidate") from error
    return result.stdout.strip()


def _candidate_sha() -> str:
    sha = _git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RemoteError("local candidate SHA is invalid")
    if _git("branch", "--show-current") != PHASE_BRANCH:
        raise RemoteError(f"remote work requires branch {PHASE_BRANCH}")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RemoteError("remote work requires a clean candidate checkout")
    if _git("rev-parse", f"origin/{PHASE_BRANCH}") != sha:
        raise RemoteError("push the exact Phase 3 candidate before remote work")
    try:
        runtime_metadata = _SCAN_RECEIPT.parent.lstat()
        metadata = _SCAN_RECEIPT.lstat()
        scanned = _SCAN_RECEIPT.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RemoteError("run make secret-scan on the exact pushed candidate before remote work") from error
    if (
        stat.S_ISLNK(runtime_metadata.st_mode)
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or runtime_metadata.st_mode & 0o077
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or scanned != sha
    ):
        raise RemoteError("run make secret-scan on the exact pushed candidate before remote work")
    return sha


def _write_launch_receipt(config: RemoteConfig, candidate: str, target: RemoteTarget) -> None:
    if _LAUNCH_RECEIPT.parent.is_symlink():
        raise RemoteError(".runtime must be a real private directory")
    _LAUNCH_RECEIPT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not _LAUNCH_RECEIPT.parent.is_dir():
        raise RemoteError(".runtime must be a real private directory")
    _LAUNCH_RECEIPT.parent.chmod(0o700)
    payload = {
        "backend": config.policy_backend,
        "profile": config.policy_profile.name,
        "port": config.policy_port,
        "remote_dir": config.remote_dir,
        "route": target.route,
        "source_sha": candidate,
        "ssh_alias": config.ssh_alias,
        "wsl_distro": target.distro,
    }
    try:
        descriptor = os.open(_LAUNCH_RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RemoteError("a policy launch receipt already exists; run make stop first") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")


def _read_launch_receipt() -> dict[str, object] | None:
    if not _LAUNCH_RECEIPT.exists() and not _LAUNCH_RECEIPT.is_symlink():
        return None
    try:
        runtime_metadata = _LAUNCH_RECEIPT.parent.lstat()
        metadata = _LAUNCH_RECEIPT.lstat()
        payload = json.loads(_LAUNCH_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RemoteError("the policy launch receipt is unreadable; refusing lifecycle changes") from error
    expected = {"backend", "profile", "port", "remote_dir", "route", "source_sha", "ssh_alias", "wsl_distro"}
    if not isinstance(payload, dict):
        raise RemoteError("the policy launch receipt is invalid; refusing lifecycle changes")
    valid_path = isinstance(payload.get("remote_dir"), str) and payload["remote_dir"].startswith(("/", "~/"))
    if (
        stat.S_ISLNK(runtime_metadata.st_mode)
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or runtime_metadata.st_mode & 0o077
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or set(payload) != expected
        or payload.get("backend") not in {"jax", "pytorch"}
        or payload.get("profile") not in {"pi0_aloha_sim", "pi05_aloha_base"}
        or payload.get("route") not in _ROUTES
        or not isinstance(payload.get("ssh_alias"), str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", payload["ssh_alias"])
        or not isinstance(payload.get("wsl_distro"), str)
        or not payload["wsl_distro"]
        or not isinstance(payload.get("port"), int)
        or not 1 <= payload["port"] <= 65535
        or not valid_path
        or not isinstance(payload.get("source_sha"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", payload["source_sha"])
    ):
        raise RemoteError("the policy launch receipt is invalid; refusing lifecycle changes")
    return payload


def _setup_script(config: RemoteConfig, candidate: str) -> str:
    data_home = config.data_home
    return f"""#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
umask 077
{_encoded_assignment('remote_input', config.remote_dir)}
{_encoded_assignment('data_input', data_home)}
{_encoded_assignment('repo_url', PUBLIC_REPO)}
{_encoded_assignment('legacy_repo_url', 'https://github.com/therealjaysun/aloha-openpi-remote.git')}
{_encoded_assignment('branch', PHASE_BRANCH)}
{_encoded_assignment('candidate', candidate)}
{_encoded_assignment('upstream_sha', UPSTREAM_SHA)}
min_free_gib={config.min_free_gib}
policy_port={config.policy_port}
case "$remote_input" in "~/"*) remote_dir="$HOME/${{remote_input:2}}" ;; /*) remote_dir="$remote_input" ;; *) exit 2 ;; esac
if [[ -z "$data_input" ]]; then data_home="$HOME/.cache/openpi"; else data_home="$data_input"; fi
[[ "$data_home" == /* ]] || {{ echo 'OPENPI_DATA_HOME must resolve to an absolute path' >&2; exit 2; }}
lifecycle_state="$HOME/.local/state/aloha-openpi-remote"
mkdir -p "$lifecycle_state"
chmod 700 "$lifecycle_state"
exec 9>"$lifecycle_state/lifecycle.lock"
flock -n 9 || {{ echo 'Another policy lifecycle operation is active.' >&2; exit 1; }}
[[ ! -e "$remote_dir/.runtime/server.json" && ! -L "$remote_dir/.runtime/server.json" ]] || {{
    echo 'Stop the owned policy server before setup.' >&2
    exit 1
}}
ss -H -ltn "sport = :$policy_port" | grep -q . && {{ echo 'Policy port is occupied; stop it before setup.' >&2; exit 1; }}
mkdir -p "$(dirname "$remote_dir")" "$data_home"
require_space() {{
    local available
    available="$(df -Pk "$1" | awk 'NR==2 {{print $4}}')"
    (( available >= min_free_gib * 1024 * 1024 )) || {{ echo 'Insufficient WSL disk space.' >&2; exit 1; }}
}}
require_space "$(dirname "$remote_dir")"
require_space "$data_home"
created=no
if [[ ! -e "$remote_dir" && ! -L "$remote_dir" ]]; then
    git clone --filter=blob:none --no-checkout "$repo_url" "$remote_dir"
    created=yes
    mkdir -p "$remote_dir/.runtime"
    printf '%s\n' "$repo_url" >"$remote_dir/.runtime/managed-checkout"
    chmod 600 "$remote_dir/.runtime/managed-checkout"
elif [[ -L "$remote_dir" || ! -d "$remote_dir/.git" ]]; then
    echo 'OPENPI_REMOTE_DIR exists but is not the project repository; move it aside.' >&2
    exit 1
fi
marker="$remote_dir/.runtime/managed-checkout"
origin_url="$(git -C "$remote_dir" remote get-url origin)"
marker_url="$(cat "$marker" 2>/dev/null || true)"
if [[ -f "$marker" && ! -L "$marker" && "$marker_url" == "$legacy_repo_url" ]] &&
    [[ "$origin_url" == "$legacy_repo_url" || "$origin_url" == "$repo_url" ]]; then
    git -C "$remote_dir" remote set-url origin "$repo_url"
    printf '%s\n' "$repo_url" >"$marker"
    origin_url="$repo_url"
    marker_url="$repo_url"
fi
[[ "$origin_url" == "$repo_url" ]] || {{ echo 'Remote origin mismatch.' >&2; exit 1; }}
[[ -f "$marker" && ! -L "$marker" && "$marker_url" == "$repo_url" ]] || {{
    echo 'Refusing to mutate a checkout not created by this project.' >&2
    exit 1
}}
[[ "$created" == yes ]] || [[ -z "$(git -C "$remote_dir" status --porcelain --untracked-files=all)" ]] || {{
    echo 'Remote checkout is dirty.' >&2
    exit 1
}}
git -C "$remote_dir" fetch --no-tags origin "refs/heads/$branch"
[[ "$(git -C "$remote_dir" rev-parse FETCH_HEAD)" == "$candidate" ]] || {{ echo 'Published branch does not match candidate SHA.' >&2; exit 1; }}
git -C "$remote_dir" cat-file -e "$upstream_sha^{{commit}}"
git -C "$remote_dir" merge-base --is-ancestor "$upstream_sha" "$candidate"
git -C "$remote_dir" checkout --detach "$candidate"
git -C "$remote_dir" submodule sync --recursive
git -C "$remote_dir" submodule update --init --recursive
! git -C "$remote_dir" submodule status --recursive | grep -Eq '^[+-U]'
[[ -z "$(git -C "$remote_dir" status --porcelain --untracked-files=all)" ]]
command -v uv >/dev/null || {{ echo 'uv is missing in WSL; install uv, then rerun make setup-pc.' >&2; exit 1; }}
cd "$remote_dir"
GIT_LFS_SKIP_SMUDGE=1 OPENPI_DATA_HOME="$data_home" uv sync --locked
checkpoint_cache="$data_home/openpi-assets/checkpoints"
if [[ -d "$checkpoint_cache" ]] && find "$checkpoint_cache" -name '*.partial' -print -quit | grep -q .; then
    echo 'A partial OpenPI cache entry exists; it was preserved for diagnosis.' >&2
    exit 1
fi
OPENPI_DATA_HOME="$data_home" .venv/bin/python - <<'PY'
import jax

assert jax.default_backend() == "gpu", f"CPU fallback rejected: {{jax.default_backend()}}"
assert any("3090" in device.device_kind for device in jax.devices()), jax.devices()
print("JAX GPU validation passed.")
PY
printf '__ALOHA_PROJECT_SHA__=%s\n' "$(git rev-parse HEAD)"
printf '__ALOHA_UPSTREAM_SHA__=%s\n' "$upstream_sha"
printf '__ALOHA_SETUP__=passed\n'
"""


def setup(session: RemoteSession) -> None:
    if _read_launch_receipt() is not None:
        raise RemoteError("run make stop before changing the remote installation")
    candidate = _candidate_sha()
    target = doctor(session)
    output = session.run_wsl(
        target,
        _setup_script(session.config, candidate),
        timeout=session.config.server_startup_timeout_seconds + 45,
        label="setup-pc",
        command_timeout=session.config.server_startup_timeout_seconds,
    )
    facts = _markers(output)
    if facts.get("SETUP") != "passed" or facts.get("PROJECT_SHA") != candidate:
        raise RemoteError("remote setup did not prove the exact candidate SHA")
    print(f"Remote setup passed at candidate {candidate}.")


def convert(session: RemoteSession) -> None:
    if _read_launch_receipt() is not None:
        raise RemoteError("run make stop before checkpoint conversion")
    config = session.config
    candidate = _candidate_sha()
    target = doctor(session)
    args = [
        config.policy_profile.name,
        config.data_home,
        candidate,
        str(config.policy_port),
        config.conversion_restore_mode,
    ]
    output = session.run_wsl(
        target,
        _remote_script_command(config, "convert_policy_checkpoint.sh", args),
        timeout=7275,
        label="checkpoint-conversion",
        command_timeout=7200,
    )
    facts = _markers(output)
    model_hash = facts.get("MODEL_HASH", "")
    remote_evidence = facts.get("REMOTE_EVIDENCE", "")
    selected_mode = facts.get("CONVERSION_RESTORE_MODE", "")
    available_ram_kib = facts.get("AVAILABLE_RAM_KIB", "")
    metric_names = ("FULL_MAX_RSS_KIB", "GPU_PEAK_MIB", "GPU_SAMPLES")
    metrics = {name: facts.get(name, "") for name in metric_names}
    expected_mode = config.conversion_restore_mode
    if expected_mode == "auto" and available_ram_kib.isdigit():
        expected_mode = "partial-bfloat16" if int(available_ram_kib) < 16 * 1024 * 1024 else "full-float32"
    probe_rss = facts.get("PROBE_MAX_RSS_KIB", "")
    if (
        facts.get("CONVERSION") != "passed"
        or facts.get("CONVERSION_PARTIAL") != "absent"
        or facts.get("PROFILE") != config.policy_profile.name
        or facts.get("PROJECT_SHA") != candidate
        or selected_mode != expected_mode
        or not available_ram_kib.isdigit()
        or int(available_ram_kib) <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", model_hash)
        or not re.fullmatch(r"\.runtime/conversion/[0-9]{8}T[0-9]{6}Z-[0-9]+", remote_evidence)
        or any(not value.isdigit() or int(value) <= 0 for value in metrics.values())
        or not probe_rss.isdigit()
        or (selected_mode == "partial-bfloat16" and int(probe_rss) <= 0)
        or (selected_mode == "full-float32" and int(probe_rss) != 0)
    ):
        raise RemoteError("checkpoint conversion did not return complete validated evidence")
    summary = {
        "full_max_rss_kib": int(metrics["FULL_MAX_RSS_KIB"]),
        "gpu_peak_mib": int(metrics["GPU_PEAK_MIB"]),
        "gpu_samples": int(metrics["GPU_SAMPLES"]),
        "model_hash": model_hash,
        "probe_max_rss_kib": int(probe_rss),
        "profile": config.policy_profile.name,
        "restore_mode": selected_mode,
        "available_ram_kib": int(available_ram_kib),
        "remote_evidence": remote_evidence,
        "status": "passed",
    }
    print(json.dumps(summary, sort_keys=True))


def _remote_script_command(config: RemoteConfig, script_name: str, arguments: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'export PATH="$HOME/.local/bin:$PATH"',
        _encoded_assignment("remote_input", config.remote_dir),
    ]
    lines.append(
        'case "$remote_input" in "~/"*) repo="$HOME/${remote_input:2}" ;; /*) repo="$remote_input" ;; *) exit 2 ;; esac'
    )
    for index, value in enumerate(arguments):
        lines.append(_encoded_assignment(f"arg{index}", value))
    quoted_args = " ".join(f'"$arg{index}"' for index in range(len(arguments)))
    lines.append('cd "$repo"')
    lines.append(f'exec "$repo/scripts/{script_name}" {quoted_args}')
    return "\n".join(lines) + "\n"


def server(session: RemoteSession) -> None:
    if _read_launch_receipt() is not None:
        raise RemoteError("a policy launch receipt already exists; run make stop first")
    config = session.config
    candidate = _candidate_sha()
    target = session.discover_target()
    args = [
        config.policy_profile.name,
        config.policy_backend,
        config.policy_host,
        str(config.policy_port),
        str(config.server_startup_timeout_seconds),
        config.data_home,
        config.jax_mem_fraction,
        candidate,
    ]
    _write_launch_receipt(config, candidate, target)
    session.run_wsl(
        target,
        _remote_script_command(config, "start_policy_server.sh", args),
        timeout=config.server_startup_timeout_seconds + 75,
        label="server-start",
    )
    check = _remote_script_command(
        config,
        "check_policy_server.sh",
        [config.policy_profile.name, config.policy_host, str(config.policy_port), candidate],
    )
    facts = _markers(
        session.run_wsl(
            target,
            check,
            timeout=config.ssh_connect_timeout_seconds + 15,
            label="server-survival",
        )
    )
    if facts.get("SERVER") != "ready":
        raise RemoteError("policy server did not survive the start SSH session")
    print(f"{config.policy_backend} policy server ready for {config.policy_profile.name} on WSL loopback.")


def stop(session: RemoteSession) -> None:
    receipt = _read_launch_receipt()
    config = session.config
    if receipt is not None:
        config = replace(
            config,
            policy_backend=str(receipt["backend"]),
            ssh_alias=str(receipt["ssh_alias"]),
            remote_dir=str(receipt["remote_dir"]),
            wsl_distro=str(receipt["wsl_distro"]),
            policy_port=int(receipt["port"]),
        )
        session.config = config
        target = RemoteTarget(str(receipt["route"]), str(receipt["wsl_distro"]))
    else:
        target = session.discover_target()
    session.run_wsl(
        target,
        _remote_script_command(config, "stop_policy_server.sh", [str(config.policy_port)]),
        timeout=config.ssh_connect_timeout_seconds + 60,
        label="server-stop",
    )
    if receipt is not None:
        _LAUNCH_RECEIPT.unlink()
    print("Owned policy server stopped or already absent.")


def smoke(session: RemoteSession) -> None:
    config = session.config
    candidate = _candidate_sha()
    receipt = _read_launch_receipt()
    if receipt is None or (
        receipt["profile"],
        receipt["backend"],
        receipt["port"],
        receipt["remote_dir"],
        receipt["source_sha"],
        receipt["ssh_alias"],
    ) != (
        config.policy_profile.name,
        config.policy_backend,
        config.policy_port,
        config.remote_dir,
        candidate,
        config.ssh_alias,
    ):
        raise RemoteError("the running-server receipt does not match this smoke-test configuration")
    if config.wsl_distro and receipt["wsl_distro"] != config.wsl_distro:
        raise RemoteError("the running-server receipt does not match OPENPI_WSL_DISTRO")
    target = RemoteTarget(str(receipt["route"]), str(receipt["wsl_distro"]))
    args = [
        config.policy_profile.name,
        config.policy_backend,
        config.policy_host,
        str(config.policy_port),
        str(config.policy_inference_timeout_seconds),
        candidate,
    ]
    session.run_wsl(
        target,
        _remote_script_command(config, "smoke_policy.sh", args),
        timeout=config.policy_inference_timeout_seconds + 30,
        label="policy-smoke",
    )
    facts = _markers(
        session.run_wsl(
            target,
            _remote_script_command(
                config,
                "check_policy_server.sh",
                [config.policy_profile.name, config.policy_host, str(config.policy_port), candidate],
            ),
            timeout=config.ssh_connect_timeout_seconds + 15,
            label="policy-survival",
        )
    )
    if facts.get("SERVER") != "ready":
        raise RemoteError("policy server did not survive inference")
    print(f"WSL-local inference passed for {config.policy_profile.name} at {candidate}.")


def route(session: RemoteSession) -> None:
    config = session.config
    candidate = _candidate_sha()
    receipt = _read_launch_receipt()
    if receipt is None or (
        receipt["profile"],
        receipt["backend"],
        receipt["port"],
        receipt["source_sha"],
        receipt["ssh_alias"],
    ) != (
        config.policy_profile.name,
        config.policy_backend,
        config.policy_port,
        candidate,
        config.ssh_alias,
    ):
        raise RemoteError("the running-server receipt does not match the route-check configuration")
    target = RemoteTarget(str(receipt["route"]), str(receipt["wsl_distro"]))
    facts = _markers(
        session.run_wsl(
            target,
            _remote_script_command(
                config,
                "check_policy_server.sh",
                [config.policy_profile.name, config.policy_host, str(config.policy_port), candidate],
            ),
            timeout=config.ssh_connect_timeout_seconds + 15,
            label="wsl-route-check",
        )
    )
    if facts.get("SERVER") != "ready":
        raise RemoteError("the owned WSL loopback server is not ready")
    if target.route == "bash":
        print("SSH lands directly in WSL; loopback policy routing passed.")
        return

    script = f"""
$ErrorActionPreference = 'Stop'
$response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:{config.policy_port}/healthz' -TimeoutSec {config.ssh_connect_timeout_seconds}
if ($response.StatusCode -ne 200 -or $response.Content.Trim() -ne 'OK') {{ exit 1 }}
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort {config.policy_port} -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty LocalAddress -Unique)
$listenerJson = ConvertTo-Json -Compress -InputObject $listeners
$listenerBytes = [Text.Encoding]::UTF8.GetBytes($listenerJson)
$listenerEncoded = [Convert]::ToBase64String($listenerBytes)
Write-Output '__ALOHA_WINDOWS_WSL_ROUTE__=ready'
Write-Output "__ALOHA_WINDOWS_LISTENERS__=$listenerEncoded"
""".strip()
    _, output = session.ssh(
        powershell_command(script),
        timeout=config.ssh_connect_timeout_seconds + 15,
        label="windows-wsl-route",
    )
    markers = _markers(output)
    encoded_listeners = markers.get("WINDOWS_LISTENERS", "")
    try:
        listeners = json.loads(base64.b64decode(encoded_listeners, validate=True).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteError("Windows listener evidence is unreadable") from error
    if markers.get("WINDOWS_WSL_ROUTE") != "ready" or not windows_listener_addresses_are_private(listeners):
        raise RemoteError("Windows loopback cannot safely reach the WSL policy server")
    print("Windows loopback reaches the owned WSL policy server without a wildcard listener.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded remote policy operations through the private robot-gpu alias."
    )
    parser.add_argument("command", choices=("doctor", "setup", "convert", "server", "stop", "smoke", "route"))
    args = parser.parse_args()
    session = RemoteSession(load_remote_config())
    try:
        if args.command == "doctor":
            doctor(session)
        elif args.command == "setup":
            setup(session)
        elif args.command == "convert":
            convert(session)
        elif args.command == "server":
            server(session)
        elif args.command == "stop":
            stop(session)
        elif args.command == "route":
            route(session)
        else:
            smoke(session)
    except RemoteError as error:
        parser.exit(1, f"{error}\nIgnored raw evidence: {session.evidence_dir}\n")


if __name__ == "__main__":
    main()
