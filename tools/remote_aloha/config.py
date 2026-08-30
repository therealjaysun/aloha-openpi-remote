from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess

from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import ScenarioSpec
from tools.remote_aloha.scenarios import get_scenario

DEFAULT_TASK = "gym_aloha/AlohaTransferCube-v0"
DEFAULT_POLICY_PROFILE = "pi05_aloha_base"
DEFAULT_POLICY_BACKEND = "pytorch"
DEFAULT_CONVERSION_RESTORE_MODE = "auto"
LOOPBACK_HOST = "127.0.0.1"
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UINT = re.compile(r"[0-9]+\Z")
_SSH_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_JAX_MEM_FRACTIONS = ("0.75", "0.80", "0.85", "0.90", "0.95")


@dataclass(frozen=True)
class MacSimConfig:
    task: str = DEFAULT_TASK
    scenario: ScenarioSpec = SCENARIOS["transfer_cube"]
    display: bool = False
    seed: int = 0
    episodes: int = 3
    episode_steps: int = 300
    action_horizon: int = 30
    prefetch_steps: int = 25
    chunk_crossfade_steps: int = 0
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    env: str
    config_name: str
    checkpoint_uri: str
    checkpoint_label: str
    default_prompt: str | None
    experimental: bool
    action_horizon: int = 50
    action_dimension: int = 14


POLICY_PROFILES = {
    "pi0_aloha_sim": PolicyProfile(
        name="pi0_aloha_sim",
        env="ALOHA_SIM",
        config_name="pi0_aloha_sim",
        checkpoint_uri="gs://openpi-assets/checkpoints/pi0_aloha_sim",
        checkpoint_label="pi0_aloha_sim",
        default_prompt=None,
        experimental=False,
    ),
    "pi05_aloha_base": PolicyProfile(
        name="pi05_aloha_base",
        env="ALOHA",
        config_name="pi05_aloha",
        checkpoint_uri="gs://openpi-assets/checkpoints/pi05_base",
        checkpoint_label="pi05_base",
        default_prompt="Transfer cube",
        experimental=True,
    ),
}


@dataclass(frozen=True)
class RemoteConfig:
    ssh_alias: str = "robot-gpu"
    remote_dir: str = "~/src/openpi"
    wsl_distro: str = ""
    data_home: str = ""
    jax_mem_fraction: str = "0.90"
    local_policy_host: str = LOOPBACK_HOST
    local_policy_port: int = 8000
    policy_host: str = LOOPBACK_HOST
    policy_port: int = 8000
    policy_profile: PolicyProfile = POLICY_PROFILES[DEFAULT_POLICY_PROFILE]
    policy_backend: str = DEFAULT_POLICY_BACKEND
    conversion_restore_mode: str = DEFAULT_CONVERSION_RESTORE_MODE
    ssh_connect_timeout_seconds: int = 10
    policy_connect_timeout_seconds: int = 60
    policy_metadata_timeout_seconds: int = 30
    server_startup_timeout_seconds: int = 1800
    policy_inference_timeout_seconds: int = 300
    policy_close_timeout_seconds: int = 10
    policy_retry_count: int = 2
    policy_retry_backoff_seconds: float = 2.0
    gpu_metrics_interval_seconds: float = 1.0
    min_free_gib: int = 40


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if not _KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid key {key!r}")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"{path}:{line_number}: unmatched quote")
            value = value[1:-1]
        values[key] = value
    return values


def _uint(name: str, value: str, *, maximum: int | None = None) -> int:
    if not _UINT.fullmatch(value):
        raise ValueError(f"{name} must be an unsigned decimal integer")
    parsed = int(value)
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return parsed


def _positive_float(name: str, value: str, *, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return parsed


def _remote_path(name: str, value: str, *, allow_empty: bool = False, allow_tilde: bool = False) -> str:
    if allow_empty and value == "":
        return value
    has_expansion = "$" in value or "`" in value
    allowed_prefix = value.startswith("/") or (allow_tilde and value.startswith("~/"))
    if any(ord(character) < 32 for character in value) or has_expansion or not allowed_prefix:
        expected = (
            "an absolute WSL path or a path beginning with tilde-slash" if allow_tilde else "an absolute WSL path"
        )
        raise ValueError(f"{name} must be {expected}; expansion syntax and other relative paths are rejected")
    normalized = value[2:] if value.startswith("~/") else value
    parts = PurePosixPath(normalized).parts
    if (
        ".." in parts
        or value == "/"
        or any(value == root or value.startswith(f"{root}/") for root in ("/dev", "/proc", "/sys"))
    ):
        raise ValueError(f"{name} must not target root, pseudo-filesystems, or parent traversal")
    return value


def get_policy_profile(name: str) -> PolicyProfile:
    try:
        return POLICY_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(POLICY_PROFILES)
        raise ValueError(f"OPENPI_POLICY_PROFILE must be one of: {choices}") from error


def validate_output_root(path: Path) -> Path:
    repository = Path.cwd().resolve()
    resolved = path.resolve()
    if resolved == repository or repository in resolved.parents:
        relative = resolved.relative_to(repository)
        ignored = subprocess.run(["git", "check-ignore", "--quiet", "--", str(relative)], timeout=10, check=False)
        if ignored.returncode:
            raise ValueError("RUN_OUTPUT_DIR inside the repository must be ignored by Git")
    elif not path.is_absolute():
        raise ValueError("RUN_OUTPUT_DIR outside the repository must be absolute")
    return path


def load_mac_sim_config(env_file: str | Path = ".env", environ: Mapping[str, str] | None = None) -> MacSimConfig:
    values = _read_env_file(Path(env_file))
    source = os.environ if environ is None else environ
    for key in (
        "ALOHA_SCENARIO",
        "ALOHA_DISPLAY",
        "ALOHA_SEED",
        "ALOHA_EPISODES",
        "ALOHA_EPISODE_STEPS",
        "ALOHA_ACTION_HORIZON",
        "ALOHA_PREFETCH_STEPS",
        "ALOHA_CHUNK_CROSSFADE_STEPS",
        "RUN_OUTPUT_DIR",
    ):
        if key in source:
            values[key] = source[key]

    scenario = get_scenario(values.get("ALOHA_SCENARIO", "transfer_cube"))
    display_value = values.get("ALOHA_DISPLAY", "0")
    if display_value not in {"0", "1"}:
        raise ValueError("ALOHA_DISPLAY must be exactly 0 or 1")
    seed = _uint("ALOHA_SEED", values.get("ALOHA_SEED", "0"), maximum=2**32 - 1)
    episodes = _uint("ALOHA_EPISODES", values.get("ALOHA_EPISODES", "3"))
    if episodes < 1:
        raise ValueError("ALOHA_EPISODES must be positive")
    if seed + episodes - 1 > 2**32 - 1:
        raise ValueError("ALOHA_SEED + ALOHA_EPISODES exceeds the seed range")
    episode_steps = _uint("ALOHA_EPISODE_STEPS", values.get("ALOHA_EPISODE_STEPS", "300"), maximum=6000)
    if episode_steps < 1:
        raise ValueError("ALOHA_EPISODE_STEPS must be positive")
    action_horizon = _uint("ALOHA_ACTION_HORIZON", values.get("ALOHA_ACTION_HORIZON", "30"), maximum=50)
    prefetch_steps = _uint("ALOHA_PREFETCH_STEPS", values.get("ALOHA_PREFETCH_STEPS", "25"), maximum=50)
    if not 1 <= prefetch_steps < action_horizon:
        raise ValueError("ALOHA buffering must satisfy 1 <= ALOHA_PREFETCH_STEPS < ALOHA_ACTION_HORIZON <= 50")
    chunk_crossfade_steps = _uint("ALOHA_CHUNK_CROSSFADE_STEPS", values.get("ALOHA_CHUNK_CROSSFADE_STEPS", "0"))
    if chunk_crossfade_steps not in {0, 5}:
        raise ValueError("ALOHA_CHUNK_CROSSFADE_STEPS must be exactly 0 or 5")
    output = values.get("RUN_OUTPUT_DIR", "outputs")
    if not output or "\x00" in output:
        raise ValueError("RUN_OUTPUT_DIR must be a nonempty path")
    return MacSimConfig(
        task=scenario.gym_id,
        scenario=scenario,
        display=display_value == "1",
        seed=seed,
        episodes=episodes,
        episode_steps=episode_steps,
        action_horizon=action_horizon,
        prefetch_steps=prefetch_steps,
        chunk_crossfade_steps=chunk_crossfade_steps,
        output_dir=Path(output),
    )


def load_remote_config(env_file: str | Path = ".env", environ: Mapping[str, str] | None = None) -> RemoteConfig:
    values = _read_env_file(Path(env_file))
    source = os.environ if environ is None else environ
    keys = (
        "ROBOT_GPU_SSH_ALIAS",
        "OPENPI_REMOTE_DIR",
        "OPENPI_WSL_DISTRO",
        "OPENPI_DATA_HOME",
        "OPENPI_JAX_MEM_FRACTION",
        "LOCAL_POLICY_PORT",
        "REMOTE_POLICY_PORT",
        "OPENPI_POLICY_PROFILE",
        "OPENPI_POLICY_BACKEND",
        "OPENPI_CONVERSION_RESTORE_MODE",
        "SSH_CONNECT_TIMEOUT_SECONDS",
        "OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS",
        "OPENPI_POLICY_METADATA_TIMEOUT_SECONDS",
        "OPENPI_SERVER_STARTUP_TIMEOUT_SECONDS",
        "OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS",
        "OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS",
        "OPENPI_POLICY_RETRY_COUNT",
        "OPENPI_POLICY_RETRY_BACKOFF_SECONDS",
        "GPU_METRICS_INTERVAL_SECONDS",
        "OPENPI_MIN_FREE_GIB",
    )
    for key in keys:
        if key in source:
            values[key] = source[key]

    alias = values.get("ROBOT_GPU_SSH_ALIAS", "robot-gpu")
    if not _SSH_ALIAS.fullmatch(alias):
        raise ValueError("ROBOT_GPU_SSH_ALIAS must be one safe SSH host alias")
    distro = values.get("OPENPI_WSL_DISTRO", "")
    if distro and (distro.startswith("-") or any(ord(character) < 32 for character in distro)):
        raise ValueError("OPENPI_WSL_DISTRO must be one printable distro name and cannot start with '-'")
    jax_mem_fraction = values.get("OPENPI_JAX_MEM_FRACTION", "0.90")
    if jax_mem_fraction not in _JAX_MEM_FRACTIONS:
        choices = ", ".join(_JAX_MEM_FRACTIONS)
        raise ValueError(f"OPENPI_JAX_MEM_FRACTION must be one of: {choices}")
    local_port = _uint("LOCAL_POLICY_PORT", values.get("LOCAL_POLICY_PORT", "8000"), maximum=65535)
    port = _uint("REMOTE_POLICY_PORT", values.get("REMOTE_POLICY_PORT", "8000"), maximum=65535)
    if min(local_port, port) < 1:
        raise ValueError("policy ports must be between 1 and 65535")
    connect_timeout = _uint("SSH_CONNECT_TIMEOUT_SECONDS", values.get("SSH_CONNECT_TIMEOUT_SECONDS", "10"), maximum=300)
    policy_connect_timeout = _uint(
        "OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS",
        values.get("OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS", "60"),
        maximum=1800,
    )
    metadata_timeout = _uint(
        "OPENPI_POLICY_METADATA_TIMEOUT_SECONDS",
        values.get("OPENPI_POLICY_METADATA_TIMEOUT_SECONDS", "30"),
        maximum=300,
    )
    startup_timeout = _uint(
        "OPENPI_SERVER_STARTUP_TIMEOUT_SECONDS",
        values.get("OPENPI_SERVER_STARTUP_TIMEOUT_SECONDS", "1800"),
        maximum=7200,
    )
    inference_timeout = _uint(
        "OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS",
        values.get("OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS", "300"),
        maximum=1800,
    )
    close_timeout = _uint(
        "OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS",
        values.get("OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS", "10"),
        maximum=300,
    )
    retry_count = _uint("OPENPI_POLICY_RETRY_COUNT", values.get("OPENPI_POLICY_RETRY_COUNT", "2"), maximum=10)
    retry_backoff = _positive_float(
        "OPENPI_POLICY_RETRY_BACKOFF_SECONDS",
        values.get("OPENPI_POLICY_RETRY_BACKOFF_SECONDS", "2"),
        maximum=60,
    )
    gpu_metrics_interval = _positive_float(
        "GPU_METRICS_INTERVAL_SECONDS",
        values.get("GPU_METRICS_INTERVAL_SECONDS", "1"),
        maximum=60,
    )
    if gpu_metrics_interval < 0.1:
        raise ValueError("GPU_METRICS_INTERVAL_SECONDS must be at least 0.1")
    min_free_gib = _uint("OPENPI_MIN_FREE_GIB", values.get("OPENPI_MIN_FREE_GIB", "40"), maximum=1024)
    if (
        min(
            connect_timeout,
            policy_connect_timeout,
            metadata_timeout,
            startup_timeout,
            inference_timeout,
            close_timeout,
            min_free_gib,
        )
        < 1
    ):
        raise ValueError("remote timeouts and OPENPI_MIN_FREE_GIB must be positive")

    policy_backend = values.get("OPENPI_POLICY_BACKEND", DEFAULT_POLICY_BACKEND)
    if policy_backend not in {"jax", "pytorch"}:
        raise ValueError("OPENPI_POLICY_BACKEND must be one of: jax, pytorch")

    conversion_restore_mode = values.get("OPENPI_CONVERSION_RESTORE_MODE", DEFAULT_CONVERSION_RESTORE_MODE)
    if conversion_restore_mode not in {"auto", "full-float32", "partial-bfloat16"}:
        raise ValueError("OPENPI_CONVERSION_RESTORE_MODE must be one of: auto, full-float32, partial-bfloat16")

    return RemoteConfig(
        ssh_alias=alias,
        remote_dir=_remote_path("OPENPI_REMOTE_DIR", values.get("OPENPI_REMOTE_DIR", "~/src/openpi"), allow_tilde=True),
        wsl_distro=distro,
        data_home=_remote_path("OPENPI_DATA_HOME", values.get("OPENPI_DATA_HOME", ""), allow_empty=True),
        jax_mem_fraction=jax_mem_fraction,
        local_policy_port=local_port,
        policy_port=port,
        policy_profile=get_policy_profile(values.get("OPENPI_POLICY_PROFILE", DEFAULT_POLICY_PROFILE)),
        policy_backend=policy_backend,
        conversion_restore_mode=conversion_restore_mode,
        ssh_connect_timeout_seconds=connect_timeout,
        policy_connect_timeout_seconds=policy_connect_timeout,
        policy_metadata_timeout_seconds=metadata_timeout,
        server_startup_timeout_seconds=startup_timeout,
        policy_inference_timeout_seconds=inference_timeout,
        policy_close_timeout_seconds=close_timeout,
        policy_retry_count=retry_count,
        policy_retry_backoff_seconds=retry_backoff,
        gpu_metrics_interval_seconds=gpu_metrics_interval,
        min_free_gib=min_free_gib,
    )
