from pathlib import Path

import pytest

from tools.remote_aloha.config import DEFAULT_TASK
from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import load_remote_config


def test_defaults_when_env_file_is_missing(tmp_path: Path) -> None:
    config = load_mac_sim_config(tmp_path / "missing", {})
    assert (config.task, config.seed, config.episodes, config.output_dir) == (DEFAULT_TASK, 0, 3, Path("outputs"))


def test_file_values_and_environment_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALOHA_SEED=4\nALOHA_EPISODES='2'\nRUN_OUTPUT_DIR=from-file\n", encoding="utf-8")
    config = load_mac_sim_config(env_file, {"RUN_OUTPUT_DIR": "from-environment"})
    assert (config.seed, config.episodes, config.output_dir) == (4, 2, Path("from-environment"))


@pytest.mark.parametrize(
    ("contents", "environment"),
    [
        ("ALOHA_SEED=-1\n", {}),
        ("ALOHA_SEED=nan\n", {}),
        ("ALOHA_SEED=4294967296\n", {}),
        ("ALOHA_EPISODES=0\n", {}),
        ("ALOHA_SEED=4294967295\nALOHA_EPISODES=2\n", {}),
        ("ALOHA_TASK=other\n", {}),
        ("RUN_OUTPUT_DIR=\n", {}),
        ("ALOHA_SEED=1 trailing\n", {}),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, contents: str, environment: dict[str, str]) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="must|nonempty|exceeds"):
        load_mac_sim_config(env_file, environment)


def test_env_file_is_data_not_shell(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("export ALOHA_SEED=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid key"):
        load_mac_sim_config(env_file, {})


def test_remote_defaults_and_profile_contract(tmp_path: Path) -> None:
    config = load_remote_config(tmp_path / "missing", {})
    assert (
        config.ssh_alias,
        config.remote_dir,
        config.wsl_distro,
        config.data_home,
        config.jax_mem_fraction,
        config.policy_host,
        config.policy_port,
        config.policy_profile.name,
        config.policy_backend,
        config.conversion_restore_mode,
        config.min_free_gib,
    ) == (
        "robot-gpu",
        "~/src/openpi",
        "",
        "",
        "0.90",
        "127.0.0.1",
        8000,
        "pi0_aloha_sim",
        "jax",
        "auto",
        40,
    )
    assert set(POLICY_PROFILES) == {"pi0_aloha_sim", "pi05_aloha_base"}
    assert (
        POLICY_PROFILES["pi0_aloha_sim"].env,
        POLICY_PROFILES["pi0_aloha_sim"].config_name,
        POLICY_PROFILES["pi0_aloha_sim"].checkpoint_uri,
        POLICY_PROFILES["pi0_aloha_sim"].experimental,
    ) == ("ALOHA_SIM", "pi0_aloha_sim", "gs://openpi-assets/checkpoints/pi0_aloha_sim", False)
    assert (
        POLICY_PROFILES["pi05_aloha_base"].env,
        POLICY_PROFILES["pi05_aloha_base"].config_name,
        POLICY_PROFILES["pi05_aloha_base"].checkpoint_uri,
        POLICY_PROFILES["pi05_aloha_base"].default_prompt,
        POLICY_PROFILES["pi05_aloha_base"].experimental,
    ) == ("ALOHA", "pi05_aloha", "gs://openpi-assets/checkpoints/pi05_base", "Transfer cube", True)
    assert {profile.action_horizon for profile in POLICY_PROFILES.values()} == {50}
    assert {profile.action_dimension for profile in POLICY_PROFILES.values()} == {14}


def test_remote_file_values_and_environment_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENPI_POLICY_PROFILE=pi05_aloha_base\nREMOTE_POLICY_PORT=9000\n"
        "OPENPI_WSL_DISTRO='Ubuntu Dev'\nOPENPI_JAX_MEM_FRACTION=0.85\nOPENPI_POLICY_BACKEND=pytorch\n"
        "OPENPI_CONVERSION_RESTORE_MODE=partial-bfloat16\n",
        encoding="utf-8",
    )
    config = load_remote_config(env_file, {"REMOTE_POLICY_PORT": "8123", "OPENPI_DATA_HOME": "/srv/open pi's"})
    assert config.policy_profile.experimental
    assert config.policy_port == 8123
    assert config.wsl_distro == "Ubuntu Dev"
    assert config.data_home == "/srv/open pi's"
    assert config.jax_mem_fraction == "0.85"
    assert config.policy_backend == "pytorch"
    assert config.conversion_restore_mode == "partial-bfloat16"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("ROBOT_GPU_SSH_ALIAS", "-oProxyCommand=id", "SSH host alias"),
        ("ROBOT_GPU_SSH_ALIAS", "user@host", "SSH host alias"),
        ("ROBOT_GPU_SSH_ALIAS", "bad host", "SSH host alias"),
        ("ROBOT_GPU_SSH_ALIAS", "bad\nhost", "SSH host alias"),
        ("OPENPI_WSL_DISTRO", "-Ubuntu", "distro"),
        ("OPENPI_WSL_DISTRO", "Ubuntu\nOther", "distro"),
        ("OPENPI_REMOTE_DIR", "relative/path", "absolute WSL path"),
        ("OPENPI_REMOTE_DIR", "~user/path", "absolute WSL path"),
        ("OPENPI_REMOTE_DIR", "$HOME/path", "absolute WSL path"),
        ("OPENPI_REMOTE_DIR", "/srv/$(id)", "absolute WSL path"),
        ("OPENPI_REMOTE_DIR", "/", "must not target"),
        ("OPENPI_REMOTE_DIR", "~/src/../other", "must not target"),
        ("OPENPI_DATA_HOME", "~/cache", "absolute WSL path"),
        ("OPENPI_DATA_HOME", "/proc/cache", "must not target"),
        ("OPENPI_DATA_HOME", "/sys", "must not target"),
        ("REMOTE_POLICY_HOST", "0.0.0.0", "literal loopback"),
        ("REMOTE_POLICY_PORT", "0", "between"),
        ("REMOTE_POLICY_PORT", "65536", "at most"),
        ("REMOTE_POLICY_PORT", "8000 trailing", "unsigned"),
        ("OPENPI_POLICY_PROFILE", "pi0_aloha_sim;id", "must be one of"),
        ("OPENPI_POLICY_BACKEND", "tensorflow", "must be one of"),
        ("OPENPI_CONVERSION_RESTORE_MODE", "partial;id", "must be one of"),
        ("OPENPI_JAX_MEM_FRACTION", "0.9", "must be one of"),
        ("OPENPI_JAX_MEM_FRACTION", "1.00", "must be one of"),
        ("OPENPI_JAX_MEM_FRACTION", "0.90;id", "must be one of"),
        ("SSH_CONNECT_TIMEOUT_SECONDS", "0", "positive"),
        ("SSH_CONNECT_TIMEOUT_SECONDS", "1.5", "unsigned"),
        ("OPENPI_SERVER_STARTUP_TIMEOUT_SECONDS", "7201", "at most"),
        ("OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS", "0", "positive"),
        ("OPENPI_MIN_FREE_GIB", "0", "positive"),
        ("OPENPI_MIN_FREE_GIB", "1025", "at most"),
    ],
)
def test_invalid_remote_configuration_is_rejected(tmp_path: Path, key: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_remote_config(tmp_path / "missing", {key: value})
