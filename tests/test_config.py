from pathlib import Path

import pytest

from tools.remote_aloha.config import DEFAULT_TASK
from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.scenarios import SCENARIOS


def test_defaults_when_env_file_is_missing(tmp_path: Path) -> None:
    config = load_mac_sim_config(tmp_path / "missing", {})
    assert (
        config.task,
        config.scenario.key,
        config.display,
        config.seed,
        config.episodes,
        config.episode_steps,
        config.action_horizon,
        config.prefetch_steps,
        config.prompt_schedule,
        config.output_dir,
    ) == (DEFAULT_TASK, "transfer_cube", False, 0, 3, 300, 30, 25, "fixed", Path("outputs"))


def test_file_values_and_environment_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALOHA_SEED=4\nALOHA_EPISODES='2'\nRUN_OUTPUT_DIR=from-file\n", encoding="utf-8")
    config = load_mac_sim_config(env_file, {"RUN_OUTPUT_DIR": "from-environment"})
    assert (config.seed, config.episodes, config.output_dir) == (4, 2, Path("from-environment"))


def test_fixed_scenario_resolves_task_prompt_and_display(tmp_path: Path) -> None:
    config = load_mac_sim_config(
        tmp_path / "missing",
        {"ALOHA_SCENARIO": "push_letters_single", "ALOHA_DISPLAY": "1"},
    )
    assert config.scenario is SCENARIOS["push_letters_single"]
    assert config.task == "pi_robotics/PushLettersSingleArm-v0"
    assert config.scenario.prompt == "Using only the left arm, push the P and I blocks onto their matching targets."
    assert config.display is True


def test_episode_step_override_is_bounded(tmp_path: Path) -> None:
    config = load_mac_sim_config(tmp_path / "missing", {"ALOHA_EPISODE_STEPS": "6000"})
    assert config.episode_steps == 6000


def test_staged_prompt_schedule_is_one_exact_scenario_one_episode_diagnostic(tmp_path: Path) -> None:
    config = load_mac_sim_config(
        tmp_path / "missing",
        {
            "ALOHA_SCENARIO": "push_pi_single",
            "ALOHA_EPISODES": "1",
            "ALOHA_EPISODE_STEPS": "6000",
            "ALOHA_PROMPT_SCHEDULE": "push_pi_single_left_staged_v1",
        },
    )
    assert config.prompt_schedule == "push_pi_single_left_staged_v1"


@pytest.mark.parametrize(
    ("contents", "environment"),
    [
        ("ALOHA_SEED=-1\n", {}),
        ("ALOHA_SEED=nan\n", {}),
        ("ALOHA_SEED=4294967296\n", {}),
        ("ALOHA_EPISODES=0\n", {}),
        ("ALOHA_EPISODE_STEPS=0\n", {}),
        ("ALOHA_EPISODE_STEPS=6001\n", {}),
        ("ALOHA_SEED=4294967295\nALOHA_EPISODES=2\n", {}),
        ("ALOHA_TASK=other\n", {}),
        ("ALOHA_SCENARIO=\n", {}),
        ("ALOHA_SCENARIO=push_letters_single;id\n", {}),
        ("ALOHA_DISPLAY=true\n", {}),
        ("ALOHA_DISPLAY=2\n", {}),
        ("ALOHA_ACTION_HORIZON=10\nALOHA_PREFETCH_STEPS=10\n", {}),
        ("ALOHA_ACTION_HORIZON=51\n", {}),
        ("ALOHA_PREFETCH_STEPS=0\n", {}),
        ("ALOHA_PROMPT_SCHEDULE=custom\n", {}),
        ("ALOHA_PROMPT_SCHEDULE=push_pi_single_left_staged_v1\n", {}),
        (
            "ALOHA_SCENARIO=push_pi_dual\nALOHA_EPISODES=1\nALOHA_EPISODE_STEPS=6000\n"
            "ALOHA_PROMPT_SCHEDULE=push_pi_single_left_staged_v1\n",
            {},
        ),
        (
            "ALOHA_SCENARIO=push_pi_single\nALOHA_EPISODES=2\nALOHA_EPISODE_STEPS=6000\n"
            "ALOHA_PROMPT_SCHEDULE=push_pi_single_left_staged_v1\n",
            {},
        ),
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
        config.local_policy_host,
        config.local_policy_port,
        config.policy_host,
        config.policy_port,
        config.policy_profile.name,
        config.policy_backend,
        config.conversion_restore_mode,
        config.policy_connect_timeout_seconds,
        config.policy_metadata_timeout_seconds,
        config.policy_inference_timeout_seconds,
        config.policy_close_timeout_seconds,
        config.policy_retry_count,
        config.policy_retry_backoff_seconds,
        config.gpu_metrics_interval_seconds,
        config.min_free_gib,
    ) == (
        "robot-gpu",
        "~/src/openpi",
        "",
        "",
        "0.90",
        "127.0.0.1",
        8000,
        "127.0.0.1",
        8000,
        "pi0_aloha_sim",
        "pytorch",
        "auto",
        60,
        30,
        300,
        10,
        2,
        2.0,
        1.0,
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
        "OPENPI_CONVERSION_RESTORE_MODE=partial-bfloat16\nLOCAL_POLICY_PORT=9001\n"
        "OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS=61\nOPENPI_POLICY_METADATA_TIMEOUT_SECONDS=31\n"
        "OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS=11\n"
        "OPENPI_POLICY_RETRY_COUNT=3\nOPENPI_POLICY_RETRY_BACKOFF_SECONDS=0.5\n"
        "GPU_METRICS_INTERVAL_SECONDS=1.5\n",
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
    assert config.local_policy_port == 9001
    assert (
        config.policy_connect_timeout_seconds,
        config.policy_metadata_timeout_seconds,
        config.policy_close_timeout_seconds,
        config.policy_retry_count,
        config.policy_retry_backoff_seconds,
        config.gpu_metrics_interval_seconds,
    ) == (61, 31, 11, 3, 0.5, 1.5)


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
        ("LOCAL_POLICY_HOST", "localhost", "literal loopback"),
        ("LOCAL_POLICY_PORT", "0", "between"),
        ("LOCAL_POLICY_PORT", "65536", "at most"),
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
        ("OPENPI_POLICY_CONNECT_TIMEOUT_SECONDS", "0", "positive"),
        ("OPENPI_POLICY_METADATA_TIMEOUT_SECONDS", "301", "at most"),
        ("OPENPI_POLICY_INFERENCE_TIMEOUT_SECONDS", "0", "positive"),
        ("OPENPI_POLICY_CLOSE_TIMEOUT_SECONDS", "0", "positive"),
        ("OPENPI_POLICY_RETRY_COUNT", "11", "at most"),
        ("OPENPI_POLICY_RETRY_BACKOFF_SECONDS", "nan", "finite positive"),
        ("OPENPI_POLICY_RETRY_BACKOFF_SECONDS", "61", "at most"),
        ("GPU_METRICS_INTERVAL_SECONDS", "0", "finite positive"),
        ("GPU_METRICS_INTERVAL_SECONDS", "0.01", "at least"),
        ("GPU_METRICS_INTERVAL_SECONDS", "61", "at most"),
        ("OPENPI_MIN_FREE_GIB", "0", "positive"),
        ("OPENPI_MIN_FREE_GIB", "1025", "at most"),
    ],
)
def test_invalid_remote_configuration_is_rejected(tmp_path: Path, key: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_remote_config(tmp_path / "missing", {key: value})
