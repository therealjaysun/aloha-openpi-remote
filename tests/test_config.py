from pathlib import Path

import pytest

from tools.remote_aloha.config import DEFAULT_TASK
from tools.remote_aloha.config import load_mac_sim_config


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
