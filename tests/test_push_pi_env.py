import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gym_aloha")
pytest.importorskip("dm_control")

import examples.aloha_sim.push_pi_env  # noqa: E402,F401
from tools.remote_aloha.scenarios import CUSTOM_SCENARIOS  # noqa: E402
from tools.remote_aloha.scenarios import SCENARIOS  # noqa: E402
from tools.remote_aloha.scenarios import TABLETOP_SHA256  # noqa: E402


@pytest.mark.parametrize("scenario", CUSTOM_SCENARIOS)
def test_registered_environment_contract(scenario: str) -> None:
    spec = SCENARIOS[scenario]
    environment = gym.make(spec.gym_id, obs_type="pixels_agent_pos")
    try:
        observation, info = environment.reset(seed=0)
        assert environment.spec.max_episode_steps == 300
        assert environment.metadata["render_fps"] == 50
        assert environment.action_space.shape == (14,)
        assert observation["pixels"]["top"].shape == (480, 640, 3)
        assert observation["pixels"]["top"].dtype == np.uint8
        assert observation["agent_pos"].shape == (14,)
        assert observation["agent_pos"].dtype == np.float64
        assert np.isfinite(observation["agent_pos"]).all()
        assert info["scenario"] == scenario
        assert info["is_success"] is False
        assert len(info["scene_hash"]) == len(info["layout_hash"]) == 64

        command = environment.unwrapped.home_joint_positions
        observation, reward, terminated, truncated, info = environment.step(command)
        assert np.isfinite(observation["agent_pos"]).all()
        assert reward in {0.0, 1.0}
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info["is_success"], bool)
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("single", "dual", "expected_nq", "expected_nv"),
    [
        ("push_pi_single", "push_pi_dual", 23, 22),
        ("push_letters_single", "push_letters_dual", 30, 28),
    ],
)
def test_seeded_single_dual_pair_has_identical_scene_and_layout(
    single: str, dual: str, expected_nq: int, expected_nv: int
) -> None:
    environments = [gym.make(SCENARIOS[key].gym_id, obs_type="pixels_agent_pos") for key in (single, dual)]
    try:
        infos = [environment.reset(seed=2)[1] for environment in environments]
        assert infos[0]["scene_hash"] == infos[1]["scene_hash"]
        assert infos[0]["layout_hash"] == infos[1]["layout_hash"]
        assert infos[0]["sampled_poses"] == infos[1]["sampled_poses"]
        assert infos[0]["settled_poses"] == infos[1]["settled_poses"]
        for environment in environments:
            model = environment.unwrapped._env.physics.model  # noqa: SLF001
            assert (model.nq, model.nv, model.nu) == (expected_nq, expected_nv, 16)
    finally:
        for environment in environments:
            environment.close()


def test_uppercase_i_is_one_dotless_body_with_three_parts() -> None:
    environment = gym.make(SCENARIOS["push_letters_single"].gym_id, obs_type="pixels_agent_pos")
    try:
        environment.reset(seed=0)
        model = environment.unwrapped._env.physics.model  # noqa: SLF001
        names = {model.id2name(index, "geom") for index in range(model.ngeom)}
        assert {"push_pi/I_0", "push_pi/I_1", "push_pi/I_2"} <= names
        assert not any(name and "dot" in name.lower() for name in names)
        assert environment.unwrapped.scene_hash
        assert TABLETOP_SHA256 == "76a1571d1aa36520f2bd81c268991b99816c2a7819464d718e0fd9976fe30dce"
    finally:
        environment.close()


def test_custom_environment_rejects_invalid_action() -> None:
    environment = gym.make(SCENARIOS["push_pi_single"].gym_id, obs_type="pixels_agent_pos")
    try:
        environment.reset(seed=0)
        with pytest.raises(ValueError, match="finite 14-vector"):
            environment.step(np.zeros(13))
        invalid = np.zeros(14)
        invalid[3] = np.nan
        with pytest.raises(ValueError, match="finite 14-vector"):
            environment.step(invalid)
    finally:
        environment.close()
