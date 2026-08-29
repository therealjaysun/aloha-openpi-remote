import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gym_aloha")
pytest.importorskip("dm_control")

import examples.aloha_sim.push_pi_env  # noqa: E402,F401
from tools.remote_aloha.run import _convert_environment_observation  # noqa: E402
from tools.remote_aloha.run import _scenario_info_fields  # noqa: E402
from tools.remote_aloha.scenarios import CUSTOM_SCENARIOS  # noqa: E402
from tools.remote_aloha.scenarios import SCENARIOS  # noqa: E402
from tools.remote_aloha.scenarios import TABLETOP_SHA256  # noqa: E402
from tools.remote_aloha.scenarios import descriptor_sha256  # noqa: E402
from tools.remote_aloha.scenarios import effective_layout_seed  # noqa: E402
from tools.remote_aloha.scenarios import sample_layout  # noqa: E402


def _assert_policy_views(environment, raw: dict, prompt: str) -> None:
    images = _convert_environment_observation(environment, raw, prompt)["images"]
    assert tuple(images) == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    assert all(image.shape == (3, 224, 224) and image.dtype == np.uint8 for image in images.values())
    assert all(np.ptp(image) > 0 for image in images.values())
    assert not np.array_equal(images["cam_high"], images["cam_left_wrist"])
    assert not np.array_equal(images["cam_left_wrist"], images["cam_right_wrist"])


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
        model = environment.unwrapped._env.physics.model  # noqa: SLF001
        camera_names = {model.id2name(index, "camera") for index in range(model.ncam)}
        assert {"top", "left_wrist", "right_wrist"} <= camera_names
        _assert_policy_views(environment, observation, spec.prompt)
        assert info["scenario"] == scenario
        assert info["is_success"] is False
        assert info["held_steps"] == 0
        assert len(info["scene_hash"]) == len(info["layout_hash"]) == 64
        assert info["descriptor_sha256"] == descriptor_sha256()
        assert _scenario_info_fields(spec) <= set(info)
        assert all(
            np.isfinite(info[key])
            for key in _scenario_info_fields(spec)
            if key.startswith("body_") and isinstance(info[key], float)
        )
        assert all(
            len(row) == 8 and isinstance(row[0], str) and np.isfinite(row[1:]).all()
            for row in info["sampled_poses"] + info["settled_poses"]
        )

        command = environment.unwrapped.home_joint_positions
        observation, reward, terminated, truncated, info = environment.step(command)
        assert np.isfinite(observation["agent_pos"]).all()
        assert reward in {0.0, 1.0}
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info["is_success"], bool)
        _assert_policy_views(environment, observation, spec.prompt)
    finally:
        environment.close()


def test_stock_policy_wrist_renders_reset_and_step() -> None:
    environment = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    try:
        raw, _ = environment.reset(seed=0)
        _assert_policy_views(environment, raw, "Transfer cube")
        raw, *_ = environment.step(np.zeros(14, dtype=np.float64))
        _assert_policy_views(environment, raw, "Transfer cube")
    finally:
        environment.close()


def test_custom_episode_limit_is_applied() -> None:
    environment = gym.make(
        SCENARIOS["push_pi_single"].gym_id,
        obs_type="pixels_agent_pos",
        episode_steps=2,
        max_episode_steps=2,
    )
    try:
        environment.reset(seed=0)
        command = environment.unwrapped.home_joint_positions
        _, _, terminated, truncated, _ = environment.step(command)
        assert not terminated
        assert not truncated
        _, _, _, truncated, info = environment.step(command)
        assert truncated
        assert info["terminal_reason"] == "time_limit"
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


@pytest.mark.parametrize(
    ("scenario", "expected_masses"),
    [
        ("push_pi_single", {"pi": 0.05544}),
        ("push_letters_single", {"P": 0.05544, "I": 0.04368}),
    ],
)
def test_visual_geoms_do_not_duplicate_body_mass(scenario: str, expected_masses: dict[str, float]) -> None:
    environment = gym.make(SCENARIOS[scenario].gym_id, obs_type="pixels_agent_pos")
    try:
        environment.reset(seed=0)
        model = environment.unwrapped._env.physics.model  # noqa: SLF001
        for body, expected in expected_masses.items():
            body_id = model.name2id(f"push_pi/{body}", "body")
            assert model.body_mass[body_id] == pytest.approx(expected)
    finally:
        environment.close()


def test_reset_retries_deterministically_and_reports_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = gym.make(SCENARIOS["push_pi_single"].gym_id, obs_type="pixels_agent_pos")
    try:
        task = environment.unwrapped._push_task  # noqa: SLF001
        original = task.capture_reset
        calls = 0

        def fail_once(physics: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("synthetic unstable layout")
            original(physics)

        monkeypatch.setattr(task, "capture_reset", fail_once)
        _, info = environment.reset(seed=7)
        assert calls == 2
        expected_seed = effective_layout_seed(7, 1)
        assert info["sampled_poses"] == [[pose.name, *pose.vector()] for pose in sample_layout("pi", expected_seed)]
        assert info["layout_provenance"] == {"requested_seed": 7, "effective_seed": expected_seed, "attempt": 1}

        attempted = []

        def always_fail(_physics: object) -> None:
            attempted.append(task.sampled)
            raise ValueError("synthetic unstable layout")

        monkeypatch.setattr(task, "capture_reset", always_fail)
        with pytest.raises(ValueError, match="exhausted 8 layouts"):
            environment.reset(seed=7)
        assert attempted == [sample_layout("pi", effective_layout_seed(7, attempt)) for attempt in range(8)]
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
