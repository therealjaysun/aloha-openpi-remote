import copy
import json
import math
from pathlib import Path

import pytest

from tools.remote_aloha.config import POLICY_PROFILES
from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.remote import UPSTREAM_SHA
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.scenario_matrix import run_matrix
from tools.remote_aloha.scenario_matrix import summarize_latest
from tools.remote_aloha.scenario_matrix import validate_matrix
import tools.remote_aloha.scenarios as scenarios
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.scenarios import BodyDescriptor
from tools.remote_aloha.scenarios import BodyState
from tools.remote_aloha.scenarios import OutcomeState
from tools.remote_aloha.scenarios import Part
from tools.remote_aloha.scenarios import Participation
from tools.remote_aloha.scenarios import advance_outcome
from tools.remote_aloha.scenarios import body_descriptors
from tools.remote_aloha.scenarios import descriptor_sha256
from tools.remote_aloha.scenarios import footprint_overlap_coverage
from tools.remote_aloha.scenarios import layout_hash
from tools.remote_aloha.scenarios import sample_layout
from tools.remote_aloha.scenarios import scene_hash
from tools.remote_aloha.scenarios import update_participation


def _state(body, *, x=None, y=None, z=0.012, yaw=None, roll=0.0, com_z=None) -> BodyState:
    yaw = body.target_yaw if yaw is None else yaw
    cosine_roll, sine_roll = math.cos(roll / 2), math.sin(roll / 2)
    cosine_yaw, sine_yaw = math.cos(yaw / 2), math.sin(yaw / 2)
    return BodyState(
        body.name,
        body.target_x if x is None else x,
        body.target_y if y is None else y,
        z,
        cosine_roll * cosine_yaw,
        sine_roll * cosine_yaw,
        -sine_roll * sine_yaw,
        cosine_roll * sine_yaw,
        com_z,
    )


def test_fixed_scenario_contract_uses_uppercase_dotless_letters() -> None:
    assert tuple(SCENARIOS) == (
        "transfer_cube",
        "push_pi_single",
        "push_pi_dual",
        "push_letters_single",
        "push_letters_dual",
    )
    assert {key: SCENARIOS[key].prompt for key in tuple(SCENARIOS)[1:]} == {
        "push_pi_single": (
            "A PI-shaped block and a dotted target outline are on the table. Using only the left arm, push the block "
            "until it covers and aligns with the target outline."
        ),
        "push_pi_dual": (
            "A PI-shaped block and a dotted target outline are on the table. Using both arms, push the block until it "
            "covers and aligns with the target outline."
        ),
        "push_letters_single": (
            "A red P-shaped block, a blue I-shaped block, and their dotted target outlines are on the table. Using "
            "only the left arm, push the red P-shaped block onto the red dotted P outline and the blue I-shaped block "
            "onto the blue dotted I outline, until each block covers and aligns with its outline."
        ),
        "push_letters_dual": (
            "P and I blocks and their dotted target outlines are on the table. Using both arms, push each block until "
            "it covers and aligns with its matching target outline."
        ),
    }
    bodies = body_descriptors("letters")
    assert tuple(body.name for body in bodies) == ("P", "I")
    assert len(bodies[1].parts) == 3
    assert bodies[1].yaw_period == math.pi


def test_task_outcomes_are_recorded_without_ending_the_episode() -> None:
    source = Path("examples/aloha_sim/push_pi_env.py").read_text(encoding="utf-8")
    assert "terminated = False" in source
    assert "truncated = self._step_count >= self._episode_steps" in source


def test_layout_is_deterministic_and_arm_mode_independent() -> None:
    for object_kind in ("pi", "letters"):
        for seed in range(3):
            first = sample_layout(object_kind, seed)
            assert first == sample_layout(object_kind, seed)
            assert layout_hash(first) == layout_hash(sample_layout(object_kind, seed))
    assert SCENARIOS["push_pi_single"].object_kind == SCENARIOS["push_pi_dual"].object_kind
    assert SCENARIOS["push_letters_single"].object_kind == SCENARIOS["push_letters_dual"].object_kind


def test_layout_rejects_bad_seed_and_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="seed"):
        sample_layout("pi", -1)
    monkeypatch.setattr(scenarios, "SPAWN_BOUNDS", (0.0, 0.01, 0.4, 0.41))
    with pytest.raises(ValueError, match="exhausted"):
        sample_layout("pi", 7, max_attempts=1)


def test_success_requires_five_held_steps_and_i_yaw_is_pi_symmetric() -> None:
    bodies = body_descriptors("letters")
    states = {body.name: _state(body, yaw=math.pi if body.name == "I" else 0.0) for body in bodies}
    rest = {body.name: 0.012 for body in bodies}
    outcome = OutcomeState()
    for index in range(5):
        outcome, metrics = advance_outcome(bodies, states, rest, outcome)
        assert outcome.success is (index == 4)
    assert metrics["body_1_yaw_error"] == pytest.approx(0.0)
    moved = {body.name: _state(body, x=0.0, y=0.4) for body in bodies}
    assert advance_outcome(bodies, moved, rest, outcome)[0].success


def test_target_area_coverage_is_exact_for_translation_rotation_and_compound_unions() -> None:
    box = BodyDescriptor("box", (Part(0.0, 0.0, 0.5, 0.5),), 0.0, 0.0, 0.0, (1.0, 0.0, 0.0, 1.0))
    assert footprint_overlap_coverage(box, _state(box)) == pytest.approx(1.0)
    assert footprint_overlap_coverage(box, _state(box, x=0.5)) == pytest.approx(0.5)
    assert footprint_overlap_coverage(box, _state(box, x=2.0)) == pytest.approx(0.0)
    assert footprint_overlap_coverage(box, _state(box, yaw=math.pi / 4)) == pytest.approx(2 * math.sqrt(2) - 2)

    duplicate = BodyDescriptor("duplicate", (box.parts[0], box.parts[0]), 0.0, 0.0, 0.0, box.rgba)
    assert footprint_overlap_coverage(duplicate, _state(duplicate)) == pytest.approx(1.0)
    compound = BodyDescriptor(
        "compound",
        (Part(-0.25, 0.0, 0.5, 0.5), Part(0.25, 0.0, 0.5, 0.5)),
        0.0,
        0.0,
        0.0,
        box.rgba,
    )
    assert footprint_overlap_coverage(compound, _state(compound, x=0.5)) == pytest.approx(2 / 3)


def test_target_area_coverage_rejects_nonfinite_or_degenerate_geometry() -> None:
    box = BodyDescriptor("box", (Part(0.0, 0.0, 0.5, 0.5),), 0.0, 0.0, 0.0, (1.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="state"):
        footprint_overlap_coverage(box, _state(box, x=float("nan")))
    degenerate = BodyDescriptor("box", (Part(0.0, 0.0, 0.0, 0.5),), 0.0, 0.0, 0.0, box.rgba)
    with pytest.raises(ValueError, match="positive"):
        footprint_overlap_coverage(degenerate, _state(degenerate))


def test_letters_reject_one_correct_swapped_lift_fall_and_off_table() -> None:
    bodies = body_descriptors("letters")
    rest = {body.name: 0.012 for body in bodies}
    one_correct = {"P": _state(bodies[0]), "I": _state(bodies[1], x=0.0, y=0.4)}
    outcome, _ = advance_outcome(bodies, one_correct, rest)
    assert not outcome.success
    assert outcome.held_steps == 0

    swapped = {
        "P": _state(bodies[0], x=bodies[1].target_x, y=bodies[1].target_y),
        "I": _state(bodies[1], x=bodies[0].target_x, y=bodies[0].target_y),
    }
    assert advance_outcome(bodies, swapped, rest)[0].held_steps == 0

    lifted = {body.name: _state(body, com_z=0.023) for body in bodies}
    lifted_outcome, _ = advance_outcome(bodies, lifted, rest)
    assert lifted_outcome.lifted_ever
    assert not lifted_outcome.success
    exact = {body.name: _state(body) for body in bodies}
    for _ in range(6):
        lifted_outcome, _ = advance_outcome(bodies, exact, rest, lifted_outcome)
    assert not lifted_outcome.success

    fallen = {"P": _state(bodies[0], roll=math.radians(31)), "I": _state(bodies[1])}
    fallen_outcome, _ = advance_outcome(bodies, fallen, rest)
    assert fallen_outcome.terminal_reason == "fallen"
    assert advance_outcome(bodies, exact, rest, fallen_outcome)[0].fallen
    outside = {"P": _state(bodies[0], x=scenarios.TABLE_BOUNDS[1]), "I": _state(bodies[1])}
    outside_outcome, _ = advance_outcome(bodies, outside, rest)
    assert outside_outcome.terminal_reason == "off_table"
    assert advance_outcome(bodies, exact, rest, outside_outcome)[0].off_table


def test_contacts_are_order_independent_and_interference_is_same_step() -> None:
    geom_to_body = {"push_pi/P_0": "P"}
    left = "vx300s_left/10_left_gripper_finger"
    right = "vx300s_right/10_right_gripper_finger"
    state = update_participation([("push_pi/P_0", left)], geom_to_body)
    assert state == Participation(left_contact_ever=True)
    state = update_participation([(right, "push_pi/P_0"), (left, "push_pi/P_0")], geom_to_body, state)
    assert state.both_arms_participated
    assert state.interference_ever


def test_scene_hash_covers_only_physical_xml_assets_and_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    assets = {"a.xml": b"a", "mesh.stl": b"b"}
    baseline = scene_hash(b"<mujoco/>", assets, "pi")
    assert baseline == scene_hash(b"<mujoco/>", dict(reversed(list(assets.items()))), "pi")
    assert baseline != scene_hash(b"<mujoco/>", assets, "letters")
    assert baseline != scene_hash(b"<mujoco changed='1'/>", assets, "pi")
    monkeypatch.setattr(scenarios, "descriptor_sha256", lambda: "execution-metadata-only-change")
    assert baseline == scene_hash(b"<mujoco/>", assets, "pi")


def test_descriptor_hash_freezes_every_calibrated_value() -> None:
    assert descriptor_sha256() == "1aa4f0f1a63adf7bb20f290c0a4021f689dd69f8d8d5cc9b37a4bcd9699e6814"


@pytest.mark.parametrize(
    ("metric", "limit", "over"),
    [
        ("xy", scenarios.SUCCESS_XY_METERS, 1e-9),
        ("yaw", scenarios.SUCCESS_YAW_RADIANS, 1e-9),
        ("roll", scenarios.SUCCESS_TILT_RADIANS, 1e-9),
        ("height", scenarios.SUCCESS_HEIGHT_METERS, 1e-9),
    ],
)
def test_success_thresholds_accept_exact_boundary_and_reject_just_over(metric: str, limit: float, over: float) -> None:
    body = body_descriptors("pi")[0]
    rest = {body.name: 0.012}

    def state(value: float) -> BodyState:
        kwargs = {"x": body.target_x, "y": body.target_y, "yaw": body.target_yaw, "roll": 0.0, "com_z": 0.012}
        if metric == "xy":
            kwargs["x"] += value
        elif metric == "yaw":
            kwargs["yaw"] += value
        elif metric == "roll":
            kwargs["roll"] = value
        else:
            kwargs["com_z"] += value
        return _state(body, **kwargs)

    assert advance_outcome((body,), {body.name: state(limit)}, rest)[0].held_steps == 1
    assert advance_outcome((body,), {body.name: state(limit + over)}, rest)[0].held_steps == 0


def test_lift_and_fall_thresholds_trigger_only_just_over_boundary() -> None:
    body = body_descriptors("pi")[0]
    rest = {body.name: 0.012}
    exact_lift = _state(body, com_z=0.012 + scenarios.LIFT_METERS)
    over_lift = _state(body, com_z=0.012 + scenarios.LIFT_METERS + 1e-9)
    assert advance_outcome((body,), {body.name: exact_lift}, rest)[0].lifted_ever is False
    assert advance_outcome((body,), {body.name: over_lift}, rest)[0].lifted_ever is True
    exact_fall = _state(body, roll=scenarios.FALL_RADIANS)
    over_fall = _state(body, roll=scenarios.FALL_RADIANS + 1e-9)
    assert advance_outcome((body,), {body.name: exact_fall}, rest)[0].fallen is False
    assert advance_outcome((body,), {body.name: over_fall}, rest)[0].fallen is True


def _matrix_info(scenario_key: str, layout: str, *, terminal: bool = False) -> dict[str, object]:
    body_count = 1 if SCENARIOS[scenario_key].object_kind == "pi" else 2
    return {
        "is_success": False,
        "scenario": scenario_key,
        "scene_hash": ("a" if body_count == 1 else "b") * 64,
        "layout_hash": layout,
        "body_count": body_count,
        "held_steps": 0,
        "lifted_ever": False,
        "off_table": terminal,
        "fallen": False,
        "terminal_reason": "off_table" if terminal else "running",
        "left_contact_ever": True,
        "right_contact_ever": False,
        "both_arms_participated": False,
        "interference_ever": False,
        "left_joint_travel": 0.1,
        "right_joint_travel": 0.0,
        "target_area_coverage": 0.1,
        **{
            f"body_{index}_{suffix}": 0.1
            for index in range(body_count)
            for suffix in ("xy_error", "yaw_error", "roll", "pitch", "height_error", "target_area_coverage")
        },
    }


def _synthetic_matrix(tmp_path: Path) -> dict[str, object]:
    source_sha = "d" * 40
    upstream_sha = UPSTREAM_SHA
    run_id = "f" * 32
    runs = {}
    for scenario_key in scenarios.CUSTOM_SCENARIOS:
        spec = SCENARIOS[scenario_key]
        episodes = []
        for seed in range(3):
            root = tmp_path / scenario_key / f"seed-{seed}"
            root.mkdir(parents=True)
            poses = sample_layout(str(spec.object_kind), seed)
            pose_rows = [[pose.name, *pose.vector()] for pose in poses]
            pose_hash = layout_hash(poses)
            reset = {
                **_matrix_info(scenario_key, pose_hash),
                "sampled_poses": pose_rows,
                "settled_poses": pose_rows,
                "home_joint_positions": [0.0] * 14,
                "pusher_position": 0.5,
                "layout_provenance": {"requested_seed": seed, "effective_seed": seed, "attempt": 0},
                "descriptor_sha256": descriptor_sha256(),
            }
            final = _matrix_info(scenario_key, pose_hash, terminal=True)
            command = [value / 20 for value in range(14)]
            plot_id = f"{run_id}-{scenario_key}-{seed}-plot"
            video_id = f"{run_id}-{scenario_key}-{seed}"
            rows = [
                {
                    "schema": 1,
                    "event": "metadata",
                    "timestamp_utc": "2026-08-28T12:00:00.000Z",
                    "monotonic_ns": 1,
                    "run_id": run_id,
                    "profile": "pi0_aloha_sim",
                    "checkpoint_label": "pi0_aloha_sim",
                    "source_sha": source_sha,
                    "upstream_sha": upstream_sha,
                    "seeds": [seed],
                    "task": spec.gym_id,
                    "scenario": scenario_key,
                    "scene_hash": final["scene_hash"],
                    "target_area_coverage_method": "exact-planar-union-v1",
                },
                {
                    "schema": 1,
                    "event": "step",
                    "timestamp_utc": "2026-08-28T12:00:00.020Z",
                    "monotonic_ns": 2,
                    "step": 0,
                    "applied_step": 1,
                    "elapsed_seconds": 0.02,
                    "actual_joint_positions": [0.0] * 14,
                    "commanded_joint_positions": command,
                    "scenario_info": final,
                },
                {
                    "schema": 1,
                    "event": "terminal",
                    "timestamp_utc": "2026-08-28T12:00:00.021Z",
                    "monotonic_ns": 3,
                    "status": "complete",
                    "episodes": 1,
                    "infrastructure_pass": True,
                    "steps_applied": 1,
                    "trajectory_sample_count": 1,
                    "trajectory_joint_count": 14,
                    "trajectory_step_coverage": 1.0,
                    "trajectory_plot_status": "passed",
                    "trajectory_plot_id": plot_id,
                    "video_ids": [video_id],
                    "push_success": 0,
                    "lifted_count": 0,
                    "off_table_count": 1,
                    "fallen_count": 0,
                    "left_contact_count": 1,
                    "right_contact_count": 0,
                    "both_arms_count": 0,
                    "interference_count": 0,
                    "time_limit_count": 0,
                    "videos_passed": 1,
                    "coverage_sample_count": 1,
                    "initial_target_area_coverage_percent": 10.0,
                    "final_target_area_coverage_percent": 10.0,
                    "best_target_area_coverage_percent": 10.0,
                    "best_target_area_coverage_step": 1,
                    "time_to_best_target_area_coverage_seconds": 0.02,
                    "episode_elapsed_seconds": 0.02,
                },
            ]
            telemetry_path = root / "telemetry.jsonl"
            telemetry_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            plot_path = root / "joint-trajectory.png"
            video_path = root / "episode.mp4"
            plot_path.write_bytes(b"plot")
            video_path.write_bytes(b"video")
            episodes.append(
                {
                    "status": "complete",
                    "infrastructure_pass": True,
                    "cleanup_pending": False,
                    "errors": [],
                    "profile": "pi0_aloha_sim",
                    "checkpoint_label": "pi0_aloha_sim",
                    "policy_backend": "pytorch",
                    "server_metadata": {
                        "policy_profile": "pi0_aloha_sim",
                        "config_name": POLICY_PROFILES["pi0_aloha_sim"].config_name,
                        "checkpoint_label": "pi0_aloha_sim",
                        "checkpoint_variant": "pi0_aloha_sim_pytorch",
                        "policy_backend": "pytorch",
                        "action_horizon": 50,
                        "action_dimension": 14,
                        "source_sha": source_sha,
                        "compact_masked_images": True,
                        "torch_platform": "cuda",
                        "torch_device": "NVIDIA GeForce RTX 3090",
                        "torch_model_device": "cuda:0",
                    },
                    "source_sha": source_sha,
                    "upstream_sha": upstream_sha,
                    "task": spec.gym_id,
                    "scenario": scenario_key,
                    "scene_hash": final["scene_hash"],
                    "seed": seed,
                    "episode": {
                        "steps_applied": 1,
                        "terminated": True,
                        "truncated": False,
                        "task_success": False,
                        "coverage_method": "exact-planar-union-v1",
                        "coverage_sample_count": 1,
                        "initial_target_area_coverage_percent": 10.0,
                        "final_target_area_coverage_percent": 10.0,
                        "best_target_area_coverage_percent": 10.0,
                        "best_target_area_coverage_step": 1,
                        "time_to_best_target_area_coverage_seconds": 0.02,
                        "wall_seconds": 0.02,
                        "reset_info": reset,
                        "final_info": final,
                    },
                    "telemetry": {"path": str(telemetry_path), "writer_closed": True, "write_p95_ms": 0.1},
                    "trajectory": {
                        "sample_count": 1,
                        "joint_count": 14,
                        "step_coverage": 1.0,
                        "plot_status": "passed",
                        "plot_id": plot_id,
                        "actual_series_count": 14,
                        "commanded_series_count": 14,
                        "path": str(plot_path),
                    },
                    "video": {
                        "id": video_id,
                        "status": "complete",
                        "path": str(video_path),
                        "frames": 1,
                        "camera_views": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        "layout": "horizontal",
                        "validation": {"frames": 1, "fps": 50.0, "shape": [224, 672, 3]},
                    },
                }
            )
        runs[scenario_key] = {"status": "passed", "gpu_coverage_pass": True, "episodes": episodes}
    return {
        "schema": 1,
        "status": "passed",
        "batch_id": "20260828T120000.000000Z",
        "run_id": run_id,
        "profile": "pi0_aloha_sim",
        "checkpoint_label": "pi0_aloha_sim",
        "source_sha": source_sha,
        "upstream_sha": upstream_sha,
        "descriptor_sha256": descriptor_sha256(),
        "gpu_metrics_interval_seconds": 1.0,
        "gpu_coverage_pass": True,
        "seeds": [0, 1, 2],
        "scenarios": list(scenarios.CUSTOM_SCENARIOS),
        "scenario_runs": runs,
        "error": None,
    }


def test_valid_matrix_is_exact_safe_and_allows_zero_successes(tmp_path: Path) -> None:
    public = validate_matrix(_synthetic_matrix(tmp_path), require_gpu=False)
    assert set(public) == {
        "schema",
        "status",
        "batch_id",
        "profile",
        "checkpoint_label",
        "source_sha",
        "upstream_sha",
        "descriptor_sha256",
        "seeds",
        "episode_count",
        "infrastructure_pass",
        "pairing_pass",
        "scenarios",
        "results",
    }
    assert public["episode_count"] == 12
    assert all(result["push_success"] == 0 for result in public["results"])
    assert str(tmp_path) not in json.dumps(public)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "pairing",
        "infra",
        "trajectory",
        "video",
        "overhead",
        "source",
        "run_id",
        "terminal",
        "running",
        "both_end",
        "partial_tail",
        "aliased_artifact",
        "server_metadata",
        "final_mismatch",
        "layout_hash",
        "pose_nonfinite",
        "upstream",
    ],
)
def test_matrix_rejects_incomplete_mixed_or_corrupt_evidence(tmp_path: Path, mutation: str) -> None:
    raw = _synthetic_matrix(tmp_path)
    runs = raw["scenario_runs"]
    if mutation == "missing":
        del runs["push_pi_dual"]
    elif mutation == "pairing":
        runs["push_pi_dual"]["episodes"][0]["episode"]["reset_info"]["sampled_poses"] = [[9.0]]
    elif mutation == "upstream":
        raw["upstream_sha"] = "0" * 40
        for run in runs.values():
            for episode in run["episodes"]:
                episode["upstream_sha"] = "0" * 40
                path = Path(episode["telemetry"]["path"])
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                rows[0]["upstream_sha"] = "0" * 40
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    else:
        episode = runs["push_pi_single"]["episodes"][0]
        if mutation == "infra":
            episode["infrastructure_pass"] = False
        elif mutation == "trajectory":
            episode["trajectory"]["sample_count"] = 0
        elif mutation == "video":
            episode["video"]["frames"] = 0
        elif mutation == "overhead":
            episode["telemetry"]["write_p95_ms"] = 1.0
        elif mutation == "source":
            episode["source_sha"] = "0" * 40
        elif mutation == "both_end":
            episode["episode"]["truncated"] = True
        elif mutation == "aliased_artifact":
            episode["trajectory"]["path"] = episode["video"]["path"]
        elif mutation == "server_metadata":
            episode["server_metadata"]["torch_device"] = "CPU"
        elif mutation == "pose_nonfinite":
            episode["episode"]["reset_info"]["settled_poses"][0][1] = float("nan")
        else:
            path = Path(episode["telemetry"]["path"])
            if mutation == "partial_tail":
                path.write_bytes(path.read_bytes() + b"{")
            else:
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                if mutation == "run_id":
                    rows[0]["run_id"] = "0" * 32
                elif mutation == "terminal":
                    rows[-1]["steps_applied"] = 2
                elif mutation == "final_mismatch":
                    rows[1]["scenario_info"]["left_contact_ever"] = False
                elif mutation == "layout_hash":
                    for info in (
                        episode["episode"]["reset_info"],
                        episode["episode"]["final_info"],
                        rows[1]["scenario_info"],
                    ):
                        info["layout_hash"] = "9" * 64
                else:
                    episode["episode"]["final_info"]["terminal_reason"] = "running"
                    episode["episode"]["final_info"]["off_table"] = False
                    rows[1]["scenario_info"]["terminal_reason"] = "running"
                    rows[1]["scenario_info"]["off_table"] = False
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix"):
        validate_matrix(copy.deepcopy(raw), require_gpu=False)


def test_matrix_interruption_preserves_partial_raw_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Sampler:
        def check(self) -> None:
            return None

        def stop(self, root: Path) -> Path:
            return root / "gpu-metrics.jsonl"

    calls = 0

    def interrupt(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return {"status": "complete"}

    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_mac_sim_config", lambda: MacSimConfig(output_dir=tmp_path)
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.load_remote_config", RemoteConfig)
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.verify_ready_tunnel", lambda config: ({}, "d" * 40))
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.start_gpu_sampler", lambda *args: Sampler())
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix._run_seed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_matrix()
    raw_path = next(tmp_path.glob("scenarios_0827/*/pi05_aloha_base/matrix.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["status"] == "interrupted"
    assert len(raw["scenario_runs"]["push_pi_single"]["episodes"]) == 1
    assert not (raw_path.parent / "matrix-summary.json").exists()
    progress = capsys.readouterr()
    assert progress.out == ""
    assert "matrix start profile=pi05_aloha_base" in progress.err
    assert "matrix end status=interrupted episodes=1/12" in progress.err


def test_matrix_rejects_diagnostic_episode_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_mac_sim_config",
        lambda: MacSimConfig(episode_steps=6000),
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.load_remote_config", RemoteConfig)
    with pytest.raises(RemoteError, match="300-step"):
        run_matrix()


def test_matrix_interruption_remains_primary_when_sampler_stop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sampler:
        def check(self) -> None:
            return None

        def stop(self, root: Path) -> Path:
            raise RuntimeError("stop failed")

    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_mac_sim_config", lambda: MacSimConfig(output_dir=tmp_path)
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.load_remote_config", RemoteConfig)
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.verify_ready_tunnel", lambda config: ({}, "d" * 40))
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.start_gpu_sampler", lambda *args: Sampler())
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix._run_seed",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        run_matrix()
    raw_path = next(tmp_path.glob("scenarios_0827/*/pi05_aloha_base/matrix.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["status"] == "interrupted"
    assert raw["error"]["type"] == "KeyboardInterrupt"
    assert raw["cleanup_error"]["type"] == "RuntimeError"


def test_matrix_uses_one_sampler_and_checks_before_and_after_all_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lifecycle = []

    class Sampler:
        def check(self) -> None:
            lifecycle.append("check")

        def stop(self, root: Path) -> Path:
            lifecycle.append("stop")
            return root / "gpu-metrics.jsonl"

    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_mac_sim_config", lambda: MacSimConfig(output_dir=tmp_path)
    )
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_remote_config",
        lambda: RemoteConfig(policy_profile=POLICY_PROFILES["pi0_aloha_sim"]),
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix.verify_ready_tunnel", lambda config: ({}, "d" * 40))
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.start_gpu_sampler",
        lambda *args: lifecycle.append("start") or Sampler(),
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix._run_seed", lambda *args: {"status": "complete"})
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix._validate_batch_gpu", lambda *args: None)
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix._write_performance_summary", lambda *args: None)
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.validate_matrix",
        lambda *args, **kwargs: {"schema": 1, "status": "passed"},
    )
    summary = run_matrix()
    assert summary.is_file()
    assert lifecycle == ["start", *(["check"] * 24), "stop"]
    progress = capsys.readouterr()
    assert progress.out == ""
    assert "matrix validating evidence" in progress.err
    assert "matrix end status=passed episodes=12/12" in progress.err


def test_scenario_metrics_rejects_stale_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _synthetic_matrix(tmp_path / "evidence")
    root = tmp_path / "scenarios_0827" / "batch" / "pi0_aloha_sim"
    root.mkdir(parents=True)
    (root / "matrix.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_mac_sim_config", lambda: MacSimConfig(output_dir=tmp_path)
    )
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_remote_config",
        lambda: RemoteConfig(policy_profile=POLICY_PROFILES["pi0_aloha_sim"]),
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix._candidate_sha", lambda: "0" * 40)
    with pytest.raises(RemoteError, match="exact current candidate"):
        summarize_latest()


def test_scenario_metrics_revalidates_missing_gpu_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _synthetic_matrix(tmp_path / "evidence")
    root = tmp_path / "scenarios_0827" / "batch" / "pi0_aloha_sim"
    root.mkdir(parents=True)
    (root / "matrix.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_mac_sim_config", lambda: MacSimConfig(output_dir=tmp_path)
    )
    monkeypatch.setattr(
        "tools.remote_aloha.scenario_matrix.load_remote_config",
        lambda: RemoteConfig(policy_profile=POLICY_PROFILES["pi0_aloha_sim"]),
    )
    monkeypatch.setattr("tools.remote_aloha.scenario_matrix._candidate_sha", lambda: raw["source_sha"])
    with pytest.raises(ValueError, match="GPU evidence"):
        summarize_latest()
