from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from tools.remote_aloha.config import MacSimConfig
from tools.remote_aloha.config import RemoteConfig
from tools.remote_aloha.metrics import summarize_latest
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.telemetry import JsonlWriter
from tools.remote_aloha.telemetry import aggregate_events
from tools.remote_aloha.telemetry import aggregate_jsonl
from tools.remote_aloha.telemetry import publishable_summary
from tools.remote_aloha.telemetry import read_jsonl
from tools.remote_aloha.telemetry import render_markdown
from tools.remote_aloha.telemetry import summarize_values
from tools.remote_aloha.telemetry import write_summary


def _event(name: str, monotonic_ns: int, **fields: object) -> dict[str, object]:
    return {
        "schema": 1,
        "event": name,
        "timestamp_utc": "2026-08-28T12:00:00.000Z",
        "monotonic_ns": monotonic_ns,
        **fields,
    }


def test_metrics_requires_a_completed_selected_profile_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.metrics.load_mac_sim_config", lambda: MacSimConfig(output_dir=tmp_path / "outputs")
    )
    monkeypatch.setattr("tools.remote_aloha.metrics.load_remote_config", RemoteConfig)
    with pytest.raises(RemoteError, match="no Phase 5 run exists"):
        summarize_latest()


def test_stock_metrics_directs_custom_scenarios_to_matrix_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.remote_aloha.metrics.load_mac_sim_config",
        lambda: MacSimConfig(
            task="pi_robotics/PushPiSingleArm-v0",
            scenario=SCENARIOS["push_pi_single"],
        ),
    )
    with pytest.raises(RemoteError, match="scenario-metrics"):
        summarize_latest()


def test_jsonl_writer_is_private_line_buffered_and_round_trips_numpy_scalar(tmp_path: Path) -> None:
    ticks = iter((10, 20))
    path = tmp_path / "events.jsonl"
    writer = JsonlWriter(path, utc_now=lambda: "2026-08-28T12:00:00.000Z", monotonic_ns=lambda: next(ticks))
    writer.write("metadata", profile="pi0_aloha_sim", seed=np.int64(0))
    writer.write("step", step=1, metrics={"sim_step_ms": np.float32(0.5)})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(lines[0])["seed"] == 0
    assert json.loads(lines[1])["metrics"]["sim_step_ms"] == 0.5
    writer.close()
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write("step", step=2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.array([1.0])])
def test_writer_rejects_non_json_or_non_finite_values_without_writing(tmp_path: Path, value: object) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlWriter(path) as writer, pytest.raises(ValueError, match="telemetry|NumPy"):
        writer.write("step", value=value)
    assert path.read_bytes() == b""


def test_reader_keeps_valid_lines_and_ignores_only_an_incomplete_final_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    valid = json.dumps(_event("metadata", 1, profile="pi0_aloha_sim"))
    path.write_bytes((valid + '\n{"schema":1,"event":').encode())
    result = read_jsonl(path)
    assert [event["event"] for event in result.events] == ["metadata"]
    assert result.partial_final_line_ignored is True

    path.write_text(valid + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)


def test_reader_rejects_json_nan(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"schema":1,"event":"metadata","timestamp_utc":"2026-08-28T12:00:00Z",' '"monotonic_ns":1,"value":NaN}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path)


def test_reader_rejects_boolean_schema(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({**_event("metadata", 1), "schema": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema-1"):
        read_jsonl(path)


def test_known_aggregation_fixture_and_empty_and_single_samples() -> None:
    assert summarize_values([]) == {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    assert summarize_values([7]) == {"count": 1, "mean": 7.0, "p50": 7.0, "p95": 7.0, "max": 7.0}
    result = summarize_values([1, 2, 3, 4])
    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["max"] == 4.0


def test_aggregate_reports_counts_metrics_terminal_status_and_step_coverage() -> None:
    events = [
        _event(
            "metadata",
            1,
            profile="pi0_aloha_sim",
            source_sha="a" * 40,
            upstream_sha="b" * 40,
            seeds=[0],
        ),
        _event("policy_result", 2, metrics={"cold_inference_ms": 400.0}),
        _event("step", 3, step=0, metrics={"sim_step_ms": 0.5}),
        _event("step", 4, step=1, metrics={"sim_step_ms": 1.5}),
        _event("terminal", 5, status="complete", steps_applied=2, infrastructure_pass=True),
    ]
    summary = aggregate_events(events)
    assert summary["status"] == "complete"
    assert summary["event_counts"] == {"metadata": 1, "policy_result": 1, "step": 2, "terminal": 1}
    assert summary["metrics"]["sim_step_ms"] == {
        "count": 2,
        "mean": 1.0,
        "p50": 1.0,
        "p95": pytest.approx(1.45),
        "max": 1.5,
    }
    assert summary["telemetry"]["step_coverage"] == 1.0


def test_partial_jsonl_aggregates_without_inventing_terminal_status(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(_event("metadata", 1, profile="pi0_aloha_sim")) + '\n{"schema":', encoding="utf-8")
    summary = aggregate_jsonl(path)
    assert summary["status"] == "partial"
    assert summary["event_count"] == 1
    assert summary["telemetry"]["partial_final_line_ignored"] is True
    assert summary["telemetry"]["terminal_event_present"] is False


def test_aggregation_rejects_missing_metadata_duplicate_terminal_and_nonfinite_metric() -> None:
    with pytest.raises(ValueError, match="first"):
        aggregate_events([_event("step", 1)])
    with pytest.raises(ValueError, match="last and unique"):
        aggregate_events(
            [
                _event("metadata", 1, profile="pi0_aloha_sim"),
                _event("terminal", 2, status="failed"),
                _event("terminal", 3, status="complete"),
            ]
        )
    with pytest.raises(ValueError, match="finite"):
        summarize_values([np.float32(np.nan)])


def test_publishable_summary_is_allowlisted_labels_profile_and_does_not_mutate_raw() -> None:
    private_path = "/" + "Users/private/output"
    raw = aggregate_events(
        [
            _event(
                "metadata",
                1,
                profile="pi05_aloha_base",
                source_sha="a" * 40,
                upstream_sha="b" * 40,
                run_id="c" * 32,
                private_hostname="private.example",
                absolute_path=private_path,
            ),
            _event(
                "terminal",
                2,
                status="complete",
                infrastructure_pass=True,
                trajectory_sample_count=300,
                trajectory_joint_count=14,
                trajectory_step_coverage=1.0,
                trajectory_plot_status="passed",
                trajectory_plot_id="run-seed-0-joint-trajectory",
                private_username="secret-user",
                metrics={"warm_inference_ms": 350.0, "private_metric": 1.0},
            ),
        ]
    )
    public = publishable_summary(raw)
    encoded = json.dumps(public, sort_keys=True)
    assert public["metadata"]["profile"] == "pi05_aloha_base"
    assert public["metrics"].keys() == {"warm_inference_ms"}
    assert public["result"]["trajectory_sample_count"] == 300
    assert public["result"]["trajectory_joint_count"] == 14
    assert public["result"]["trajectory_plot_id"] == "run-seed-0-joint-trajectory"
    assert "private.example" not in encoded
    assert private_path not in encoded
    assert "secret-user" not in encoded
    assert "private_hostname" in raw["metadata"]
    markdown = render_markdown(raw)
    assert "pi05_aloha_base" in markdown
    assert "private" not in markdown


def test_publishable_custom_scenario_keeps_safe_identity_counts_and_omits_private_fields() -> None:
    raw = aggregate_events(
        [
            _event(
                "metadata",
                1,
                profile="pi0_aloha_sim",
                checkpoint_label="pi0_aloha_sim",
                run_id="c" * 32,
                task="pi_robotics/PushLettersSingleArm-v0",
                scenario="push_letters_single",
                scene_hash="d" * 64,
                target_area_coverage_method="exact-planar-union-v1",
                camera_views=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                prompt="private run prompt",
                absolute_path="/private/output",
            ),
            _event(
                "terminal",
                2,
                status="complete",
                steps_applied=300,
                trajectory_sample_count=300,
                push_success=1,
                lifted_count=0,
                both_arms_count=0,
                time_limit_count=2,
                videos_passed=3,
                episodes=3,
                coverage_sample_count=300,
                initial_target_area_coverage_percent=12.0,
                final_target_area_coverage_percent=80.0,
                best_target_area_coverage_percent=95.0,
                best_target_area_coverage_step=120,
                time_to_best_target_area_coverage_seconds=2.4,
                episode_elapsed_seconds=6.0,
                private_machine="desktop-name",
            ),
        ]
    )
    public = publishable_summary(raw)
    encoded = json.dumps(public, sort_keys=True)
    assert public["metadata"]["scenario"] == "push_letters_single"
    assert public["metadata"]["scene_hash"] == "d" * 64
    assert public["metadata"]["camera_views"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert public["result"]["push_success"] == 1
    assert public["result"]["lifted_count"] == public["result"]["both_arms_count"] == 0
    assert public["result"]["time_limit_count"] == 2
    assert public["result"]["videos_passed"] == 3
    assert public["result"]["best_target_area_coverage_percent"] == 95.0
    assert public["result"]["time_to_best_target_area_coverage_seconds"] == 2.4
    assert "private run prompt" not in encoded
    assert "/private/output" not in encoded
    assert "desktop-name" not in encoded


@pytest.mark.parametrize(
    ("task", "scenario", "scene_hash"),
    [
        (None, "push_pi_single", "d" * 64),
        ("pi_robotics/PushPiSingleArm-v0", None, "d" * 64),
        ("gym_aloha/AlohaTransferCube-v0", "push_pi_single", "d" * 64),
        ("pi_robotics/PushPiSingleArm-v0", "push_pi_single", None),
        ("pi_robotics/PushPiSingleArm-v0", "push_pi_single", "bad"),
        ("gym_aloha/AlohaTransferCube-v0", "transfer_cube", "d" * 64),
    ],
)
def test_publishable_scenario_identity_is_fail_closed(
    task: str | None, scenario: str | None, scene_hash: str | None
) -> None:
    raw = aggregate_events(
        [
            _event(
                "metadata",
                1,
                profile="pi0_aloha_sim",
                task=task,
                scenario=scenario,
                scene_hash=scene_hash,
            ),
            _event("terminal", 2, status="failed", steps_applied=0),
        ]
    )
    with pytest.raises(ValueError, match="scenario"):
        publishable_summary(raw)


def test_publishable_per_episode_counts_are_bounded_without_aggregate_episode_field() -> None:
    raw = aggregate_events(
        [
            _event("metadata", 1, profile="pi0_aloha_sim"),
            _event("terminal", 2, status="complete", push_success=2),
        ]
    )
    with pytest.raises(ValueError, match="episode count"):
        publishable_summary(raw)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("metadata", "package_versions", {"numpy": "/" + "Users/name"}),
        ("metadata", "package_versions", {"numpy": "DESKTOP-" + "EXAMPLE"}),
        ("metadata", "camera_views", ["cam_high"]),
        ("result", "request_count", "192" + ".168.1.2"),
        ("result", "trajectory_plot_id", "/" + "Users/private/plot.png"),
        ("result", "trajectory_joint_count", 13),
        ("result", "trajectory_step_coverage", 1.01),
        ("result", "best_target_area_coverage_percent", 101.0),
        ("result", "time_to_best_target_area_coverage_seconds", -1.0),
        ("metrics", "warm_inference_ms", {"count": 1, "mean": "raw error", "p50": 1, "p95": 1, "max": 1}),
    ],
)
def test_publishable_allowed_containers_reject_private_or_wrong_typed_values(
    section: str, key: str, value: object
) -> None:
    summary = aggregate_events(
        [
            _event("metadata", 1, profile="pi0_aloha_sim", run_id="c" * 32),
            _event("terminal", 2, status="complete", request_count=1, metrics={"warm_inference_ms": 1.0}),
        ]
    )
    summary[section][key] = value
    with pytest.raises(ValueError, match="publishable"):
        publishable_summary(summary)


def test_summary_write_is_atomic_private_and_preserves_old_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "summary.json"
    path.write_text('{"old":true}\n', encoding="utf-8")
    path.chmod(0o600)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_summary(path, {"status": "partial"})
    assert path.read_text(encoding="utf-8") == '{"old":true}\n'
    assert list(tmp_path.iterdir()) == [path]


def test_line_buffered_writer_p95_overhead_is_below_one_millisecond(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    durations = []
    with JsonlWriter(path) as writer:
        for step in range(300):
            started = time.perf_counter_ns()
            writer.write(
                "step",
                episode=0,
                step=step,
                applied_step=step + 1,
                elapsed_seconds=(step + 1) / 50,
                actual_joint_positions=[float(step)] * 14,
                commanded_joint_positions=[float(step + 1)] * 14,
                scenario_info={
                    "is_success": False,
                    "scenario": "push_letters_dual",
                    "scene_hash": "a" * 64,
                    "layout_hash": "b" * 64,
                    "body_count": 2,
                    "held_steps": 0,
                    "lifted_ever": False,
                    "off_table": False,
                    "fallen": False,
                    "terminal_reason": "running",
                    "left_contact_ever": True,
                    "right_contact_ever": True,
                    "both_arms_participated": True,
                    "interference_ever": False,
                    "left_joint_travel": 0.1,
                    "right_joint_travel": 0.1,
                    "target_area_coverage": 0.1,
                    **{
                        f"body_{body}_{metric}": 0.1
                        for body in range(2)
                        for metric in (
                            "xy_error",
                            "yaw_error",
                            "roll",
                            "pitch",
                            "height_error",
                            "target_area_coverage",
                        )
                    },
                },
                metrics={"sim_step_ms": 0.5},
            )
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
    assert summarize_values(durations)["p95"] < 1.0
