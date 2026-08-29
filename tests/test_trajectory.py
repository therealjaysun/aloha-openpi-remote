from __future__ import annotations

import math
import os
from pathlib import Path

from matplotlib.axes import Axes
from PIL import Image
import pytest

from tools.remote_aloha import trajectory

_EXPECTED_JOINT_LIMITS = tuple(
    (f"{side}_{name}", lower, upper)
    for side in ("left", "right")
    for name, lower, upper in (
        ("waist", -3.14158, 3.14158),
        ("shoulder", -1.85005, 1.25664),
        ("elbow", -1.76278, 1.6057),
        ("forearm_roll", -3.14158, 3.14158),
        ("wrist_angle", -1.8675, 2.23402),
        ("wrist_rotate", -3.14158, 3.14158),
        ("gripper", 0.0, 1.0),
    )
)


def _step(number: int, elapsed: float, *, commanded: bool = True, fraction: float | None = None) -> dict[str, object]:
    fraction = number / 4 if fraction is None else fraction
    values = [lower + (upper - lower) * fraction for _, lower, upper in _EXPECTED_JOINT_LIMITS]
    event: dict[str, object] = {
        "schema": 1,
        "event": "step",
        "step": number - 1,
        "applied_step": number,
        "elapsed_seconds": elapsed,
        "actual_joint_positions": values,
    }
    if commanded:
        event["commanded_joint_positions"] = values
    return event


@pytest.mark.parametrize(
    "value",
    [[0.0] * 13, [0.0] * 15, [0.0] * 13 + [math.nan], [0.0] * 13 + [math.inf], [0.0] * 13 + [True]],
)
def test_joint_vector_rejects_wrong_shape_and_nonfinite_values(value: list[float]) -> None:
    with pytest.raises(ValueError, match=r"exactly|finite"):
        trajectory.validate_joint_vector(value, "joints")


@pytest.mark.parametrize(
    ("events", "steps_applied", "message"),
    [
        ([_step(2, 0.0)], 1, "sequential"),
        ([{**_step(1, 0.0), "step": 1}], 1, "simulation step"),
        ([_step(1, 0.1), _step(2, 0.05)], 2, "monotonic"),
        ([_step(1, math.nan)], 1, "finite"),
        ([_step(1, -0.1)], 1, "nonnegative"),
        ([_step(1, 0.0)], 2, "samples"),
        ([{**_step(1, 0.0), "schema": 2}], 1, "schema-1"),
        ([{**_step(1, 0.0), "schema": True}], 1, "schema-1"),
        ([{key: value for key, value in _step(1, 0.0).items() if key != "actual_joint_positions"}], 1, "exactly"),
        ([_step(1, 0.0), _step(2, 0.1, commanded=False)], 2, "every trajectory row"),
    ],
)
def test_trajectory_rejects_invalid_step_coverage_or_rows(
    events: list[dict[str, object]], steps_applied: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        trajectory.summarize_trajectory(events, steps_applied)


def test_valid_interrupted_episode_keeps_exact_partial_coverage(tmp_path: Path) -> None:
    events = [_step(1, 0.0), _step(2, 0.02)]
    summary = trajectory.summarize_trajectory(events, 2)
    assert summary == {
        "sample_count": 2,
        "joint_count": 14,
        "step_coverage": 1.0,
        "plot_status": "not_generated",
        "plot_id": None,
        "actual_series_count": 14,
        "commanded_series_count": 14,
    }
    plotted = trajectory.write_trajectory_plot(events, 2, tmp_path / "partial.png", "episode-partial-joints")
    assert plotted["plot_status"] == "passed"
    assert plotted["sample_count"] == 2


def test_normalization_uses_fixed_physical_limits_for_a_single_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = [100 * (0 - lower) / (upper - lower) for _, lower, upper in _EXPECTED_JOINT_LIMITS]
    calls = []
    original_plot = Axes.plot

    def capture(self: Axes, *args: object, **kwargs: object):
        calls.append((args, kwargs))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    trajectory.write_trajectory_plot(
        [
            {
                "schema": 1,
                "event": "step",
                "step": 0,
                "applied_step": 1,
                "elapsed_seconds": 0.02,
                "actual_joint_positions": [0.0] * 14,
            }
        ],
        1,
        tmp_path / "fixed-limits.png",
        "fixed-limits",
    )
    assert [call[0][1][0] for call in calls] == pytest.approx(expected)


def test_empty_episode_returns_no_samples_without_creating_a_file(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.png"
    result = trajectory.write_trajectory_plot([], 0, path, "episode-0-joints")
    assert result["plot_status"] == "no_samples"
    assert result["plot_id"] is None
    assert not path.exists()


def test_atomic_plot_contains_all_actual_and_commanded_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    references = []
    original_plot = Axes.plot
    original_axhline = Axes.axhline

    def capture(self: Axes, *args: object, **kwargs: object):
        calls.append((args, kwargs))
        return original_plot(self, *args, **kwargs)

    def capture_reference(self: Axes, y: float = 0, *args: object, **kwargs: object):
        references.append((y, kwargs))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    monkeypatch.setattr(Axes, "axhline", capture_reference)
    path = tmp_path / "trajectory.png"
    result = trajectory.write_trajectory_plot(
        [_step(1, 0.0, fraction=0.25), _step(2, 0.02, fraction=0.5), _step(3, 0.04, fraction=0.75)],
        3,
        path,
        "episode-0-joints",
    )
    with Image.open(path) as image:
        image.verify()
        assert image.format == "PNG"
    assert result["plot_status"] == "passed"
    assert trajectory.JOINT_LIMITS == _EXPECTED_JOINT_LIMITS
    assert result["actual_series_count"] == 14
    assert result["commanded_series_count"] == 14
    actual_calls = [call for call in calls if call[1].get("label") != "_nolegend_"]
    assert len(actual_calls) == 14
    assert len([call for call in calls if call[1].get("linestyle") == "--"]) == 14
    assert all(call[0][1] == pytest.approx([25.0, 50.0, 75.0]) for call in actual_calls)
    assert references == [(50, {"color": "0.45", "linestyle": ":", "linewidth": 0.8})]
    assert path.stat().st_mode & 0o777 == 0o600


def test_atomic_replace_failure_preserves_previous_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "trajectory.png"
    path.write_bytes(b"previous")
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(OSError, match="replace failed"):
        trajectory.write_trajectory_plot([_step(1, 0.0)], 1, path, "episode-0-joints")
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]
