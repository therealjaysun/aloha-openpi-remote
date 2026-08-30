from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import os
from pathlib import Path
import re
import tempfile

# Pinned gym-aloha 0.1.1 XML arm limits; its public gripper coordinate is documented as 0=closed, 1=open.
_JOINTS_PER_ARM = (
    ("waist", -3.14158, 3.14158),
    ("shoulder", -1.85005, 1.25664),
    ("elbow", -1.76278, 1.6057),
    ("forearm_roll", -3.14158, 3.14158),
    ("wrist_angle", -1.8675, 2.23402),
    ("wrist_rotate", -3.14158, 3.14158),
    ("gripper", 0.0, 1.0),
)
# gym-aloha 0.1.1 exposes agent_pos in left-arm then right-arm order.
JOINT_LIMITS = tuple(
    (f"{side}_{name}", lower, upper) for side in ("left", "right") for name, lower, upper in _JOINTS_PER_ARM
)
JOINT_COUNT = len(JOINT_LIMITS)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def validate_joint_vector(value: object, label: str, joint_limits=JOINT_LIMITS) -> list[float]:
    joint_count = len(joint_limits)
    if not isinstance(value, list | tuple) or len(value) != joint_count:
        raise ValueError(f"{label} must contain exactly {joint_count} joints")
    joints = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            raise ValueError(f"{label} must contain only finite numbers")
        joints.append(float(item))
    return joints


def _trajectory_rows(
    events: Iterable[Mapping[str, object]], steps_applied: int, joint_limits=JOINT_LIMITS
) -> list[dict[str, object]]:
    if isinstance(steps_applied, bool) or not isinstance(steps_applied, int) or steps_applied < 0:
        raise ValueError("steps_applied must be a nonnegative integer")
    rows: list[dict[str, object]] = []
    commanded_present: bool | None = None
    previous_elapsed = 0.0
    for event in events:
        if not isinstance(event, Mapping) or event.get("event") != "step":
            continue
        schema = event.get("schema")
        if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
            raise ValueError("trajectory rows must be schema-1 step events")
        applied_step = event.get("applied_step")
        if isinstance(applied_step, bool) or not isinstance(applied_step, int) or applied_step != len(rows) + 1:
            raise ValueError("trajectory applied_step values must be sequential and one-based")
        simulation_step = event.get("step")
        if (
            isinstance(simulation_step, bool)
            or not isinstance(simulation_step, int)
            or simulation_step != applied_step - 1
        ):
            raise ValueError("trajectory step must exactly match the zero-based simulation step")
        elapsed = event.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed < 0
            or (rows and elapsed < previous_elapsed)
        ):
            raise ValueError("trajectory elapsed_seconds must be finite, nonnegative, and monotonic")
        has_command = "commanded_joint_positions" in event
        if commanded_present is None:
            commanded_present = has_command
        elif commanded_present != has_command:
            raise ValueError("commanded joint positions must be present for every trajectory row or none")
        row: dict[str, object] = {
            "applied_step": applied_step,
            "elapsed_seconds": float(elapsed),
            "actual": validate_joint_vector(
                event.get("actual_joint_positions"), "actual_joint_positions", joint_limits
            ),
        }
        if has_command:
            row["commanded"] = validate_joint_vector(
                event.get("commanded_joint_positions"), "commanded_joint_positions", joint_limits
            )
        rows.append(row)
        previous_elapsed = float(elapsed)
    if len(rows) != steps_applied:
        raise ValueError(f"trajectory has {len(rows)} samples for {steps_applied} successfully applied steps")
    return rows


def _summary(rows: list[dict[str, object]], steps_applied: int, joint_count: int) -> dict[str, object]:
    return {
        "sample_count": len(rows),
        "joint_count": joint_count,
        "step_coverage": 1.0 if steps_applied == 0 else len(rows) / steps_applied,
        "plot_status": "not_generated" if rows else "no_samples",
        "plot_id": None,
        "actual_series_count": joint_count if rows else 0,
        "commanded_series_count": joint_count if rows and "commanded" in rows[0] else 0,
    }


def summarize_trajectory(
    events: Iterable[Mapping[str, object]], steps_applied: int, *, joint_limits=JOINT_LIMITS
) -> dict[str, object]:
    return _summary(_trajectory_rows(events, steps_applied, joint_limits), steps_applied, len(joint_limits))


def _normalize(vector: list[float], joint_limits) -> list[float]:
    return [
        100.0 * (value - lower) / (upper - lower) for value, (_, lower, upper) in zip(vector, joint_limits, strict=True)
    ]


def write_trajectory_plot(
    events: Iterable[Mapping[str, object]],
    steps_applied: int,
    path: str | Path,
    plot_id: str,
    *,
    joint_limits=JOINT_LIMITS,
    title: str = "ALOHA joint trajectory",
    footnote: str = (
        "Arms use fixed gym-aloha 0.1.1 XML radian limits; grippers use its documented 0-1 close/open "
        "coordinate. 50% is zero only for symmetric waist/roll joints."
    ),
) -> dict[str, object]:
    rows = _trajectory_rows(events, steps_applied, joint_limits)
    summary = _summary(rows, steps_applied, len(joint_limits))
    if not rows:
        return summary
    if not isinstance(plot_id, str) or not _SAFE_ID.fullmatch(plot_id):
        raise ValueError("plot_id must be a safe local identifier")
    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("trajectory plot path must end in .png")

    # Import lazily so non-plot telemetry paths do not pay Matplotlib startup cost.
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    figure = Figure(figsize=(14, 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    figure.subplots_adjust(left=0.08, right=0.78, bottom=0.15, top=0.92)
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    actual = [_normalize(row["actual"], joint_limits) for row in rows]  # type: ignore[arg-type]
    commanded = [_normalize(row["commanded"], joint_limits) for row in rows] if "commanded" in rows[0] else []  # type: ignore[arg-type]
    colors = [
        colormaps["Blues"](0.42 + index * 0.08) if index < 7 else colormaps["Oranges"](0.42 + (index - 7) * 0.08)
        for index in range(len(joint_limits))
    ]
    for index, ((name, _, _), color) in enumerate(zip(joint_limits, colors, strict=True)):
        axes.plot(elapsed, [row[index] for row in actual], color=color, linewidth=1.5, label=name.replace("_", " "))
        if commanded:
            axes.plot(
                elapsed,
                [row[index] for row in commanded],
                color=color,
                linewidth=0.8,
                linestyle="--",
                alpha=0.65,
                label="_nolegend_",
            )
    axes.axhline(50, color="0.45", linestyle=":", linewidth=0.8)
    axes.set(xlabel="Monotonic elapsed time (s)", ylabel="Authoritative physical range (%)", ylim=(0, 100))
    axes.grid(alpha=0.2)
    axes.set_title(title)
    handles, labels = axes.get_legend_handles_labels()
    if commanded:
        handles.append(Line2D([0], [0], color="0.35", linewidth=0.8, linestyle="--"))
        labels.append("commanded (dashed)")
    axes.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")
    figure.text(
        0.08,
        0.025,
        footnote,
        fontsize=8,
        color="0.35",
    )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        os.chmod(temporary, 0o600)
        figure.savefig(temporary, format="png", dpi=150)
        os.replace(temporary, output)
        temporary = None
    finally:
        figure.clear()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {**summary, "plot_status": "passed", "plot_id": plot_id}
