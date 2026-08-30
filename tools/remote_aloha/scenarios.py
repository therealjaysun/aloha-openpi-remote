from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
import hashlib
from itertools import pairwise
import json
import math

import numpy as np

JOINT_COUNT = 14
PUSHER_POSITION = 0.5
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844
PUPPET_GRIPPER_POSITION_OPEN = 0.058
PUSHER_PHYSICAL_POSITION = PUPPET_GRIPPER_POSITION_CLOSE + PUSHER_POSITION * (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)
TABLETOP_SHA256 = "76a1571d1aa36520f2bd81c268991b99816c2a7819464d718e0fd9976fe30dce"
TABLE_BOUNDS = (-0.6096, 0.6096, 0.219, 0.981)
SPAWN_BOUNDS = (-0.22, 0.22, 0.32, 0.58)
OBJECT_HALF_HEIGHT = 0.012
SPAWN_HEIGHT = 0.04
MAX_LAYOUT_ATTEMPTS = 256
RESET_LAYOUT_ATTEMPTS = 8
RESET_SEED_STRIDE = 0x9E3779B9
SPAWN_YAW_RADIANS = math.pi / 3
BODY_CLEARANCE_METERS = 0.02
TARGET_CLEARANCE_METERS = 0.04
SUCCESS_XY_METERS = 0.03
SUCCESS_YAW_RADIANS = math.radians(15)
SUCCESS_TILT_RADIANS = math.radians(10)
SUCCESS_HEIGHT_METERS = 0.005
SUCCESS_HOLD_STEPS = 5
LIFT_METERS = 0.01
FALL_RADIANS = math.radians(30)
SETTLE_STEPS = 200
FREE_JOINT_FRICTIONLOSS = 0.01
GEOM_DENSITY = 350
GEOM_FRICTION = (1.0, 0.005, 0.0001)
GEOM_CONDIM = 4
GEOM_SOLIMP = (2.0, 1.0, 0.01)
GEOM_SOLREF = (0.01, 1.0)
VISUAL_GEOM_GROUP = 0
VISUAL_GEOM_MASS = 0.0
VISUAL_CONTACT_BITS = (0, 0)
COLLISION_GEOM_GROUP = 4
COLLISION_CONTACT_BITS = (1, 1)
TARGET_HALF_HEIGHT = 0.001
TARGET_DOT_RADIUS = 0.005
TARGET_DOT_SPACING = 0.024
TARGET_CONTACT_BITS = (0, 0)
DISPLAY_EVERY_STEPS = 5
PARKED_JOINT_TOLERANCE = 0.01
RESET_SETTLE_XY_METERS = 0.001
RESET_SETTLE_YAW_RADIANS = 0.006
RESET_TILT_RADIANS = math.radians(1)
CALIBRATION_MIN_PUSH_METERS = 0.02
CALIBRATION_MAX_HEIGHT_ERROR_METERS = 0.01
MIN_VISIBLE_PIXELS = 20
COLOR_MASK_RULES = {
    "pi": (("r", "b", 40), ("g", "b", 20), ("r", "g", 0)),
    "P": (("r", "g", 35), ("r", "b", 35)),
    "I": (("b", "r", 35), ("b", "g", 25)),
}
DESCRIPTOR_VERSION = "push-pi-v2"
TARGET_AREA_COVERAGE_METHOD = "exact-planar-union-v1"


@dataclass(frozen=True)
class Part:
    x: float
    y: float
    half_x: float
    half_y: float


@dataclass(frozen=True)
class BodyDescriptor:
    name: str
    parts: tuple[Part, ...]
    target_x: float
    target_y: float
    target_yaw: float
    rgba: tuple[float, float, float, float]
    yaw_period: float = 2 * math.pi

    @property
    def footprint_radius(self) -> float:
        return max(math.hypot(abs(part.x) + part.half_x, abs(part.y) + part.half_y) for part in self.parts)


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    gym_id: str
    object_kind: str | None
    arm_mode: str
    prompt: str | None

    @property
    def is_custom(self) -> bool:
        return self.object_kind is not None


@dataclass(frozen=True)
class Pose:
    name: str
    x: float
    y: float
    z: float
    yaw: float

    def vector(self) -> tuple[float, ...]:
        half = self.yaw / 2
        return (self.x, self.y, self.z, math.cos(half), 0.0, 0.0, math.sin(half))


@dataclass(frozen=True)
class BodyState:
    name: str
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    com_z: float | None = None


@dataclass(frozen=True)
class OutcomeState:
    held_steps: int = 0
    lifted_ever: bool = False
    success: bool = False
    off_table: bool = False
    fallen: bool = False
    terminal_reason: str = "running"


@dataclass(frozen=True)
class Participation:
    left_contact_ever: bool = False
    right_contact_ever: bool = False
    interference_ever: bool = False

    @property
    def both_arms_participated(self) -> bool:
        return self.left_contact_ever and self.right_contact_ever


PI_BODY = BodyDescriptor(
    "pi",
    (
        Part(0.0, 0.04, 0.065, 0.01),
        Part(-0.045, -0.01, 0.01, 0.05),
        Part(0.045, -0.01, 0.01, 0.05),
    ),
    0.0,
    0.66,
    0.0,
    (0.95, 0.55, 0.08, 1.0),
)
P_BODY = BodyDescriptor(
    "P",
    (
        Part(-0.04, 0.0, 0.01, 0.06),
        Part(0.0, 0.05, 0.04, 0.01),
        Part(0.0, 0.0, 0.04, 0.01),
        Part(0.04, 0.025, 0.01, 0.025),
    ),
    -0.11,
    0.66,
    0.0,
    (0.85, 0.12, 0.12, 1.0),
)
I_BODY = BodyDescriptor(
    "I",
    (
        Part(0.0, 0.05, 0.04, 0.01),
        Part(0.0, 0.0, 0.01, 0.05),
        Part(0.0, -0.05, 0.04, 0.01),
    ),
    0.11,
    0.66,
    0.0,
    (0.1, 0.35, 0.9, 1.0),
    math.pi,
)

BODIES = {"pi": (PI_BODY,), "letters": (P_BODY, I_BODY)}

LEFT_PUSH_WAYPOINTS = (
    (-0.728704, 0.404774, 1.041295, -0.724138, -1.577214, 0.082932),
    (-0.728808, 0.779605, 0.935115, -0.743773, -1.777464, -0.097735),
    (-0.653191, 0.789274, 0.974137, -0.674375, -1.822692, -0.120419),
    (-0.483485, 0.785856, 1.045402, -0.519026, -1.8675, -0.129524),
    (-0.273045, 0.760113, 1.10316, -0.308, -1.8675, -0.08898),
    (-0.000176, 0.746699, 1.127625, -0.000321, -1.8675, -0.000113),
    (0.272805, 0.760081, 1.103206, 0.307559, -1.8675, 0.088836),
    (0.483362, 0.78583, 1.045448, 0.518799, -1.8675, 0.12947),
    (0.471712, 0.391959, 1.157457, 0.470839, -1.651826, -0.009732),
)
RIGHT_PUSH_WAYPOINTS = (
    (0.728669, 0.404771, 1.041313, 0.724104, -1.577223, -0.082919),
    (0.728773, 0.779608, 0.935131, 0.743741, -1.777481, 0.097745),
    (0.653153, 0.789278, 0.974152, 0.674338, -1.822712, 0.120425),
    (0.483442, 0.78585, 1.045416, 0.518985, -1.8675, 0.129519),
    (0.272989, 0.760108, 1.10317, 0.307937, -1.8675, 0.088963),
    (0.000108, 0.746699, 1.127625, 0.00024, -1.8675, 0.000087),
    (-0.272862, 0.760086, 1.103196, -0.307621, -1.8675, -0.088853),
    (-0.483405, 0.785835, 1.045434, -0.51884, -1.8675, -0.129476),
    (-0.471759, 0.39196, 1.157445, -0.470885, -1.651816, 0.009738),
)
CALIBRATION_SEGMENT_STEPS = (75, 25, 25, 25, 25, 25, 25, 25, 50)
LEFT_FINGER_GEOMS = frozenset({"vx300s_left/10_left_gripper_finger", "vx300s_left/10_right_gripper_finger"})
RIGHT_FINGER_GEOMS = frozenset({"vx300s_right/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger"})
CANONICAL_LAYOUTS = {
    "pi_left": (Pose("pi", -0.15, 0.47, SPAWN_HEIGHT, 0.0),),
    "pi_right": (Pose("pi", 0.15, 0.47, SPAWN_HEIGHT, 0.0),),
    "P_left": (
        Pose("P", -0.15, 0.47, SPAWN_HEIGHT, -math.pi / 2),
        Pose("I", 0.15, 0.35, SPAWN_HEIGHT, 0.0),
    ),
    "I_right": (
        Pose("P", -0.15, 0.35, SPAWN_HEIGHT, 0.0),
        Pose("I", 0.15, 0.47, SPAWN_HEIGHT, math.pi / 2),
    ),
}

SCENARIOS = {
    "transfer_cube": ScenarioSpec("transfer_cube", "gym_aloha/AlohaTransferCube-v0", None, "stock", None),
    "push_pi_single": ScenarioSpec(
        "push_pi_single",
        "pi_robotics/PushPiSingleArm-v0",
        "pi",
        "left",
        "A PI-shaped block and a dotted target outline are on the table. Using only the left arm, push the block "
        "until it covers and aligns with the target outline.",
    ),
    "push_pi_dual": ScenarioSpec(
        "push_pi_dual",
        "pi_robotics/PushPiBimanual-v0",
        "pi",
        "both",
        "A PI-shaped block and a dotted target outline are on the table. Using both arms, push the block until it "
        "covers and aligns with the target outline.",
    ),
    "push_letters_single": ScenarioSpec(
        "push_letters_single",
        "pi_robotics/PushLettersSingleArm-v0",
        "letters",
        "left",
        "P and I blocks and their dotted target outlines are on the table. Using only the left arm, push each "
        "block until it covers and aligns with its matching target outline.",
    ),
    "push_letters_dual": ScenarioSpec(
        "push_letters_dual",
        "pi_robotics/PushLettersBimanual-v0",
        "letters",
        "both",
        "P and I blocks and their dotted target outlines are on the table. Using both arms, push each block until "
        "it covers and aligns with its matching target outline.",
    ),
}
CUSTOM_SCENARIOS = tuple(key for key, value in SCENARIOS.items() if value.is_custom)
TASK_TO_SCENARIO = {spec.gym_id: spec.key for spec in SCENARIOS.values()}


def get_scenario(key: str) -> ScenarioSpec:
    try:
        return SCENARIOS[key]
    except KeyError as error:
        raise ValueError(f"ALOHA_SCENARIO must be one of: {', '.join(SCENARIOS)}") from error


def body_descriptors(object_kind: str) -> tuple[BodyDescriptor, ...]:
    try:
        return BODIES[object_kind]
    except KeyError as error:
        raise ValueError(f"unknown Push-PI object kind: {object_kind}") from error


def target_outline_points(body: BodyDescriptor) -> tuple[tuple[float, float], ...]:
    points = set()
    for part in body.parts:
        left, right = part.x - part.half_x, part.x + part.half_x
        bottom, top = part.y - part.half_y, part.y + part.half_y
        edges = (
            ((left, bottom), (left, top), (-1, 0)),
            ((right, bottom), (right, top), (1, 0)),
            ((left, bottom), (right, bottom), (0, -1)),
            ((left, top), (right, top), (0, 1)),
        )
        for start, end, normal in edges:
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            intervals = max(1, math.ceil(length / TARGET_DOT_SPACING))
            for index in range(intervals + 1):
                fraction = index / intervals
                x = start[0] + fraction * (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                probe_x = x + normal[0] * 1e-7
                probe_y = y + normal[1] * 1e-7
                if any(
                    other.x - other.half_x <= probe_x <= other.x + other.half_x
                    and other.y - other.half_y <= probe_y <= other.y + other.half_y
                    for other in body.parts
                    if other is not part
                ):
                    continue
                points.add((round(x, 9), round(y, 9)))
    return tuple(sorted(points))


def _separated(pose: Pose, body: BodyDescriptor, accepted: list[tuple[Pose, BodyDescriptor]]) -> bool:
    for other_pose, other_body in accepted:
        if math.hypot(pose.x - other_pose.x, pose.y - other_pose.y) < (
            body.footprint_radius + other_body.footprint_radius + BODY_CLEARANCE_METERS
        ):
            return False
    for target in body_descriptors("letters") if body.name in {"P", "I"} else (PI_BODY,):
        if (
            math.hypot(pose.x - target.target_x, pose.y - target.target_y)
            < body.footprint_radius + TARGET_CLEARANCE_METERS
        ):
            return False
    return True


def sample_layout(object_kind: str, seed: int, *, max_attempts: int = MAX_LAYOUT_ATTEMPTS) -> tuple[Pose, ...]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("layout seed must be an unsigned 32-bit integer")
    if max_attempts < 1:
        raise ValueError("layout attempts must be positive")
    rng = np.random.default_rng(seed)
    accepted: list[tuple[Pose, BodyDescriptor]] = []
    low_x, high_x, low_y, high_y = SPAWN_BOUNDS
    for body in body_descriptors(object_kind):
        radius = body.footprint_radius
        if low_x + radius >= high_x - radius or low_y + radius >= high_y - radius:
            raise ValueError(f"layout sampling exhausted for {object_kind} seed {seed}")
        for _ in range(max_attempts):
            pose = Pose(
                body.name,
                float(rng.uniform(low_x + radius, high_x - radius)),
                float(rng.uniform(low_y + radius, high_y - radius)),
                SPAWN_HEIGHT,
                float(rng.uniform(-SPAWN_YAW_RADIANS, SPAWN_YAW_RADIANS)),
            )
            if _separated(pose, body, accepted):
                accepted.append((pose, body))
                break
        else:
            raise ValueError(f"layout sampling exhausted for {object_kind} seed {seed}")
    return tuple(pose for pose, _ in accepted)


def effective_layout_seed(requested_seed: int, attempt: int) -> int:
    if isinstance(requested_seed, bool) or not isinstance(requested_seed, int) or not 0 <= requested_seed < 2**32:
        raise ValueError("requested layout seed must be an unsigned 32-bit integer")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 0 <= attempt < RESET_LAYOUT_ATTEMPTS:
        raise ValueError("layout attempt is out of range")
    return (requested_seed + attempt * RESET_SEED_STRIDE) % 2**32


def quaternion_euler(state: BodyState) -> tuple[float, float, float]:
    norm = math.sqrt(state.qw**2 + state.qx**2 + state.qy**2 + state.qz**2)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("body quaternion must be finite and nonzero")
    w, x, y, z = (state.qw / norm, state.qx / norm, state.qy / norm, state.qz / norm)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrapped_angle(value: float, period: float = 2 * math.pi) -> float:
    return (value + period / 2) % period - period / 2


def _within(value: float, limit: float) -> bool:
    return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-12)


def _transformed_corners(body: BodyDescriptor, state: BodyState) -> Iterable[tuple[float, float]]:
    _, _, yaw = quaternion_euler(state)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    for part in body.parts:
        for local_x in (part.x - part.half_x, part.x + part.half_x):
            for local_y in (part.y - part.half_y, part.y + part.half_y):
                yield (
                    state.x + cosine * local_x - sine * local_y,
                    state.y + sine * local_x + cosine * local_y,
                )


@cache
def _union_tiles(parts: tuple[Part, ...]) -> tuple[Part, ...]:
    if not parts or any(
        not all(math.isfinite(value) for value in (part.x, part.y, part.half_x, part.half_y))
        or part.half_x <= 0
        or part.half_y <= 0
        for part in parts
    ):
        raise ValueError("body footprint parts must be finite, positive rectangles")
    edges = sorted({part.x - part.half_x for part in parts} | {part.x + part.half_x for part in parts})
    tiles = []
    for left, right in pairwise(edges):
        middle = (left + right) / 2
        intervals = sorted(
            (part.y - part.half_y, part.y + part.half_y)
            for part in parts
            if part.x - part.half_x < middle < part.x + part.half_x
        )
        merged: list[list[float]] = []
        for low, high in intervals:
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        tiles.extend(
            Part((left + right) / 2, (low + high) / 2, (right - left) / 2, (high - low) / 2) for low, high in merged
        )
    if not tiles:
        raise ValueError("body footprint area must be positive")
    return tuple(tiles)


def _part_polygon(part: Part, x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (x + cosine * px - sine * py, y + sine * px + cosine * py)
        for px, py in (
            (part.x - part.half_x, part.y - part.half_y),
            (part.x + part.half_x, part.y - part.half_y),
            (part.x + part.half_x, part.y + part.half_y),
            (part.x - part.half_x, part.y + part.half_y),
        )
    ]


def _clip_convex(subject: list[tuple[float, float]], clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    output = subject
    for first, second in zip(clip, clip[1:] + clip[:1], strict=True):
        source, output = output, []
        if not source:
            break

        def side(
            point: tuple[float, float], first: tuple[float, float] = first, second: tuple[float, float] = second
        ) -> float:
            return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])

        previous = source[-1]
        previous_side = side(previous)
        for current in source:
            current_side = side(current)
            if current_side >= -1e-12:
                if previous_side < -1e-12:
                    fraction = previous_side / (previous_side - current_side)
                    output.append(
                        (
                            previous[0] + fraction * (current[0] - previous[0]),
                            previous[1] + fraction * (current[1] - previous[1]),
                        )
                    )
                output.append(current)
            elif previous_side >= -1e-12:
                fraction = previous_side / (previous_side - current_side)
                output.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
            previous, previous_side = current, current_side
    return output


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return (
        abs(
            math.fsum(
                first[0] * second[1] - second[0] * first[1]
                for first, second in zip(points, points[1:] + points[:1], strict=True)
            )
        )
        / 2
    )


def footprint_overlap_coverage(body: BodyDescriptor, state: BodyState) -> float:
    """Return exact planar overlap with this body's matching target, as a fraction."""
    if state.name != body.name or not all(
        math.isfinite(value) for value in (state.x, state.y, body.target_x, body.target_y, body.target_yaw)
    ):
        raise ValueError("body footprint state is invalid")
    _, _, yaw = quaternion_euler(state)
    tiles = _union_tiles(body.parts)
    target_area = math.fsum(4 * tile.half_x * tile.half_y for tile in tiles)
    actual = [_part_polygon(tile, state.x, state.y, yaw) for tile in tiles]
    target = [_part_polygon(tile, body.target_x, body.target_y, body.target_yaw) for tile in tiles]
    overlap = math.fsum(_polygon_area(_clip_convex(source, goal)) for source in actual for goal in target)
    tolerance = max(1e-12, target_area * 1e-9)
    if not math.isfinite(overlap) or overlap < -tolerance or overlap > target_area + tolerance:
        raise ValueError("body footprint overlap is outside its target area")
    return min(1.0, max(0.0, overlap / target_area))


def advance_outcome(
    bodies: tuple[BodyDescriptor, ...],
    states: Mapping[str, BodyState],
    rest_heights: Mapping[str, float],
    previous: OutcomeState | None = None,
) -> tuple[OutcomeState, dict[str, float]]:
    previous = previous or OutcomeState()
    if set(states) != {body.name for body in bodies} or set(rest_heights) != set(states):
        raise ValueError("outcome body state does not match its descriptor")
    all_at_goal = True
    lifted = previous.lifted_ever
    fallen = previous.fallen
    off_table = previous.off_table
    metrics: dict[str, float] = {}
    coverage_area = total_area = 0.0
    min_x, max_x, min_y, max_y = TABLE_BOUNDS
    for index, body in enumerate(bodies):
        state = states[body.name]
        com_z = state.z if state.com_z is None else state.com_z
        values = (state.x, state.y, state.z, state.qw, state.qx, state.qy, state.qz, com_z, rest_heights[body.name])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("outcome body state must be finite")
        roll, pitch, yaw = quaternion_euler(state)
        xy_error = math.hypot(state.x - body.target_x, state.y - body.target_y)
        yaw_error = abs(wrapped_angle(yaw - body.target_yaw, body.yaw_period))
        height_error = abs(com_z - rest_heights[body.name])
        target_area = math.fsum(4 * tile.half_x * tile.half_y for tile in _union_tiles(body.parts))
        coverage = footprint_overlap_coverage(body, state)
        coverage_area += coverage * target_area
        total_area += target_area
        metrics.update(
            {
                f"body_{index}_xy_error": xy_error,
                f"body_{index}_yaw_error": yaw_error,
                f"body_{index}_roll": abs(roll),
                f"body_{index}_pitch": abs(pitch),
                f"body_{index}_height_error": height_error,
                f"body_{index}_target_area_coverage": coverage,
            }
        )
        lift_delta = com_z - rest_heights[body.name]
        lifted = lifted or (lift_delta > LIFT_METERS and not _within(lift_delta, LIFT_METERS))
        fallen = fallen or any(
            value > FALL_RADIANS and not _within(value, FALL_RADIANS) for value in (abs(roll), abs(pitch))
        )
        off_table = off_table or any(
            x < min_x or x > max_x or y < min_y or y > max_y for x, y in _transformed_corners(body, state)
        )
        all_at_goal = all_at_goal and (
            _within(xy_error, SUCCESS_XY_METERS)
            and _within(yaw_error, SUCCESS_YAW_RADIANS)
            and _within(abs(roll), SUCCESS_TILT_RADIANS)
            and _within(abs(pitch), SUCCESS_TILT_RADIANS)
            and _within(height_error, SUCCESS_HEIGHT_METERS)
        )
    metrics["target_area_coverage"] = coverage_area / total_area
    held = previous.held_steps + 1 if all_at_goal and not lifted and not fallen and not off_table else 0
    success = previous.success or held >= SUCCESS_HOLD_STEPS
    reason = "off_table" if off_table else "fallen" if fallen else "success" if success else "running"
    return OutcomeState(held, lifted, success, off_table, fallen, reason), metrics


def update_participation(
    contacts: Iterable[tuple[str, str]],
    geom_to_body: Mapping[str, str],
    previous: Participation | None = None,
) -> Participation:
    previous = previous or Participation()
    left_bodies: set[str] = set()
    right_bodies: set[str] = set()
    for first, second in contacts:
        for movable, finger in ((first, second), (second, first)):
            body = geom_to_body.get(movable)
            if body is None:
                continue
            if finger in LEFT_FINGER_GEOMS:
                left_bodies.add(body)
            if finger in RIGHT_FINGER_GEOMS:
                right_bodies.add(body)
    return Participation(
        previous.left_contact_ever or bool(left_bodies),
        previous.right_contact_ever or bool(right_bodies),
        previous.interference_ever or bool(left_bodies & right_bodies),
    )


def layout_hash(poses: Iterable[Pose]) -> str:
    payload = [[pose.name, *pose.vector()] for pose in poses]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def scene_hash(xml: bytes, assets: Mapping[str, bytes], object_kind: str) -> str:
    digest = hashlib.sha256()
    digest.update(object_kind.encode())
    digest.update(b"\0")
    digest.update(xml)
    for name, content in sorted(assets.items()):
        digest.update(b"\0")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def descriptor_payload() -> dict[str, object]:
    return {
        "version": DESCRIPTOR_VERSION,
        "action_mode": "identity-14d",
        "tabletop_sha256": TABLETOP_SHA256,
        "table_bounds": TABLE_BOUNDS,
        "spawn_bounds": SPAWN_BOUNDS,
        "object_half_height": OBJECT_HALF_HEIGHT,
        "spawn_height": SPAWN_HEIGHT,
        "layout_attempts": MAX_LAYOUT_ATTEMPTS,
        "reset_layout_attempts": RESET_LAYOUT_ATTEMPTS,
        "reset_seed_stride": RESET_SEED_STRIDE,
        "spawn_yaw_radians": SPAWN_YAW_RADIANS,
        "body_clearance_meters": BODY_CLEARANCE_METERS,
        "target_clearance_meters": TARGET_CLEARANCE_METERS,
        "pusher_position": PUSHER_POSITION,
        "puppet_gripper_position_close": PUPPET_GRIPPER_POSITION_CLOSE,
        "puppet_gripper_position_open": PUPPET_GRIPPER_POSITION_OPEN,
        "pusher_physical_position": PUSHER_PHYSICAL_POSITION,
        "settle_steps": SETTLE_STEPS,
        "free_joint_frictionloss": FREE_JOINT_FRICTIONLOSS,
        "geom_density": GEOM_DENSITY,
        "geom_friction": GEOM_FRICTION,
        "geom_condim": GEOM_CONDIM,
        "geom_solimp": GEOM_SOLIMP,
        "geom_solref": GEOM_SOLREF,
        "visual_geom_group": VISUAL_GEOM_GROUP,
        "visual_geom_mass": VISUAL_GEOM_MASS,
        "visual_contact_bits": VISUAL_CONTACT_BITS,
        "collision_geom_group": COLLISION_GEOM_GROUP,
        "collision_contact_bits": COLLISION_CONTACT_BITS,
        "target_half_height": TARGET_HALF_HEIGHT,
        "target_dot_radius": TARGET_DOT_RADIUS,
        "target_dot_spacing": TARGET_DOT_SPACING,
        "target_contact_bits": TARGET_CONTACT_BITS,
        "display_every_steps": DISPLAY_EVERY_STEPS,
        "parked_joint_tolerance": PARKED_JOINT_TOLERANCE,
        "reset_settle_xy_meters": RESET_SETTLE_XY_METERS,
        "reset_settle_yaw_radians": RESET_SETTLE_YAW_RADIANS,
        "reset_tilt_radians": RESET_TILT_RADIANS,
        "success_xy_meters": SUCCESS_XY_METERS,
        "success_yaw_radians": SUCCESS_YAW_RADIANS,
        "success_tilt_radians": SUCCESS_TILT_RADIANS,
        "success_height_meters": SUCCESS_HEIGHT_METERS,
        "success_hold_steps": SUCCESS_HOLD_STEPS,
        "lift_meters": LIFT_METERS,
        "fall_radians": FALL_RADIANS,
        "calibration_min_push_meters": CALIBRATION_MIN_PUSH_METERS,
        "calibration_max_height_error_meters": CALIBRATION_MAX_HEIGHT_ERROR_METERS,
        "minimum_visible_pixels": MIN_VISIBLE_PIXELS,
        "color_mask_rules": COLOR_MASK_RULES,
        "calibration_segment_steps": CALIBRATION_SEGMENT_STEPS,
        "left_finger_geoms": sorted(LEFT_FINGER_GEOMS),
        "right_finger_geoms": sorted(RIGHT_FINGER_GEOMS),
        "left_push_waypoints": LEFT_PUSH_WAYPOINTS,
        "right_push_waypoints": RIGHT_PUSH_WAYPOINTS,
        "canonical_layouts": {
            name: [[pose.name, *pose.vector()] for pose in poses] for name, poses in CANONICAL_LAYOUTS.items()
        },
        "bodies": {
            kind: [
                {
                    "name": body.name,
                    "parts": [[part.x, part.y, part.half_x, part.half_y] for part in body.parts],
                    "target": [body.target_x, body.target_y, body.target_yaw],
                    "rgba": body.rgba,
                    "yaw_period": body.yaw_period,
                }
                for body in bodies
            ]
            for kind, bodies in BODIES.items()
        },
        "scenarios": {
            key: {
                "gym_id": scenario.gym_id,
                "object_kind": scenario.object_kind,
                "arm_mode": scenario.arm_mode,
                "prompt": scenario.prompt,
            }
            for key, scenario in SCENARIOS.items()
        },
    }


def descriptor_sha256() -> str:
    payload = json.dumps(descriptor_payload(), allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
