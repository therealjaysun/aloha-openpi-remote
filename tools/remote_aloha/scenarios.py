from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
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
SPAWN_BOUNDS = (-0.25, 0.25, 0.32, 0.58)
OBJECT_HALF_HEIGHT = 0.012
SPAWN_HEIGHT = 0.04
MAX_LAYOUT_ATTEMPTS = 256
SUCCESS_XY_METERS = 0.03
SUCCESS_YAW_RADIANS = math.radians(15)
SUCCESS_TILT_RADIANS = math.radians(10)
SUCCESS_HEIGHT_METERS = 0.005
SUCCESS_HOLD_STEPS = 5
LIFT_METERS = 0.01
FALL_RADIANS = math.radians(30)


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

SCENARIOS = {
    "transfer_cube": ScenarioSpec("transfer_cube", "gym_aloha/AlohaTransferCube-v0", None, "stock", None),
    "push_pi_single": ScenarioSpec(
        "push_pi_single",
        "pi_robotics/PushPiSingleArm-v0",
        "pi",
        "left",
        "Push the pi-shaped block onto its matching target.",
    ),
    "push_pi_dual": ScenarioSpec(
        "push_pi_dual",
        "pi_robotics/PushPiBimanual-v0",
        "pi",
        "both",
        "Push the pi-shaped block onto its matching target.",
    ),
    "push_letters_single": ScenarioSpec(
        "push_letters_single",
        "pi_robotics/PushLettersSingleArm-v0",
        "letters",
        "left",
        "Push the P and I blocks onto their matching targets.",
    ),
    "push_letters_dual": ScenarioSpec(
        "push_letters_dual",
        "pi_robotics/PushLettersBimanual-v0",
        "letters",
        "both",
        "Push the P and I blocks onto their matching targets.",
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
        raise ValueError(f"unknown Push-pi object kind: {object_kind}") from error


def _separated(pose: Pose, body: BodyDescriptor, accepted: list[tuple[Pose, BodyDescriptor]]) -> bool:
    for other_pose, other_body in accepted:
        if math.hypot(pose.x - other_pose.x, pose.y - other_pose.y) < (
            body.footprint_radius + other_body.footprint_radius + 0.02
        ):
            return False
    for target in body_descriptors("letters") if body.name in {"P", "I"} else (PI_BODY,):
        if math.hypot(pose.x - target.target_x, pose.y - target.target_y) < body.footprint_radius + 0.04:
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
                float(rng.uniform(-math.pi / 3, math.pi / 3)),
            )
            if _separated(pose, body, accepted):
                accepted.append((pose, body))
                break
        else:
            raise ValueError(f"layout sampling exhausted for {object_kind} seed {seed}")
    return tuple(pose for pose, _ in accepted)


def project_action(action: object, scenario: ScenarioSpec, home: object | None = None) -> np.ndarray:
    command = np.asarray(action, dtype=np.float64)
    if command.shape != (JOINT_COUNT,) or not np.isfinite(command).all():
        raise ValueError("scenario action must be a finite 14-vector")
    command = command.copy()
    if not scenario.is_custom:
        return command
    reference = np.asarray(home, dtype=np.float64)
    if reference.shape != (JOINT_COUNT,) or not np.isfinite(reference).all():
        raise ValueError("scenario home state must be a finite 14-vector")
    if scenario.arm_mode == "left":
        command[7:13] = reference[7:13]
    elif scenario.arm_mode != "both":
        raise ValueError("custom scenario arm mode is invalid")
    command[6] = PUSHER_POSITION
    command[13] = PUSHER_POSITION
    return command


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
    fallen = False
    off_table = False
    metrics: dict[str, float] = {}
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
        metrics.update(
            {
                f"body_{index}_xy_error": xy_error,
                f"body_{index}_yaw_error": yaw_error,
                f"body_{index}_roll": abs(roll),
                f"body_{index}_pitch": abs(pitch),
                f"body_{index}_height_error": height_error,
            }
        )
        lifted = lifted or com_z - rest_heights[body.name] > LIFT_METERS
        fallen = fallen or abs(roll) > FALL_RADIANS or abs(pitch) > FALL_RADIANS
        off_table = off_table or any(
            x < min_x or x > max_x or y < min_y or y > max_y for x, y in _transformed_corners(body, state)
        )
        all_at_goal = all_at_goal and (
            xy_error <= SUCCESS_XY_METERS
            and yaw_error <= SUCCESS_YAW_RADIANS
            and abs(roll) <= SUCCESS_TILT_RADIANS
            and abs(pitch) <= SUCCESS_TILT_RADIANS
            and height_error <= SUCCESS_HEIGHT_METERS
        )
    held = previous.held_steps + 1 if all_at_goal and not lifted and not fallen and not off_table else 0
    success = held >= SUCCESS_HOLD_STEPS
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
            if finger.startswith("vx300s_left/") and finger.endswith("_gripper_finger"):
                left_bodies.add(body)
            if finger.startswith("vx300s_right/") and finger.endswith("_gripper_finger"):
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
