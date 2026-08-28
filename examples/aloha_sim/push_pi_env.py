from __future__ import annotations

from collections.abc import Mapping
import hashlib
import xml.etree.ElementTree as ET

from dm_control import mujoco
from dm_control.rl import control
from gym_aloha.constants import ASSETS_DIR
from gym_aloha.constants import DT
from gym_aloha.constants import START_ARM_POSE
from gym_aloha.env import AlohaEnv
from gym_aloha.tasks.sim import BimanualViperXTask
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.envs.registration import registry
import numpy as np

from tools.remote_aloha.scenarios import CUSTOM_SCENARIOS
from tools.remote_aloha.scenarios import OBJECT_HALF_HEIGHT
from tools.remote_aloha.scenarios import PUSHER_PHYSICAL_POSITION
from tools.remote_aloha.scenarios import PUSHER_POSITION
from tools.remote_aloha.scenarios import TABLETOP_SHA256
from tools.remote_aloha.scenarios import BodyDescriptor
from tools.remote_aloha.scenarios import BodyState
from tools.remote_aloha.scenarios import OutcomeState
from tools.remote_aloha.scenarios import Participation
from tools.remote_aloha.scenarios import Pose
from tools.remote_aloha.scenarios import advance_outcome
from tools.remote_aloha.scenarios import body_descriptors
from tools.remote_aloha.scenarios import get_scenario
from tools.remote_aloha.scenarios import layout_hash
from tools.remote_aloha.scenarios import quaternion_euler
from tools.remote_aloha.scenarios import sample_layout
from tools.remote_aloha.scenarios import scene_hash
from tools.remote_aloha.scenarios import update_participation

_BASE_XML = "bimanual_viperx_transfer_cube.xml"
_SETTLE_STEPS = 200
_GEOM_DENSITY = 350
_FRICTION = "1 0.005 0.0001"
_ROBOT_QPOS_COUNT = 16


def _asset_bytes() -> dict[str, bytes]:
    table = ASSETS_DIR / "tabletop.stl"
    if hashlib.sha256(table.read_bytes()).hexdigest() != TABLETOP_SHA256:
        raise ValueError("pinned gym-aloha tabletop asset hash changed")
    return {
        path.name: path.read_bytes()
        for path in sorted(ASSETS_DIR.iterdir())
        if path.is_file() and path.name != _BASE_XML
    }


def _numbers(*values: float) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _add_target(worldbody: ET.Element, body: BodyDescriptor) -> None:
    target = ET.SubElement(
        worldbody,
        "body",
        name=f"push_pi/target_{body.name}",
        pos=_numbers(body.target_x, body.target_y, 0),
        euler=_numbers(0, 0, body.target_yaw),
    )
    for index, part in enumerate(body.parts):
        red, green, blue, _ = body.rgba
        ET.SubElement(
            target,
            "geom",
            name=f"push_pi/target_{body.name}_{index}",
            type="box",
            pos=_numbers(part.x, part.y, 0.001),
            size=_numbers(part.half_x, part.half_y, 0.001),
            rgba=_numbers(red, green, blue, 0.32),
            contype="0",
            conaffinity="0",
        )


def _add_movable(worldbody: ET.Element, body: BodyDescriptor) -> None:
    movable = ET.SubElement(worldbody, "body", name=f"push_pi/{body.name}")
    ET.SubElement(
        movable,
        "joint",
        name=f"push_pi/{body.name}_joint",
        type="free",
        frictionloss="0.01",
    )
    for index, part in enumerate(body.parts):
        ET.SubElement(
            movable,
            "geom",
            name=f"push_pi/{body.name}_{index}",
            type="box",
            pos=_numbers(part.x, part.y, 0),
            size=_numbers(part.half_x, part.half_y, OBJECT_HALF_HEIGHT),
            rgba=_numbers(*body.rgba),
            density=str(_GEOM_DENSITY),
            group="4",
            condim="4",
            solimp="2 1 0.01",
            solref="0.01 1",
            friction=_FRICTION,
        )


def build_scene(object_kind: str) -> tuple[bytes, Mapping[str, bytes], str]:
    bodies = body_descriptors(object_kind)
    root = ET.fromstring((ASSETS_DIR / _BASE_XML).read_bytes())
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("pinned ALOHA XML has no worldbody")
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") == "box":
            worldbody.remove(child)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    for body in bodies:
        _add_target(worldbody, body)
        _add_movable(worldbody, body)
    xml = ET.tostring(root, encoding="utf-8")
    assets = _asset_bytes()
    return xml, assets, scene_hash(xml, assets, object_kind)


class PushPiTask(BimanualViperXTask):
    def __init__(self, scenario: str, scene_id: str) -> None:
        super().__init__()
        self.spec = get_scenario(scenario)
        if not self.spec.is_custom or self.spec.object_kind is None:
            raise ValueError("PushPiTask requires a custom scenario")
        self.bodies = body_descriptors(self.spec.object_kind)
        self.scene_hash = scene_id
        self.sampled: tuple[Pose, ...] = ()
        self.settled: tuple[Pose, ...] = ()
        self.layout_hash = ""
        self.home_joint_positions = np.zeros(14, dtype=np.float64)
        self.rest_heights: dict[str, float] = {}
        self.outcome = OutcomeState()
        self.participation = Participation()
        self.left_joint_travel = 0.0
        self.right_joint_travel = 0.0
        self._previous_joints: np.ndarray | None = None
        self._metrics: dict[str, float] = {}
        self._geom_to_body = {
            f"push_pi/{body.name}_{index}": body.name for body in self.bodies for index, _ in enumerate(body.parts)
        }

    def set_layout(self, poses: tuple[Pose, ...]) -> None:
        if tuple(pose.name for pose in poses) != tuple(body.name for body in self.bodies):
            raise ValueError("layout does not match the Push-pi bodies")
        self.sampled = poses

    def initialize_episode(self, physics) -> None:
        if not self.sampled:
            raise ValueError("Push-pi layout must be set before reset")
        arm_pose = np.asarray(START_ARM_POSE, dtype=np.float64).copy()
        arm_pose[6] = PUSHER_PHYSICAL_POSITION
        arm_pose[7] = -PUSHER_PHYSICAL_POSITION
        arm_pose[14] = PUSHER_PHYSICAL_POSITION
        arm_pose[15] = -PUSHER_PHYSICAL_POSITION
        with physics.reset_context():
            physics.data.qpos[:_ROBOT_QPOS_COUNT] = arm_pose
            np.copyto(physics.data.ctrl, arm_pose)
            for pose in self.sampled:
                physics.named.data.qpos[f"push_pi/{pose.name}_joint"] = pose.vector()
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics) -> np.ndarray:
        return physics.data.qpos.copy()[_ROBOT_QPOS_COUNT:]

    def _body_states(self, physics) -> dict[str, BodyState]:
        states = {}
        for body in self.bodies:
            x, y, z = (float(value) for value in physics.named.data.xpos[f"push_pi/{body.name}"])
            qw, qx, qy, qz = (float(value) for value in physics.named.data.xquat[f"push_pi/{body.name}"])
            com_z = float(physics.named.data.xipos[f"push_pi/{body.name}"][2])
            states[body.name] = BodyState(body.name, x, y, z, qw, qx, qy, qz, com_z)
        return states

    def _contacts(self, physics) -> list[tuple[str, str]]:
        result = []
        for index in range(physics.data.ncon):
            contact = physics.data.contact[index]
            first = physics.model.id2name(contact.geom1, "geom")
            second = physics.model.id2name(contact.geom2, "geom")
            if first is not None and second is not None:
                result.append((first, second))
        return result

    def capture_reset(self, physics) -> None:
        states = self._body_states(physics)
        self.rest_heights = {name: float(state.com_z) for name, state in states.items()}
        self.settled = tuple(
            Pose(name, state.x, state.y, state.z, quaternion_euler(state)[2]) for name, state in states.items()
        )
        self.layout_hash = layout_hash(self.settled)
        self.home_joint_positions = self.get_qpos(physics).astype(np.float64, copy=True)
        self.outcome = OutcomeState()
        self.participation = Participation()
        self.left_joint_travel = self.right_joint_travel = 0.0
        self._previous_joints = self.home_joint_positions.copy()
        self._metrics = {}

    def get_reward(self, physics) -> float:
        if not self.rest_heights:
            return 0.0
        joints = self.get_qpos(physics)
        if self._previous_joints is not None:
            self.left_joint_travel += float(np.abs(joints[:6] - self._previous_joints[:6]).sum())
            self.right_joint_travel += float(np.abs(joints[7:13] - self._previous_joints[7:13]).sum())
        self._previous_joints = joints
        self.participation = update_participation(self._contacts(physics), self._geom_to_body, self.participation)
        self.outcome, self._metrics = advance_outcome(
            self.bodies,
            self._body_states(physics),
            self.rest_heights,
            self.outcome,
        )
        return float(self.outcome.success)

    def info(self) -> dict[str, object]:
        return {
            "is_success": self.outcome.success,
            "scenario": self.spec.key,
            "scene_hash": self.scene_hash,
            "layout_hash": self.layout_hash,
            "body_count": len(self.bodies),
            "held_steps": self.outcome.held_steps,
            "lifted_ever": self.outcome.lifted_ever,
            "off_table": self.outcome.off_table,
            "fallen": self.outcome.fallen,
            "terminal_reason": self.outcome.terminal_reason,
            "left_contact_ever": self.participation.left_contact_ever,
            "right_contact_ever": self.participation.right_contact_ever,
            "both_arms_participated": self.participation.both_arms_participated,
            "interference_ever": self.participation.interference_ever,
            "left_joint_travel": self.left_joint_travel,
            "right_joint_travel": self.right_joint_travel,
            **self._metrics,
        }

    def reset_info(self) -> dict[str, object]:
        return {
            **self.info(),
            "sampled_poses": [list(pose.vector()) for pose in self.sampled],
            "settled_poses": [list(pose.vector()) for pose in self.settled],
            "home_joint_positions": self.home_joint_positions.tolist(),
            "pusher_position": PUSHER_POSITION,
            "pusher_physical_position": PUSHER_PHYSICAL_POSITION,
        }


class PushPiEnv(AlohaEnv):
    def __init__(self, scenario: str, **kwargs: object) -> None:
        self.scenario = get_scenario(scenario)
        if not self.scenario.is_custom or self.scenario.object_kind is None:
            raise ValueError("PushPiEnv requires a custom scenario")
        self._scene_xml, self._scene_assets, self.scene_hash = build_scene(self.scenario.object_kind)
        self._push_task: PushPiTask
        self._step_count = 0
        super().__init__(task=scenario, **kwargs)

    def _make_env_task(self, task_name: str):
        del task_name
        physics = mujoco.Physics.from_xml_string(self._scene_xml.decode(), assets=self._scene_assets)
        self._push_task = PushPiTask(self.scenario.key, self.scene_hash)
        return control.Environment(
            physics,
            self._push_task,
            float("inf"),
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )

    @property
    def home_joint_positions(self) -> np.ndarray:
        return self._push_task.home_joint_positions.copy()

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        gym.Env.reset(self, seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(2**32 - 1))
        self._env.task.random.seed(actual_seed)
        self._push_task.set_layout(sample_layout(str(self.scenario.object_kind), actual_seed))
        self._env.reset()
        self._env.physics.step(nstep=_SETTLE_STEPS)
        self._env.physics.forward()
        self._push_task.capture_reset(self._env.physics)
        raw_observation = self._push_task.get_observation(self._env.physics)
        self._step_count = 0
        return self._format_raw_obs(raw_observation), self._push_task.reset_info()

    def step(self, action: object):
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError("Push-pi action must be a finite 14-vector")
        _, reward, _, raw_observation = self._env.step(command)
        self._step_count += 1
        info = self._push_task.info()
        terminated = info["terminal_reason"] in {"success", "off_table", "fallen"}
        truncated = self._step_count >= 300 and not terminated
        if truncated:
            info = {**info, "terminal_reason": "time_limit"}
        return self._format_raw_obs(raw_observation), float(reward), bool(terminated), bool(truncated), info


for _scenario in CUSTOM_SCENARIOS:
    _spec = get_scenario(_scenario)
    if _spec.gym_id not in registry:
        register(
            id=_spec.gym_id,
            entry_point="examples.aloha_sim.push_pi_env:PushPiEnv",
            max_episode_steps=300,
            nondeterministic=True,
            kwargs={"scenario": _scenario},
        )
