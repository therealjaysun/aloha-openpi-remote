from __future__ import annotations

import dataclasses
import pathlib
import re
import tempfile

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.base_object import register_object
from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from libero.libero.envs.bddl_base_domain import register_problem
from libero.libero.envs.regions import REGION_SAMPLERS
from libero.libero.envs.regions import TableRegionSampler
from robosuite.models.objects import CompositeObject

from tools.remote_aloha.scenarios import GEOM_CONDIM
from tools.remote_aloha.scenarios import GEOM_DENSITY
from tools.remote_aloha.scenarios import GEOM_FRICTION
from tools.remote_aloha.scenarios import GEOM_SOLIMP
from tools.remote_aloha.scenarios import GEOM_SOLREF
from tools.remote_aloha.scenarios import I_BODY
from tools.remote_aloha.scenarios import OBJECT_HALF_HEIGHT
from tools.remote_aloha.scenarios import P_BODY
from tools.remote_aloha.scenarios import PI_BODY
from tools.remote_aloha.scenarios import TARGET_DOT_RADIUS
from tools.remote_aloha.scenarios import TARGET_HALF_HEIGHT
from tools.remote_aloha.scenarios import BodyDescriptor
from tools.remote_aloha.scenarios import BodyState
from tools.remote_aloha.scenarios import footprint_overlap_coverage
from tools.remote_aloha.scenarios import get_scenario
from tools.remote_aloha.scenarios import quaternion_euler
from tools.remote_aloha.scenarios import target_outline_points

CONTROL_HZ = 20
_TABLE_HALF_HEIGHT = 0.05
SCENARIOS = {
    "push_pi": (
        "PushPiLibero",
        ((PI_BODY, "pi_1", "pi_target_1"),),
        get_scenario("push_pi_single").prompt.replace("Using only the left arm", "Using the LIBERO arm"),
    ),
    "push_p_i": (
        "PushLettersLibero",
        ((P_BODY, "p_1", "p_target_1"), (I_BODY, "i_1", "i_target_1")),
        get_scenario("push_letters_single").prompt.replace("Using only the left arm", "Using the LIBERO arm"),
    ),
}
REGION_SAMPLERS.update({problem.lower(): {"table": TableRegionSampler} for problem, _, _ in SCENARIOS.values()})

_PROPERTIES = {
    "articulation": {"default_open_ranges": [], "default_close_ranges": []},
    "vis_site_names": {},
}


class _Block(CompositeObject):
    descriptor: BodyDescriptor

    def __init__(self, name="block", joints="default"):
        body = self.descriptor
        half_x = max(abs(part.x) + part.half_x for part in body.parts)
        half_y = max(abs(part.y) + part.half_y for part in body.parts)
        super().__init__(
            name=name,
            total_size=(half_x, half_y, OBJECT_HALF_HEIGHT),
            geom_types=["box"] * len(body.parts),
            geom_sizes=[(part.half_x, part.half_y, OBJECT_HALF_HEIGHT) for part in body.parts],
            geom_locations=[(part.x, part.y, 0.0) for part in body.parts],
            geom_names=[f"part_{index}" for index in range(len(body.parts))],
            geom_rgbas=[body.rgba] * len(body.parts),
            geom_frictions=[GEOM_FRICTION] * len(body.parts),
            geom_condims=[GEOM_CONDIM] * len(body.parts),
            density=GEOM_DENSITY,
            solref=GEOM_SOLREF,
            solimp=GEOM_SOLIMP,
            locations_relative_to_center=True,
            joints=joints,
            obj_types="all",
            duplicate_collision_geoms=True,
        )
        self.category_name = "_".join(re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = _PROPERTIES


class _Target(CompositeObject):
    descriptor: BodyDescriptor

    def __init__(self, name="target", joints=None):
        body = self.descriptor
        points = target_outline_points(body)
        half_x = max(abs(part.x) + part.half_x for part in body.parts)
        half_y = max(abs(part.y) + part.half_y for part in body.parts)
        super().__init__(
            name=name,
            total_size=(half_x, half_y, TARGET_HALF_HEIGHT),
            geom_types=["cylinder"] * len(points),
            geom_sizes=[(TARGET_DOT_RADIUS, TARGET_HALF_HEIGHT)] * len(points),
            geom_locations=[(x, y, _TABLE_HALF_HEIGHT + TARGET_HALF_HEIGHT) for x, y in points],
            geom_names=[f"dot_{index}" for index in range(len(points))],
            geom_rgbas=[body.rgba] * len(points),
            locations_relative_to_center=True,
            joints=joints,
            obj_types="visual",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = _PROPERTIES


def _register(name: str, base: type[CompositeObject], descriptor: BodyDescriptor) -> type[CompositeObject]:
    return register_object(type(name, (base,), {"descriptor": descriptor}))


PiBlock = _register("PiBlock", _Block, PI_BODY)
PBlock = _register("PBlock", _Block, P_BODY)
IBlock = _register("IBlock", _Block, I_BODY)
PiTarget = _register("PiTarget", _Target, PI_BODY)
PTarget = _register("PTarget", _Target, P_BODY)
ITarget = _register("ITarget", _Target, I_BODY)


class _PushPiTask(TASK_MAPPING["libero_tabletop_manipulation"]):
    body_pairs: tuple[tuple[BodyDescriptor, str, str], ...] = ()

    def _state(self, body_name: str, descriptor_name: str) -> BodyState:
        body_id = self.obj_body_id[body_name]
        x, y, z = (float(value) for value in self.sim.data.body_xpos[body_id])
        qw, qx, qy, qz = (float(value) for value in self.sim.data.body_xquat[body_id])
        return BodyState(descriptor_name, x, y, z, qw, qx, qy, qz)

    def coverage(self) -> dict[str, float]:
        values = {}
        covered_area = total_area = 0.0
        for body, movable_name, target_name in self.body_pairs:
            movable = self._state(movable_name, body.name)
            target = self._state(target_name, body.name)
            target_yaw = quaternion_euler(target)[2]
            placed = dataclasses.replace(body, target_x=target.x, target_y=target.y, target_yaw=target_yaw)
            coverage = footprint_overlap_coverage(placed, movable)
            area = sum(4 * part.half_x * part.half_y for part in body.parts)
            values[body.name] = coverage
            covered_area += coverage * area
            total_area += area
        values["overall"] = covered_area / total_area
        return values

    def _check_success(self):
        return bool(self.obj_body_id) and all(value >= 0.95 for value in self.coverage().values())


@register_problem
class PushPiLibero(_PushPiTask):
    body_pairs = SCENARIOS["push_pi"][1]


@register_problem
class PushLettersLibero(_PushPiTask):
    body_pairs = SCENARIOS["push_p_i"][1]


_BDDL = {
    "push_pi": """(define (problem PushPiLibero)
  (:domain robosuite)
  (:language Push the PI-shaped block onto its dotted target outline)
  (:regions
    (pi_spawn_region (:target main_table) (:ranges ((-0.22 -0.22 -0.02 -0.04))) (:yaw_rotation ((0 0))))
    (pi_target_region (:target main_table) (:ranges ((0.139 0.119 0.141 0.121))) (:yaw_rotation ((0 0))))
  )
  (:fixtures main_table - table pi_target_1 - pi_target)
  (:objects pi_1 - pi_block)
  (:obj_of_interest pi_1 pi_target_1)
  (:init (On pi_1 main_table_pi_spawn_region) (On pi_target_1 main_table_pi_target_region))
  (:goal (And (On pi_1 pi_target_1)))
)""",
    "push_p_i": """(define (problem PushLettersLibero)
  (:domain robosuite)
  (:language Push the P and I blocks onto their matching dotted target outlines)
  (:regions
    (p_spawn_region (:target main_table) (:ranges ((-0.26 -0.22 -0.08 -0.04))) (:yaw_rotation ((0 0))))
    (i_spawn_region (:target main_table) (:ranges ((0.07 -0.22 0.23 -0.04))) (:yaw_rotation ((0 0))))
    (p_target_region (:target main_table) (:ranges ((-0.131 0.139 -0.129 0.141))) (:yaw_rotation ((0 0))))
    (i_target_region (:target main_table) (:ranges ((0.139 0.139 0.141 0.141))) (:yaw_rotation ((0 0))))
  )
  (:fixtures main_table - table p_target_1 - p_target i_target_1 - i_target)
  (:objects p_1 - p_block i_1 - i_block)
  (:obj_of_interest p_1 i_1 p_target_1 i_target_1)
  (:init
    (On p_1 main_table_p_spawn_region)
    (On i_1 main_table_i_spawn_region)
    (On p_target_1 main_table_p_target_region)
    (On i_target_1 main_table_i_target_region)
  )
  (:goal (And (On p_1 p_target_1) (On i_1 i_target_1)))
)""",
}


def create_env(scenario: str, *, resolution: int, seed: int, horizon: int):
    try:
        _, _, prompt = SCENARIOS[scenario]
    except KeyError as error:
        raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}") from error
    temporary = tempfile.TemporaryDirectory(prefix="libero-push-pi-")
    bddl_path = pathlib.Path(temporary.name) / f"{scenario}.bddl"
    bddl_path.write_text(_BDDL[scenario], encoding="utf-8")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
        horizon=horizon,
        ignore_done=True,
    )
    env.seed(seed)
    env.push_pi_temporary_directory = temporary
    return env, prompt


def self_check() -> None:
    assert set(SCENARIOS) == {"push_pi", "push_p_i"}
    assert CONTROL_HZ * 6 == 120
    assert CONTROL_HZ * 30 == 600
    assert all(target_outline_points(body) for body in (PI_BODY, P_BODY, I_BODY))


if __name__ == "__main__":
    self_check()
