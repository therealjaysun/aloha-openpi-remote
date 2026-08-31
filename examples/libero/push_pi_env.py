from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import random
import re
import tempfile

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.base_object import register_object
from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from libero.libero.envs.bddl_base_domain import register_problem
from libero.libero.envs.regions import REGION_SAMPLERS
from libero.libero.envs.regions import MultiRegionRandomSampler
from libero.libero.envs.regions import TableRegionSampler
from robosuite.models.objects import CompositeObject
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import CustomMaterial

from tools.remote_aloha.scenarios import GEOM_CONDIM
from tools.remote_aloha.scenarios import GEOM_DENSITY
from tools.remote_aloha.scenarios import GEOM_FRICTION
from tools.remote_aloha.scenarios import GEOM_SOLIMP
from tools.remote_aloha.scenarios import GEOM_SOLREF
from tools.remote_aloha.scenarios import I_BODY
from tools.remote_aloha.scenarios import OBJECT_HALF_HEIGHT
from tools.remote_aloha.scenarios import P_BODY
from tools.remote_aloha.scenarios import PI_BODY
from tools.remote_aloha.scenarios import TARGET_HALF_HEIGHT
from tools.remote_aloha.scenarios import BodyDescriptor
from tools.remote_aloha.scenarios import BodyState
from tools.remote_aloha.scenarios import footprint_overlap_coverage
from tools.remote_aloha.scenarios import get_scenario
from tools.remote_aloha.scenarios import quaternion_euler
from tools.remote_aloha.scenarios import target_outline_points

CONTROL_HZ = 20
LIBERO_TARGET_DOT_RADIUS = 0.003
LIBERO_TARGET_DOT_SPACING = 0.012
LIBERO_PI_TABLE_HALF_SIZE = (0.5, 0.6)
LIBERO_PI_PLACEMENT_MARGIN = PI_BODY.footprint_radius + LIBERO_TARGET_DOT_RADIUS
LIBERO_PI_X_RANGE = tuple(value * (LIBERO_PI_TABLE_HALF_SIZE[0] - LIBERO_PI_PLACEMENT_MARGIN) for value in (-1, 1))
LIBERO_PI_Y_RANGE = tuple(value * (LIBERO_PI_TABLE_HALF_SIZE[1] - LIBERO_PI_PLACEMENT_MARGIN) for value in (-1, 1))
LIBERO_PI_MIN_SEPARATION = 2 * LIBERO_PI_PLACEMENT_MARGIN + 0.02
LIBERO_PI_REGION_HALF_RANGE = 0.001
LIBERO_AGENTVIEW_RESOLUTION = 256
LIBERO_AGENTVIEW_PIXEL_MARGIN = 4
LIBERO_PORTRAIT_HALF_SIZE = (0.10, 0.075)
LIBERO_PORTRAIT_HALF_THICKNESS = 0.001
LIBERO_COKE_CAN_RADIUS = 0.022
LIBERO_COKE_CAN_HALF_HEIGHT = 0.061
LIBERO_COKE_CAN_PROFILE_SIDES = 8
LIBERO_COKE_CAN_CENTER = (-0.11, 0.0)
LIBERO_COKE_LIGHT_AMBIENT = 0.25
LIBERO_COKE_LIGHT_DIFFUSE = 1.0
LIBERO_PORTRAIT_CENTERS = {
    "taylor_swift": (-0.20, 0.12),
    "ian_mckellen": (-0.20, -0.12),
    "ed_sheeran": (0.02, 0.18),
    "emma_stone": (0.02, 0.0),
    "snoop_dogg": (0.02, -0.18),
}
_PORTRAIT_ASSET_DIR = pathlib.Path(__file__).parent / "assets" / "rt2_portraits"
# Pinned LIBERO tabletop agentview world-to-pixel transform at 256x256.
_AGENTVIEW_TRANSFORM = (
    (-99.5837977, 309.019317, -80.4181527, 195.088575),
    (94.5627061, -0.0000419308495, -320.834606, 454.375772),
    (-0.777998244, -0.000000152115266, -0.62826645, 1.52412879),
)
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
    "coke_taylor": (
        "CokeOnTaylorLibero",
        (),
        "Put the red can on Taylor Swift.",
    ),
}


class _SeededTableRegionSampler(MultiRegionRandomSampler):
    def __init__(self, *args, **kwargs):
        kwargs.update(ensure_object_boundary_in_range=False, z_offset=0.01)
        super().__init__(*args, **kwargs)


REGION_SAMPLERS.update(
    {
        SCENARIOS["push_pi"][0].lower(): {"table": _SeededTableRegionSampler},
        SCENARIOS["push_p_i"][0].lower(): {"table": TableRegionSampler},
        SCENARIOS["coke_taylor"][0].lower(): {"table": _SeededTableRegionSampler},
    }
)

_PROPERTIES = {
    "articulation": {"default_open_ranges": [], "default_close_ranges": []},
    "vis_site_names": {},
}


class _Block(CompositeObject):
    descriptor: BodyDescriptor
    layout_yaw = 0.0

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
        self.rotation = (self.layout_yaw, self.layout_yaw)
        self.rotation_axis = "z"
        self.object_properties = _PROPERTIES


class _Target(CompositeObject):
    descriptor: BodyDescriptor

    def __init__(self, name="target", joints=None):
        body = self.descriptor
        points = target_outline_points(body, spacing=LIBERO_TARGET_DOT_SPACING)
        half_x = max(abs(part.x) + part.half_x for part in body.parts)
        half_y = max(abs(part.y) + part.half_y for part in body.parts)
        super().__init__(
            name=name,
            total_size=(half_x, half_y, TARGET_HALF_HEIGHT),
            geom_types=["cylinder"] * len(points),
            geom_sizes=[(LIBERO_TARGET_DOT_RADIUS, TARGET_HALF_HEIGHT)] * len(points),
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


class _Portrait(CompositeObject):
    image_file = ""

    def __init__(self, name="portrait", joints=None):
        super().__init__(
            name=name,
            total_size=(*LIBERO_PORTRAIT_HALF_SIZE, LIBERO_PORTRAIT_HALF_THICKNESS),
            geom_types=["box"],
            geom_sizes=[(*LIBERO_PORTRAIT_HALF_SIZE, LIBERO_PORTRAIT_HALF_THICKNESS)],
            geom_locations=[(0.0, 0.0, _TABLE_HALF_HEIGHT + LIBERO_PORTRAIT_HALF_THICKNESS)],
            geom_names=["photo"],
            geom_rgbas=[(1.0, 1.0, 1.0, 1.0)],
            geom_materials=["portrait"],
            geom_frictions=[GEOM_FRICTION],
            geom_condims=[GEOM_CONDIM],
            density=GEOM_DENSITY,
            solref=GEOM_SOLREF,
            solimp=GEOM_SOLIMP,
            locations_relative_to_center=True,
            joints=joints,
            obj_types="all",
            duplicate_collision_geoms=False,
        )
        self.append_material(
            CustomMaterial(
                texture=None,
                tex_name="portrait_texture",
                mat_name="portrait",
                tex_attrib={"type": "2d", "file": str(_PORTRAIT_ASSET_DIR / self.image_file)},
                mat_attrib={"reflectance": "0", "shininess": "0", "specular": "0", "texrepeat": "1 1"},
            )
        )
        self.category_name = "_".join(re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = _PROPERTIES


@register_object
class CokeCan(MujocoXMLObject):
    def __init__(self, name="coke_can", joints="default"):
        super().__init__(
            str(_PORTRAIT_ASSET_DIR / "octagonal_coke_can.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "coke_can"
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


def _register_portrait(name: str, image_file: str) -> type[CompositeObject]:
    return register_object(type(name, (_Portrait,), {"image_file": image_file}))


TaylorSwiftPhoto = _register_portrait("TaylorSwiftPhoto", "taylor_swift.png")
IanMckellenPhoto = _register_portrait("IanMckellenPhoto", "ian_mckellen.png")
EdSheeranPhoto = _register_portrait("EdSheeranPhoto", "ed_sheeran.png")
EmmaStonePhoto = _register_portrait("EmmaStonePhoto", "emma_stone.png")
SnoopDoggPhoto = _register_portrait("SnoopDoggPhoto", "snoop_dogg.png")


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

    def layout(self) -> dict[str, dict[str, object]]:
        values = {}
        for body, movable_name, target_name in self.body_pairs:
            for name in (movable_name, target_name):
                state = self._state(name, body.name)
                values[name] = {**dataclasses.asdict(state), "yaw_radians": quaternion_euler(state)[2]}
        return values

    def _check_success(self):
        return bool(self.obj_body_id) and all(value >= 0.95 for value in self.coverage().values())


@register_problem
class PushPiLibero(_PushPiTask):
    body_pairs = SCENARIOS["push_pi"][1]


@register_problem
class PushLettersLibero(_PushPiTask):
    body_pairs = SCENARIOS["push_p_i"][1]


@register_problem
class CokeOnTaylorLibero(TASK_MAPPING["libero_tabletop_manipulation"]):
    tracked = (("coke_can_1", "coke_can"), *((f"{name}_photo_1", name) for name in LIBERO_PORTRAIT_CENTERS))

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.model.light_ambient[:] = LIBERO_COKE_LIGHT_AMBIENT
        self.sim.model.light_diffuse[:] = LIBERO_COKE_LIGHT_DIFFUSE

    def _state(self, body_name: str, descriptor_name: str) -> BodyState:
        body_id = self.obj_body_id[body_name]
        x, y, z = (float(value) for value in self.sim.data.body_xpos[body_id])
        qw, qx, qy, qz = (float(value) for value in self.sim.data.body_xquat[body_id])
        return BodyState(descriptor_name, x, y, z, qw, qx, qy, qz)

    def layout(self) -> dict[str, dict[str, object]]:
        values = {}
        for body_name, descriptor_name in self.tracked:
            state = self._state(body_name, descriptor_name)
            values[body_name] = {**dataclasses.asdict(state), "yaw_radians": quaternion_euler(state)[2]}
        return values

    def coverage(self) -> dict[str, float]:
        can = self._state("coke_can_1", "coke_can")
        target = self._state("taylor_swift_photo_1", "taylor_swift")
        target_top = target.z + _TABLE_HALF_HEIGHT + 2 * LIBERO_PORTRAIT_HALF_THICKNESS
        centered = (
            abs(can.x - target.x) <= LIBERO_PORTRAIT_HALF_SIZE[0] - LIBERO_COKE_CAN_RADIUS
            and abs(can.y - target.y) <= LIBERO_PORTRAIT_HALF_SIZE[1] - LIBERO_COKE_CAN_RADIUS
        )
        placed = centered and abs(can.z - LIBERO_COKE_CAN_HALF_HEIGHT - target_top) <= 0.02
        value = float(placed)
        return {"taylor_swift": value, "overall": value}

    def _check_success(self):
        return self.coverage()["overall"] == 1.0


_BDDL = {
    "push_pi": """(define (problem PushPiLibero)
  (:domain robosuite)
  (:language Push the PI-shaped block onto its dotted target outline)
  (:regions
    (pi_spawn_region (:target main_table) (:ranges (({spawn_x_min} {spawn_y_min} {spawn_x_max} {spawn_y_max}))) (:yaw_rotation (({block_yaw} {block_yaw}))))
    (pi_target_region (:target main_table) (:ranges (({target_x_min} {target_y_min} {target_x_max} {target_y_max}))) (:yaw_rotation (({target_yaw} {target_yaw}))))
  )
  (:fixtures main_table - table pi_target_1 - pi_target)
  (:objects pi_1 - pi_block)
  (:obj_of_interest pi_1 pi_target_1)
  (:init (On pi_1 main_table_pi_spawn_region) (On pi_target_1 main_table_pi_target_region))
  (:goal (And (On pi_1 pi_target_1)))
)""",
    "push_p_i": """(define (problem PushLettersLibero)
  (:domain robosuite)
  (:language Push the red P-shaped block onto the red dotted P outline and the blue I-shaped block onto the blue dotted I outline)
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
    "coke_taylor": """(define (problem CokeOnTaylorLibero)
  (:domain robosuite)
  (:language Put the red can on Taylor Swift.)
  (:regions
{portrait_regions}
    (coke_can_region (:target main_table) (:ranges ((-0.111 -0.001 -0.109 0.001))) (:yaw_rotation ((0 0))))
  )
  (:fixtures
    main_table - table
    taylor_swift_photo_1 - taylor_swift_photo
    ian_mckellen_photo_1 - ian_mckellen_photo
    ed_sheeran_photo_1 - ed_sheeran_photo
    emma_stone_photo_1 - emma_stone_photo
    snoop_dogg_photo_1 - snoop_dogg_photo
  )
  (:objects coke_can_1 - coke_can)
  (:obj_of_interest coke_can_1 taylor_swift_photo_1 ian_mckellen_photo_1 ed_sheeran_photo_1 emma_stone_photo_1 snoop_dogg_photo_1)
  (:init
    (On taylor_swift_photo_1 main_table_taylor_swift_region)
    (On ian_mckellen_photo_1 main_table_ian_mckellen_region)
    (On ed_sheeran_photo_1 main_table_ed_sheeran_region)
    (On emma_stone_photo_1 main_table_emma_stone_region)
    (On snoop_dogg_photo_1 main_table_snoop_dogg_region)
    (On coke_can_1 main_table_coke_can_region)
  )
  (:goal (And (On coke_can_1 taylor_swift_photo_1)))
)""",
}


def push_pi_layout(seed: int) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("LIBERO layout seed must be an unsigned 32-bit integer")
    rng = random.Random(seed)
    block_yaw = rng.uniform(-math.pi, math.pi)
    target_yaw = rng.uniform(-math.pi, math.pi)
    for _ in range(1000):
        block = (rng.uniform(*LIBERO_PI_X_RANGE), rng.uniform(*LIBERO_PI_Y_RANGE))
        target = (rng.uniform(*LIBERO_PI_X_RANGE), rng.uniform(*LIBERO_PI_Y_RANGE))
        distance = math.dist(block, target)
        if distance >= LIBERO_PI_MIN_SEPARATION and _agentview_contains(block) and _agentview_contains(target):
            break
    else:
        raise RuntimeError("could not sample a separated LIBERO Push-PI layout")
    layout = {
        "block_xy": [round(value, 6) for value in block],
        "target_xy": [round(value, 6) for value in target],
        "block_yaw_radians": round(block_yaw, 6),
        "target_yaw_radians": round(target_yaw, 6),
        "center_distance": round(distance, 6),
        "agentview_visible": True,
    }
    layout["layout_hash"] = hashlib.sha256(
        json.dumps(layout, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return layout


def coke_taylor_layout(seed: int) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("LIBERO layout seed must be an unsigned 32-bit integer")
    names = list(LIBERO_PORTRAIT_CENTERS)
    random.Random(seed).shuffle(names)
    portrait_xy = dict(zip(names, LIBERO_PORTRAIT_CENTERS.values(), strict=True))
    layout = {
        "arrangement": {
            "top_row": names[:2],
            "bottom_row": names[2:],
        },
        "can_dimensions_m": {"diameter": 2 * LIBERO_COKE_CAN_RADIUS, "height": 2 * LIBERO_COKE_CAN_HALF_HEIGHT},
        "can_profile_sides": LIBERO_COKE_CAN_PROFILE_SIDES,
        "can_xy": list(LIBERO_COKE_CAN_CENTER),
        "lighting": {"ambient": LIBERO_COKE_LIGHT_AMBIENT, "diffuse": LIBERO_COKE_LIGHT_DIFFUSE},
        "portrait_asset_sha256": {
            name: hashlib.sha256((_PORTRAIT_ASSET_DIR / f"{name}.png").read_bytes()).hexdigest()
            for name in LIBERO_PORTRAIT_CENTERS
        },
        "portrait_xy": {name: list(center) for name, center in portrait_xy.items()},
    }
    layout["layout_hash"] = hashlib.sha256(
        json.dumps(layout, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return layout


def scenario_metadata(scenario: str, seed: int) -> dict[str, object] | None:
    if scenario == "push_pi":
        return push_pi_layout(seed)
    if scenario == "coke_taylor":
        return coke_taylor_layout(seed)
    if scenario == "push_p_i":
        return None
    raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}")


def _agentview_contains(
    center: tuple[float, float], radius: float = LIBERO_PI_PLACEMENT_MARGIN, z: float = 0.912
) -> bool:
    for index in range(32):
        angle = 2 * math.pi * index / 32
        world = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            z,
            1.0,
        )
        projected = [
            math.fsum(value * coordinate for value, coordinate in zip(row, world, strict=True))
            for row in _AGENTVIEW_TRANSFORM
        ]
        if projected[2] <= 0:
            return False
        pixel = (projected[0] / projected[2], projected[1] / projected[2])
        if any(
            value < LIBERO_AGENTVIEW_PIXEL_MARGIN
            or value >= LIBERO_AGENTVIEW_RESOLUTION - LIBERO_AGENTVIEW_PIXEL_MARGIN
            for value in pixel
        ):
            return False
    return True


def scenario_bddl(scenario: str, seed: int) -> str:
    try:
        template = _BDDL[scenario]
    except KeyError as error:
        raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}") from error
    if scenario == "coke_taylor":
        portrait_regions = []
        for name, (x, y) in coke_taylor_layout(seed)["portrait_xy"].items():
            portrait_regions.append(
                f"    ({name}_region (:target main_table) (:ranges (({x - 0.001:.3f} {y - 0.001:.3f} "
                f"{x + 0.001:.3f} {y + 0.001:.3f}))) (:yaw_rotation ((0 0))))"
            )
        return template.format(portrait_regions="\n".join(portrait_regions))
    if scenario != "push_pi":
        return template
    layout = push_pi_layout(seed)
    block_x, block_y = layout["block_xy"]
    target_x, target_y = layout["target_xy"]
    half_range = LIBERO_PI_REGION_HALF_RANGE
    return template.format(
        spawn_x_min=f"{block_x - half_range:.6f}",
        spawn_y_min=f"{block_y - half_range:.6f}",
        spawn_x_max=f"{block_x + half_range:.6f}",
        spawn_y_max=f"{block_y + half_range:.6f}",
        target_x_min=f"{target_x - half_range:.6f}",
        target_y_min=f"{target_y - half_range:.6f}",
        target_x_max=f"{target_x + half_range:.6f}",
        target_y_max=f"{target_y + half_range:.6f}",
        block_yaw=f"{layout['block_yaw_radians']:.6f}",
        target_yaw=f"{layout['target_yaw_radians']:.6f}",
    )


def create_env(scenario: str, *, resolution: int, seed: int, horizon: int):
    try:
        _, _, prompt = SCENARIOS[scenario]
    except KeyError as error:
        raise ValueError(f"scenario must be one of: {', '.join(SCENARIOS)}") from error
    temporary = tempfile.TemporaryDirectory(prefix="libero-push-pi-")
    bddl_path = pathlib.Path(temporary.name) / f"{scenario}.bddl"
    bddl_path.write_text(scenario_bddl(scenario, seed), encoding="utf-8")
    layout = scenario_metadata(scenario, seed)
    if scenario == "push_pi":
        # ponytail: class-scoped pose config assumes sequential environment construction; pass task context if parallel creation is added.
        PiBlock.layout_yaw = layout["block_yaw_radians"]
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
        horizon=horizon,
        ignore_done=True,
    )
    env.seed(seed)
    env.push_pi_temporary_directory = temporary
    env.scenario_layout = layout
    return env, prompt


def snapshot_layout(environment) -> dict[str, object]:
    actual = environment.env.layout()
    planned = environment.scenario_layout
    if planned is None:
        return actual
    if "portrait_xy" in planned:
        validation = {
            "agentview_visible": all(
                _agentview_contains(tuple(center), math.hypot(*LIBERO_PORTRAIT_HALF_SIZE))
                for center in LIBERO_PORTRAIT_CENTERS.values()
            )
            and _agentview_contains(LIBERO_COKE_CAN_CENTER, LIBERO_COKE_CAN_RADIUS, 0.98),
            "can_xy_error": math.dist(planned["can_xy"], (actual["coke_can_1"]["x"], actual["coke_can_1"]["y"])),
            "portrait_xy_error": {
                name: math.dist(center, (actual[f"{name}_photo_1"]["x"], actual[f"{name}_photo_1"]["y"]))
                for name, center in planned["portrait_xy"].items()
            },
            "portrait_yaw_error": {
                name: abs(_wrapped_angle(actual[f"{name}_photo_1"]["yaw_radians"])) for name in planned["portrait_xy"]
            },
            "within_table": all(
                abs(center[axis]) + LIBERO_PORTRAIT_HALF_SIZE[axis] <= LIBERO_PI_TABLE_HALF_SIZE[axis]
                for center in LIBERO_PORTRAIT_CENTERS.values()
                for axis in (0, 1)
            )
            and all(
                abs(LIBERO_COKE_CAN_CENTER[axis]) + LIBERO_COKE_CAN_RADIUS <= LIBERO_PI_TABLE_HALF_SIZE[axis]
                for axis in (0, 1)
            ),
        }
        if not (
            validation["agentview_visible"]
            and validation["within_table"]
            and validation["can_xy_error"] <= 0.002
            and max(validation["portrait_xy_error"].values()) <= 0.002
            and max(validation["portrait_yaw_error"].values()) <= 1e-6
        ):
            raise ValueError("actual LIBERO Coke-on-Taylor layout failed seeded pose validation")
        return {**actual, "validation": validation}
    block = actual["pi_1"]
    target = actual["pi_target_1"]
    block_xy = (block["x"], block["y"])
    target_xy = (target["x"], target["y"])
    validation = {
        "agentview_visible": _agentview_contains(block_xy) and _agentview_contains(target_xy),
        "block_xy_error": math.dist(planned["block_xy"], block_xy),
        "block_yaw_error": abs(_wrapped_angle(block["yaw_radians"] - planned["block_yaw_radians"])),
        "non_overlapping": math.dist(block_xy, target_xy) >= LIBERO_PI_MIN_SEPARATION,
        "target_xy_error": math.dist(planned["target_xy"], target_xy),
        "target_yaw_error": abs(_wrapped_angle(target["yaw_radians"] - planned["target_yaw_radians"])),
        "within_table": all(
            abs(point[axis]) + LIBERO_PI_PLACEMENT_MARGIN <= LIBERO_PI_TABLE_HALF_SIZE[axis]
            for point in (block_xy, target_xy)
            for axis in (0, 1)
        ),
    }
    if not (
        validation["agentview_visible"]
        and validation["non_overlapping"]
        and validation["within_table"]
        and validation["block_xy_error"] <= 0.002
        and validation["target_xy_error"] <= 0.002
        and validation["block_yaw_error"] <= 1e-6
        and validation["target_yaw_error"] <= 1e-6
    ):
        raise ValueError("actual LIBERO Push-PI layout failed seeded pose validation")
    return {**actual, "validation": validation}


def _wrapped_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def scenario_hash(scenario: str, seed: int) -> str:
    extra = (
        json.dumps(coke_taylor_layout(seed), separators=(",", ":"), sort_keys=True) if scenario == "coke_taylor" else ""
    )
    scene = f"{scenario_bddl(scenario, seed)}\0{LIBERO_TARGET_DOT_RADIUS}\0{LIBERO_TARGET_DOT_SPACING}\0{extra}"
    return hashlib.sha256(scene.encode()).hexdigest()


def self_check() -> None:
    assert set(SCENARIOS) == {"push_pi", "push_p_i", "coke_taylor"}
    assert SCENARIOS["coke_taylor"][2] == "Put the red can on Taylor Swift."
    assert CONTROL_HZ * 6 == 120
    assert CONTROL_HZ * 30 == 600
    assert CONTROL_HZ * 300 == 6000
    assert [
        len(target_outline_points(body, spacing=LIBERO_TARGET_DOT_SPACING)) for body in (PI_BODY, P_BODY, I_BODY)
    ] == [60, 57, 48]
    layouts = [push_pi_layout(seed) for seed in range(12)]
    assert len({tuple(layout["block_xy"]) for layout in layouts}) == 12
    assert len({tuple(layout["target_xy"]) for layout in layouts}) == 12
    assert len({layout["block_yaw_radians"] for layout in layouts}) == 12
    assert len({layout["target_yaw_radians"] for layout in layouts}) == 12
    assert all(layout["center_distance"] >= LIBERO_PI_MIN_SEPARATION for layout in layouts)
    assert all(layout["agentview_visible"] for layout in layouts)
    assert all(
        abs(layout[name][axis]) + LIBERO_PI_PLACEMENT_MARGIN <= LIBERO_PI_TABLE_HALF_SIZE[axis]
        for layout in layouts
        for name in ("block_xy", "target_xy")
        for axis in (0, 1)
    )
    assert len({layout["layout_hash"] for layout in layouts}) == 12
    coke_layouts = [coke_taylor_layout(seed) for seed in range(5)]
    assert len({layout["layout_hash"] for layout in coke_layouts}) == 5
    assert (
        len({tuple(layout["arrangement"]["top_row"] + layout["arrangement"]["bottom_row"]) for layout in coke_layouts})
        == 5
    )
    coke_layout = coke_layouts[0]
    assert coke_layout["lighting"] == {"ambient": 0.25, "diffuse": 1.0}
    assert coke_layout["can_dimensions_m"] == {"diameter": 0.044, "height": 0.122}
    assert coke_layout["can_profile_sides"] == 8
    assert len(set(coke_layout["portrait_asset_sha256"].values())) == 5
    assert all(len(value) == 64 for value in coke_layout["portrait_asset_sha256"].values())
    assert all(
        _agentview_contains(center, math.hypot(*LIBERO_PORTRAIT_HALF_SIZE))
        for center in LIBERO_PORTRAIT_CENTERS.values()
    )
    assert _agentview_contains(LIBERO_COKE_CAN_CENTER, LIBERO_COKE_CAN_RADIUS, 0.98)
    centers = tuple(LIBERO_PORTRAIT_CENTERS.values())
    assert all(
        abs(first[0] - second[0]) >= 2 * LIBERO_PORTRAIT_HALF_SIZE[0]
        or abs(first[1] - second[1]) >= 2 * LIBERO_PORTRAIT_HALF_SIZE[1]
        for index, first in enumerate(centers)
        for second in centers[index + 1 :]
    )
    assert all(
        abs(LIBERO_COKE_CAN_CENTER[0] - center[0]) >= LIBERO_PORTRAIT_HALF_SIZE[0] + LIBERO_COKE_CAN_RADIUS
        or abs(LIBERO_COKE_CAN_CENTER[1] - center[1]) >= LIBERO_PORTRAIT_HALF_SIZE[1] + LIBERO_COKE_CAN_RADIUS
        for center in centers
    )
    assert all(len(scenario_hash(name, 0)) == 64 for name in SCENARIOS)


if __name__ == "__main__":
    self_check()
