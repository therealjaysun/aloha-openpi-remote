from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from openpi_client import image_tools

GYM_CAMERA_VIEWS = ("top", "left_wrist", "right_wrist")
POLICY_CAMERA_VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
_GYM_TO_POLICY_CAMERA = dict(zip(GYM_CAMERA_VIEWS, POLICY_CAMERA_VIEWS, strict=True))


def _validate_gym_observation(observation: object, camera_views: tuple[str, ...]) -> tuple[Mapping, np.ndarray]:
    if not isinstance(observation, Mapping) or set(observation) != {"pixels", "agent_pos"}:
        raise ValueError("Gym observation must contain exactly pixels and agent_pos")
    pixels = observation["pixels"]
    if not isinstance(pixels, Mapping) or set(pixels) != set(camera_views):
        names = ", ".join(camera_views)
        raise ValueError(f"Gym observation.pixels must contain exactly {names}")
    for name, image in pixels.items():
        if not isinstance(image, np.ndarray) or image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"Gym observation.pixels.{name} must be uint8 HWC with shape (480, 640, 3)")
    state = observation["agent_pos"]
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (14,)
        or state.dtype != np.float64
        or not np.isfinite(state).all()
    ):
        raise ValueError("Gym observation.agent_pos must be finite float64 with shape (14,)")
    return pixels, state


def _convert_image(image: np.ndarray) -> np.ndarray:
    resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    return np.transpose(resized, (2, 0, 1))


def validate_policy_observation(observation: object) -> dict:
    if not isinstance(observation, dict) or set(observation) not in (
        {"state", "images"},
        {"state", "images", "prompt"},
    ):
        raise ValueError("observation must contain only state, images, and optional prompt")
    state = observation["state"]
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (14,)
        or state.dtype != np.float64
        or not np.isfinite(state).all()
    ):
        raise ValueError("observation.state must be finite float64 with shape (14,)")
    images = observation["images"]
    if not isinstance(images, Mapping) or set(images) != set(POLICY_CAMERA_VIEWS):
        raise ValueError("observation.images must contain exactly the high, left-wrist, and right-wrist cameras")
    for name, image in images.items():
        if not isinstance(image, np.ndarray) or image.shape != (3, 224, 224) or image.dtype != np.uint8:
            raise ValueError(f"observation.images.{name} must be uint8 CHW with shape (3, 224, 224)")
    if "prompt" in observation and (not isinstance(observation["prompt"], str) or not observation["prompt"].strip()):
        raise ValueError("observation.prompt must be a nonempty string")
    return observation


def convert_gym_observation(observation: object, prompt: str | None = None) -> dict:
    pixels, state = _validate_gym_observation(observation, GYM_CAMERA_VIEWS)
    images = {_GYM_TO_POLICY_CAMERA[name]: _convert_image(pixels[name]) for name in GYM_CAMERA_VIEWS}
    result = {"state": state, "images": images}
    if prompt is not None:
        result["prompt"] = prompt
    return validate_policy_observation(result)


def convert_gym_artifact_observation(observation: object) -> dict:
    """Convert a valid partial-artifact frame after a wrist-capture failure."""
    pixels, state = _validate_gym_observation(observation, ("top",))
    unavailable = np.zeros((3, 224, 224), dtype=np.uint8)
    return {
        "state": state,
        "images": {
            "cam_high": _convert_image(pixels["top"]),
            "cam_left_wrist": unavailable,
            "cam_right_wrist": unavailable.copy(),
        },
    }
