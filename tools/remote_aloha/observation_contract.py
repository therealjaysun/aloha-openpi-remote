from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from openpi_client import image_tools

_CAMERAS = {"cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"}


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
    if not isinstance(images, Mapping) or "cam_high" not in images or not set(images) <= _CAMERAS:
        raise ValueError("observation.images must contain cam_high and only supported ALOHA cameras")
    for name, image in images.items():
        if not isinstance(image, np.ndarray) or image.shape != (3, 224, 224) or image.dtype != np.uint8:
            raise ValueError(f"observation.images.{name} must be uint8 CHW with shape (3, 224, 224)")
    if "prompt" in observation and (not isinstance(observation["prompt"], str) or not observation["prompt"].strip()):
        raise ValueError("observation.prompt must be a nonempty string")
    return observation


def convert_gym_observation(observation: object, prompt: str | None = None) -> dict:
    if not isinstance(observation, Mapping) or set(observation) != {"pixels", "agent_pos"}:
        raise ValueError("Gym observation must contain exactly pixels and agent_pos")
    pixels = observation["pixels"]
    if not isinstance(pixels, Mapping) or set(pixels) != {"top"}:
        raise ValueError("Gym observation.pixels must contain exactly top")
    image = pixels["top"]
    state = observation["agent_pos"]
    if not isinstance(image, np.ndarray) or image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError("Gym observation.pixels.top must be uint8 HWC with shape (480, 640, 3)")
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (14,)
        or state.dtype != np.float64
        or not np.isfinite(state).all()
    ):
        raise ValueError("Gym observation.agent_pos must be finite float64 with shape (14,)")
    resized = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    result = {"state": state, "images": {"cam_high": np.transpose(resized, (2, 0, 1))}}
    if prompt is not None:
        result["prompt"] = prompt
    return validate_policy_observation(result)
