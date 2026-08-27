from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from tools.remote_aloha.config import PolicyProfile


def validate_server_metadata(metadata: object, profile: PolicyProfile, source_sha: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("server metadata must be a mapping")
    expected = {
        "policy_profile": profile.name,
        "config_name": profile.config_name,
        "checkpoint_label": profile.checkpoint_label,
        "action_horizon": profile.action_horizon,
        "action_dimension": profile.action_dimension,
        "source_sha": source_sha,
        "compact_masked_images": True,
        "jax_platform": "gpu",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"server metadata {key!r} does not match the requested profile")
    device = metadata.get("jax_device")
    if not isinstance(device, str) or "3090" not in device:
        raise ValueError("server metadata must identify the RTX 3090 JAX device")


def validate_policy_response(response: object, profile: PolicyProfile) -> np.ndarray:
    if not isinstance(response, Mapping) or "actions" not in response:
        raise ValueError("policy response must contain actions")
    actions = np.asarray(response["actions"])
    expected = (profile.action_horizon, profile.action_dimension)
    if actions.shape != expected or not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
        raise ValueError(f"actions must be finite floating values with shape {expected}")
    return actions


def validate_server_timing(response: object) -> dict[str, Real]:
    timing = response.get("server_timing") if isinstance(response, Mapping) else None
    if not isinstance(timing, Mapping):
        raise ValueError("policy response must contain server timing")
    for key in ("infer_ms", "prev_total_ms"):
        if key == "prev_total_ms" and key not in timing:
            continue
        value = timing.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"server timing {key!r} must be finite and nonnegative")
    return dict(timing)
