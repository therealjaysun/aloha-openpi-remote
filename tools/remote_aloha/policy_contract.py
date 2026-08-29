from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from tools.remote_aloha.config import PolicyProfile


def validate_server_metadata(metadata: object, profile: PolicyProfile, source_sha: str, backend: str = "jax") -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("server metadata must be a mapping")
    expected = {
        "policy_profile": profile.name,
        "config_name": profile.config_name,
        "checkpoint_label": profile.checkpoint_label,
        "checkpoint_variant": profile.checkpoint_label + ("_pytorch" if backend == "pytorch" else ""),
        "policy_backend": backend,
        "action_horizon": profile.action_horizon,
        "action_dimension": profile.action_dimension,
        "source_sha": source_sha,
        "compact_masked_images": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"server metadata {key!r} does not match the requested profile")
    platform_key = "jax_platform" if backend == "jax" else "torch_platform"
    device_key = "jax_device" if backend == "jax" else "torch_device"
    if metadata.get(platform_key) not in {"gpu", "cuda"}:
        raise ValueError(f"server metadata must identify the {backend} GPU platform")
    device = metadata.get(device_key)
    if not isinstance(device, str) or "3090" not in device:
        raise ValueError("server metadata must identify the RTX 3090 device")
    if backend == "pytorch" and metadata.get("torch_model_device") != "cuda:0":
        raise ValueError("server metadata must place the PyTorch model on cuda:0")


def validate_policy_response(response: object, profile: PolicyProfile) -> np.ndarray:
    if not isinstance(response, Mapping) or "actions" not in response:
        raise ValueError("policy response must contain actions")
    actions = np.asarray(response["actions"])
    expected = (profile.action_horizon, profile.action_dimension)
    if actions.shape != expected or not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
        raise ValueError(f"actions must be finite floating values with shape {expected}")
    return actions


def validate_policy_action(action: object, profile: PolicyProfile) -> np.ndarray:
    array = np.asarray(action)
    expected = (profile.action_dimension,)
    if array.shape != expected or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f"action must be finite floating values with shape {expected}")
    return array


def validate_server_timing(response: object) -> dict[str, Real]:
    timing = response.get("server_timing") if isinstance(response, Mapping) else None
    if not isinstance(timing, Mapping):
        raise ValueError("policy response must contain server timing")
    result: dict[str, Real] = {}
    for key in ("infer_ms", "prev_total_ms"):
        if key == "prev_total_ms" and key not in timing:
            continue
        value = timing.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"server timing {key!r} must be finite and nonnegative")
        result[key] = value
    return result


def validate_policy_timing(response: object) -> dict[str, Real]:
    timing = response.get("policy_timing") if isinstance(response, Mapping) else None
    if not isinstance(timing, Mapping):
        raise ValueError("policy response must contain policy timing")
    result: dict[str, Real] = {}
    for key in (
        "infer_ms",
        "input_transfer_ms",
        "model_ms",
        "output_transfer_ms",
        "vision_ms",
        "language_embed_ms",
        "prefix_kv_ms",
        "denoise_ms",
        "model_stages_ms",
    ):
        if key not in timing:
            continue
        value = timing[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"policy timing {key!r} must be finite and nonnegative")
        result[key] = value
    if "infer_ms" not in result:
        raise ValueError("policy timing 'infer_ms' must be finite and nonnegative")
    return result


def validate_timing_reconciliation(policy_timing: Mapping[str, Real], server_timing: Mapping[str, Real]) -> None:
    stages = ("vision_ms", "language_embed_ms", "prefix_kv_ms", "denoise_ms")
    if not any(key in policy_timing for key in stages):
        return
    required = {*stages, "model_stages_ms", "model_ms", "input_transfer_ms", "output_transfer_ms", "infer_ms"}
    if not required <= policy_timing.keys():
        raise ValueError("PyTorch policy timing stages are incomplete")
    model_stages_ms = float(policy_timing["model_stages_ms"])
    stage_sum = sum(float(policy_timing[key]) for key in stages)
    stage_tolerance = max(0.1, model_stages_ms * 0.01)
    host_tolerance = max(5.0, float(policy_timing["infer_ms"]) * 0.05)
    if abs(stage_sum - model_stages_ms) > stage_tolerance:
        raise ValueError("policy timing stages do not reconcile")
    if model_stages_ms > float(policy_timing["model_ms"]) + host_tolerance:
        raise ValueError("policy timing model stages exceed synchronized model time")
    accounted = sum(float(policy_timing[key]) for key in ("input_transfer_ms", "model_ms", "output_transfer_ms"))
    if accounted > float(policy_timing["infer_ms"]) + host_tolerance:
        raise ValueError("policy timing components exceed total inference time")
    if float(policy_timing["infer_ms"]) > float(server_timing["infer_ms"]) + host_tolerance:
        raise ValueError("policy timing exceeds server inference time")
