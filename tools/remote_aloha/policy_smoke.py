from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import time

import numpy as np
from openpi_client import msgpack_numpy

from tools.remote_aloha.config import get_policy_profile
from tools.remote_aloha.observation_contract import POLICY_CAMERA_VIEWS
from tools.remote_aloha.observation_contract import validate_policy_observation
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_policy_timing
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing
from tools.remote_aloha.policy_contract import validate_timing_reconciliation
from tools.remote_aloha.scenarios import SCENARIOS
from tools.remote_aloha.telemetry import summarize_values

_BENCHMARK_NOISE_SEED = "__openpi_benchmark_noise_seed"


def _observation_sha256(observation: dict) -> str:
    digest = hashlib.sha256()
    state = observation["state"]
    digest.update(state.dtype.str.encode())
    digest.update(state.tobytes())
    for name in POLICY_CAMERA_VIEWS:
        image = observation["images"][name]
        digest.update(name.encode())
        digest.update(image.tobytes())
    digest.update(str(observation.get("prompt", "")).encode())
    return digest.hexdigest()


def _load_policy_observation(path: Path, profile_name: str) -> tuple[dict, str]:
    manifest_path = path.parent / "manifest.json"
    try:
        metadata = path.lstat()
        manifest_metadata = manifest_path.lstat()
    except OSError:
        raise ValueError("captured observation evidence is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077 or not 1 <= metadata.st_size <= 2_000_000:
        raise ValueError("captured observation must be a private regular file of bounded size")
    if (
        not stat.S_ISREG(manifest_metadata.st_mode)
        or manifest_metadata.st_mode & 0o077
        or not 1 <= manifest_metadata.st_size <= 2_000_000
    ):
        raise ValueError("captured observation manifest must be a private regular file of bounded size")
    try:
        observation_bytes = path.read_bytes()
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        observation = msgpack_numpy.unpackb(observation_bytes)
    except Exception:
        raise ValueError("captured observation evidence is malformed") from None
    if (
        not isinstance(manifest, dict)
        or manifest.get("profile") != profile_name
        or not isinstance(manifest.get("source_sha"), str)
        or len(manifest["source_sha"]) != 40
        or any(character not in "0123456789abcdef" for character in manifest["source_sha"])
        or manifest.get("scenario") not in SCENARIOS
        or isinstance(manifest.get("seed"), bool)
        or not isinstance(manifest.get("seed"), int)
        or not 0 <= manifest["seed"] <= 2**32 - 1
    ):
        raise ValueError("captured observation manifest is invalid")
    validate_policy_observation(observation)
    return observation, manifest["source_sha"]


def run_policy_smoke(
    *,
    profile_name: str,
    backend: str,
    host: str,
    port: int,
    source_sha: str,
    connect_timeout: int | None = None,
    metadata_timeout: int | None = None,
    inference_timeout: int | None = None,
    close_timeout: int | None = None,
    warmup_requests: int = 3,
    measured_requests: int = 1,
    observation_path: Path | None = None,
    fixed_noise_seed: int | None = None,
) -> dict[str, object]:
    from openpi_client import websocket_client_policy

    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise ValueError("policy smoke requires valid IPv4 loopback")
    if not 1 <= warmup_requests <= 20 or not 1 <= measured_requests <= 100:
        raise ValueError("policy smoke requires 1-20 warmups and 1-100 measured requests")
    if fixed_noise_seed is not None and (
        isinstance(fixed_noise_seed, bool)
        or not isinstance(fixed_noise_seed, int)
        or not 0 <= fixed_noise_seed <= 2**32 - 1
    ):
        raise ValueError("fixed noise seed must be a uint32")
    profile = get_policy_profile(profile_name)
    if (observation_path is None) != (fixed_noise_seed is None):
        raise ValueError("captured observation and fixed noise seed must be provided together")
    capture_source_sha = None
    if observation_path is not None:
        if profile_name != "pi05_aloha_base" or backend != "pytorch":
            raise ValueError("fixed replay is limited to the π₀.₅ PyTorch profile")
        observation, capture_source_sha = _load_policy_observation(observation_path, profile_name)
    else:
        image = np.zeros((3, 224, 224), dtype=np.uint8)
        observation = {
            "state": np.zeros(14, dtype=np.float64),
            "images": {name: image for name in POLICY_CAMERA_VIEWS},
        }
        if profile.default_prompt is not None:
            observation["prompt"] = profile.default_prompt
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=host,
        port=port,
        connect_timeout=connect_timeout,
        metadata_timeout=metadata_timeout,
        inference_timeout=inference_timeout,
        close_timeout=close_timeout,
        retry_interval=1,
    )
    try:
        validate_server_metadata(policy.get_server_metadata(), profile, source_sha, backend)

        validate_policy_observation(observation)
        observation_sha256 = _observation_sha256(observation)
        request = dict(observation)
        if fixed_noise_seed is not None:
            request[_BENCHMARK_NOISE_SEED] = fixed_noise_seed
        latencies_ms = []
        server_timing = []
        policy_timing = []
        reference_actions = None
        for _ in range(warmup_requests + measured_requests):
            started = time.perf_counter_ns()
            response = policy.infer(request)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            actions = validate_policy_response(response, profile)
            if fixed_noise_seed is not None:
                if reference_actions is not None and not np.array_equal(actions, reference_actions):
                    raise ValueError("fixed-input/fixed-noise actions are not exactly stable")
                reference_actions = actions.copy()
            server_timing.append(validate_server_timing(response))
            policy_timing.append(validate_policy_timing(response))
            if backend == "pytorch" and "vision_ms" not in policy_timing[-1]:
                raise ValueError("PyTorch policy response must contain synchronized stage timing")
            validate_timing_reconciliation(policy_timing[-1], server_timing[-1])
        measured_latencies = latencies_ms[warmup_requests:]
        measured_server_infer = [float(timing["infer_ms"]) for timing in server_timing[warmup_requests:]]
        measured_policy_timing = policy_timing[warmup_requests:]
        result = {
            "profile": profile.name,
            "backend": backend,
            "source_sha": source_sha,
            "action_shape": list(actions.shape),
            "camera_views": list(POLICY_CAMERA_VIEWS),
            "observation_sha256": observation_sha256,
            "cold_latency_ms": latencies_ms[0],
            "warmup_requests": warmup_requests,
            "warmup_latency_ms": summarize_values(latencies_ms[1:warmup_requests]),
            "measured_requests": measured_requests,
            "warmed_latency_ms": summarize_values(measured_latencies),
            "server_infer_ms": summarize_values(measured_server_infer),
            "policy_timing": {
                key: summarize_values(timing[key] for timing in measured_policy_timing if key in timing)
                for key in sorted({key for timing in measured_policy_timing for key in timing})
            },
            "status": "passed",
        }
        if fixed_noise_seed is not None:
            result.update(
                {
                    "fixed_noise_seed": fixed_noise_seed,
                    "action_replay_exact": True,
                    "action_sha256": hashlib.sha256(np.ascontiguousarray(reference_actions).tobytes()).hexdigest(),
                    "capture_source_sha": capture_source_sha,
                }
            )
        return result
    finally:
        policy.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded WSL-local policy inference validation.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--backend", choices=("jax", "pytorch"), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--connect-timeout", type=int)
    parser.add_argument("--metadata-timeout", type=int)
    parser.add_argument("--inference-timeout", type=int)
    parser.add_argument("--close-timeout", type=int)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=1)
    parser.add_argument("--observation-path", type=Path)
    parser.add_argument("--fixed-noise-seed", type=int)
    args = parser.parse_args()
    try:
        summary = run_policy_smoke(
            profile_name=args.profile,
            backend=args.backend,
            host=args.host,
            port=args.port,
            source_sha=args.source_sha,
            connect_timeout=args.connect_timeout,
            metadata_timeout=args.metadata_timeout,
            inference_timeout=args.inference_timeout,
            close_timeout=args.close_timeout,
            warmup_requests=args.warmup_requests,
            measured_requests=args.measured_requests,
            observation_path=args.observation_path,
            fixed_noise_seed=args.fixed_noise_seed,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
