from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tools.remote_aloha.config import get_policy_profile
from tools.remote_aloha.observation_contract import POLICY_CAMERA_VIEWS
from tools.remote_aloha.observation_contract import validate_policy_observation
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_policy_timing
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing
from tools.remote_aloha.policy_contract import validate_timing_reconciliation
from tools.remote_aloha.telemetry import summarize_values


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
) -> dict[str, object]:
    from openpi_client import websocket_client_policy

    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise ValueError("policy smoke requires valid IPv4 loopback")
    if not 1 <= warmup_requests <= 20 or not 1 <= measured_requests <= 100:
        raise ValueError("policy smoke requires 1-20 warmups and 1-100 measured requests")
    profile = get_policy_profile(profile_name)
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

        image = np.zeros((3, 224, 224), dtype=np.uint8)
        observation = {
            "state": np.zeros(14, dtype=np.float64),
            "images": {name: image for name in POLICY_CAMERA_VIEWS},
        }
        if profile.default_prompt is not None:
            observation["prompt"] = profile.default_prompt
        validate_policy_observation(observation)
        latencies_ms = []
        server_timing = []
        policy_timing = []
        for _ in range(warmup_requests + measured_requests):
            started = time.perf_counter_ns()
            response = policy.infer(observation)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            actions = validate_policy_response(response, profile)
            server_timing.append(validate_server_timing(response))
            policy_timing.append(validate_policy_timing(response))
            if backend == "pytorch" and "vision_ms" not in policy_timing[-1]:
                raise ValueError("PyTorch policy response must contain synchronized stage timing")
            validate_timing_reconciliation(policy_timing[-1], server_timing[-1])
        measured_latencies = latencies_ms[warmup_requests:]
        measured_server_infer = [float(timing["infer_ms"]) for timing in server_timing[warmup_requests:]]
        measured_policy_timing = policy_timing[warmup_requests:]
        return {
            "profile": profile.name,
            "backend": backend,
            "source_sha": source_sha,
            "action_shape": list(actions.shape),
            "camera_views": list(POLICY_CAMERA_VIEWS),
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
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
