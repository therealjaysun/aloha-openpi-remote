from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tools.remote_aloha.config import get_policy_profile
from tools.remote_aloha.observation_contract import validate_policy_observation
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing


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
) -> dict[str, object]:
    from openpi_client import websocket_client_policy

    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise ValueError("policy smoke requires valid IPv4 loopback")
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
            "images": {"cam_high": image},
        }
        if profile.default_prompt is not None:
            observation["prompt"] = profile.default_prompt
        validate_policy_observation(observation)
        latencies_ms = []
        server_timing = []
        for _ in range(4):
            started = time.perf_counter_ns()
            response = policy.infer(observation)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            actions = validate_policy_response(response, profile)
            server_timing.append(validate_server_timing(response))
        return {
            "profile": profile.name,
            "backend": backend,
            "source_sha": source_sha,
            "action_shape": list(actions.shape),
            "cold_latency_ms": latencies_ms[0],
            "warmup_latency_ms": latencies_ms[1:3],
            "warmed_latency_ms": latencies_ms[3],
            "server_timing": server_timing,
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
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
