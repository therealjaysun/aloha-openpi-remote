from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tools.remote_aloha.config import get_policy_profile
from tools.remote_aloha.policy_contract import validate_policy_response
from tools.remote_aloha.policy_contract import validate_server_metadata
from tools.remote_aloha.policy_contract import validate_server_timing


def main() -> None:
    from openpi_client import websocket_client_policy

    parser = argparse.ArgumentParser(description="Run bounded WSL-local policy inference validation.")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535:
        parser.error("the Phase 2 policy smoke test requires valid IPv4 loopback")
    profile = get_policy_profile(args.profile)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    validate_server_metadata(policy.get_server_metadata(), profile, args.source_sha)

    image = np.zeros((3, 224, 224), dtype=np.uint8)
    observation = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {"cam_high": image},
    }
    latencies_ms = []
    server_timing = []
    for _ in range(4):
        started = time.perf_counter_ns()
        response = policy.infer(observation)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        actions = validate_policy_response(response, profile)
        server_timing.append(validate_server_timing(response))
    print(
        json.dumps(
            {
                "profile": profile.name,
                "source_sha": args.source_sha,
                "action_shape": list(actions.shape),
                "cold_latency_ms": latencies_ms[0],
                "warmup_latency_ms": latencies_ms[1:3],
                "warmed_latency_ms": latencies_ms[3],
                "server_timing": server_timing,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
