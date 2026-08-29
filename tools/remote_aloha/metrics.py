from __future__ import annotations

import json
from pathlib import Path

from tools.remote_aloha.config import load_mac_sim_config
from tools.remote_aloha.config import load_remote_config
from tools.remote_aloha.remote import RemoteError
from tools.remote_aloha.run import _validated_output_root
from tools.remote_aloha.run import _write_performance_summary


def summarize_latest() -> Path:
    output = _validated_output_root(load_mac_sim_config().output_dir)
    profile = load_remote_config().policy_profile.name
    candidates = sorted((output / "phase05").glob(f"*/{profile}/summary.json"))
    if not candidates:
        raise RemoteError(f"no Phase 5 run exists for {profile}; run make run first")
    summary_path = candidates[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("profile") != profile:
        raise RemoteError("the latest Phase 5 summary is invalid")
    gpu_path = summary_path.parent / "gpu-metrics.jsonl"
    _write_performance_summary(summary_path.parent, summary, gpu_path if gpu_path.exists() else None)
    return summary_path.parent / "performance-summary.json"


def main() -> None:
    try:
        print(summarize_latest())
    except (OSError, ValueError, json.JSONDecodeError, RemoteError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
