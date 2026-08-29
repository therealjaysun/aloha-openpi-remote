from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re

DEFAULT_TASK = "gym_aloha/AlohaTransferCube-v0"
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_UINT = re.compile(r"[0-9]+\Z")


@dataclass(frozen=True)
class MacSimConfig:
    task: str = DEFAULT_TASK
    seed: int = 0
    episodes: int = 3
    output_dir: Path = Path("outputs")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if not _KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid key {key!r}")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"{path}:{line_number}: unmatched quote")
            value = value[1:-1]
        values[key] = value
    return values


def _uint(name: str, value: str, *, maximum: int | None = None) -> int:
    if not _UINT.fullmatch(value):
        raise ValueError(f"{name} must be an unsigned decimal integer")
    parsed = int(value)
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return parsed


def load_mac_sim_config(env_file: str | Path = ".env", environ: Mapping[str, str] | None = None) -> MacSimConfig:
    values = _read_env_file(Path(env_file))
    source = os.environ if environ is None else environ
    for key in ("ALOHA_TASK", "ALOHA_SEED", "ALOHA_EPISODES", "RUN_OUTPUT_DIR"):
        if key in source:
            values[key] = source[key]

    task = values.get("ALOHA_TASK", DEFAULT_TASK)
    if task != DEFAULT_TASK:
        raise ValueError(f"ALOHA_TASK must be {DEFAULT_TASK!r} for Phase 01")
    seed = _uint("ALOHA_SEED", values.get("ALOHA_SEED", "0"), maximum=2**32 - 1)
    episodes = _uint("ALOHA_EPISODES", values.get("ALOHA_EPISODES", "3"))
    if episodes < 1:
        raise ValueError("ALOHA_EPISODES must be positive")
    if seed + episodes - 1 > 2**32 - 1:
        raise ValueError("ALOHA_SEED + ALOHA_EPISODES exceeds the seed range")
    output = values.get("RUN_OUTPUT_DIR", "outputs")
    if not output or "\x00" in output:
        raise ValueError("RUN_OUTPUT_DIR must be a nonempty path")
    return MacSimConfig(task=task, seed=seed, episodes=episodes, output_dir=Path(output))
