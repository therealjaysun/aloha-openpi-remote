import json
from pathlib import Path

from tools.libero import grid as libero_grid


def test_grid_preflight_requires_twelve_distinct_complete_trials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        libero_grid,
        "verify_video",
        lambda *_args, **_kwargs: {"bytes": 1, "fps": 20.0, "frames": 2400, "shape": [224, 448, 3]},
    )
    manifests = []
    for index in range(12):
        episode_dir = tmp_path / f"trial-{index}" / f"seed-{index}"
        episode_dir.mkdir(parents=True)
        video = episode_dir / "episode.mp4"
        video.write_bytes(bytes([index]))
        manifest = episode_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "infrastructure_pass": True,
                    "profile": "pi05_libero",
                    "scenario": "push_pi",
                    "source_sha": "a" * 40,
                    "scene_hash": "b" * 64,
                    "seed": index,
                    "layout": {"block_xy": [index, 0], "target_xy": [0, index], "layout_hash": f"{index:064x}"},
                    "episode": {
                        "steps_applied": 2400,
                        "step_limit": 2400,
                        "policy_seconds": 120.0,
                        "control_hz": 20,
                        "task_success": False,
                        "best_coverage": 0.0,
                        "final_coverage": {"pi": 0.0, "overall": 0.0},
                    },
                    "video": {
                        "status": "complete",
                        "frames": 2400,
                        "path": str(video),
                        "validation": {"fps": 20.0, "frames": 2400, "shape": [224, 448, 3]},
                    },
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)

    entries = libero_grid.validate_grid_inputs(manifests)

    assert [entry["seed"] for entry in entries] == list(range(12))
    assert "xstack=inputs=12:grid=4x3" in libero_grid.build_filter_graph(entries)
