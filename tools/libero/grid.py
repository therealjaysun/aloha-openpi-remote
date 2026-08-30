from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import imageio_ffmpeg

from tools.remote_aloha.config import validate_output_root
from tools.remote_aloha.run import _atomic_json
from tools.remote_aloha.sim_smoke_test import verify_video

TRIALS = 12
ROWS = 3
COLUMNS = 4
FPS = 20
FRAMES = 2400
TILE_SHAPE = (224, 448, 3)
GRID_SHAPE = (ROWS * TILE_SHAPE[0], COLUMNS * TILE_SHAPE[1], 3)
FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_grid_inputs(manifest_paths: list[Path]) -> list[dict[str, object]]:
    if len(manifest_paths) != TRIALS:
        raise ValueError(f"grid requires exactly {TRIALS} manifests")
    entries = []
    for index, source in enumerate(manifest_paths):
        manifest_path = source.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            episode = manifest["episode"]
            video = manifest["video"]
            layout = manifest["layout"]
            seed = manifest["seed"]
            valid = (
                manifest["status"] == "complete"
                and manifest["infrastructure_pass"] is True
                and manifest["profile"] == "pi05_libero"
                and manifest["scenario"] == "push_pi"
                and episode["steps_applied"] == episode["step_limit"] == FRAMES
                and episode["policy_seconds"] == FRAMES / FPS
                and episode["control_hz"] == FPS
                and video["status"] == "complete"
                and video["frames"] == FRAMES
                and video["validation"]["fps"] == FPS
                and video["validation"]["frames"] == FRAMES
                and video["validation"]["shape"] == list(TILE_SHAPE)
                and isinstance(seed, int)
                and isinstance(layout["layout_hash"], str)
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"incomplete grid manifest: {manifest_path}") from error
        if not valid:
            raise ValueError(f"ineligible grid manifest: {manifest_path}")
        video_path = Path(video["path"])
        if not video_path.is_absolute():
            video_path = (Path.cwd() / video_path).resolve()
        if video_path != manifest_path.parent / "episode.mp4":
            raise ValueError(f"manifest video path escapes its episode directory: {manifest_path}")
        validation = verify_video(video_path, FRAMES, TILE_SHAPE, expected_fps=FPS)
        entries.append(
            {
                "index": index,
                "row": index // COLUMNS,
                "column": index % COLUMNS,
                "label": f"Trial {index + 1:02d} seed {seed}",
                "seed": seed,
                "source_sha": manifest["source_sha"],
                "scene_hash": manifest["scene_hash"],
                "layout": layout,
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "video": str(video_path),
                "video_sha256": _sha256(video_path),
                "video_validation": validation,
                "task_success": episode["task_success"],
                "best_coverage": episode["best_coverage"],
                "final_coverage": episode["final_coverage"],
            }
        )
    if len({entry["source_sha"] for entry in entries}) != 1:
        raise ValueError("grid inputs must use one source commit")
    if len({entry["seed"] for entry in entries}) != TRIALS:
        raise ValueError("grid input seeds must be unique")
    if len({entry["layout"]["layout_hash"] for entry in entries}) != TRIALS:
        raise ValueError("grid input layouts must be unique")
    return entries


def build_filter_graph(entries: list[dict[str, object]]) -> str:
    filters = [
        (
            f"[{index}:v]trim=start_frame=0:end_frame={FRAMES},setpts=N/({FPS}*TB),"
            f"drawtext=fontfile='{FONT}':text='{entry['label']}':x=8:y=8:fontsize=18:fontcolor=white:"
            f"box=1:boxcolor=black@0.55:boxborderw=4[v{index}]"
        )
        for index, entry in enumerate(entries)
    ]
    filters.append(
        "".join(f"[v{index}]" for index in range(TRIALS))
        + f"xstack=inputs={TRIALS}:grid={COLUMNS}x{ROWS}:shortest=1:fill=black[v]"
    )
    return ";".join(filters)


def stitch_grid(manifest_paths: list[Path], output_dir: Path) -> dict[str, object]:
    entries = validate_grid_inputs(manifest_paths)
    if not FONT.is_file():
        raise ValueError(f"grid label font is unavailable: {FONT}")
    output_dir = validate_output_root(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    output_path = output_dir / "push-pi-12-trial-grid.mp4"
    partial = output_dir / "push-pi-12-trial-grid.partial.mp4"
    graph = build_filter_graph(entries)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-y"]
    for entry in entries:
        command.extend(("-i", entry["video"]))
    command.extend(
        (
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-frames:v",
            str(FRAMES),
            "-fps_mode",
            "cfr",
            "-r",
            str(FPS),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        )
    )
    try:
        subprocess.run(command, check=True, timeout=1800)
        subprocess.run(
            (ffmpeg, "-v", "error", "-xerror", "-i", str(partial), "-map", "0:v:0", "-an", "-f", "null", "-"),
            check=True,
            timeout=600,
        )
        validation = verify_video(partial, FRAMES, GRID_SHAPE, expected_fps=FPS)
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
    result = {
        "schema": 1,
        "status": "complete",
        "source_sha": entries[0]["source_sha"],
        "scene_hash": entries[0]["scene_hash"],
        "layout": {"rows": ROWS, "columns": COLUMNS, "order": "row-major"},
        "video": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "duration_seconds": FRAMES / FPS,
            **validation,
        },
        "filter_sha256": hashlib.sha256(graph.encode()).hexdigest(),
        "inputs": entries,
    }
    _atomic_json(output_dir / "grid-manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("manifests", nargs=TRIALS, type=Path)
    args = parser.parse_args()
    result = stitch_grid(args.manifests, args.output_dir)
    print(
        json.dumps({"manifest": str(args.output_dir / "grid-manifest.json"), "video": result["video"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
