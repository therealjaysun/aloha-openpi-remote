import logging
import os
import pathlib
import tempfile

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(self, out_dir: pathlib.Path, subsample: int = 1, filename: str | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample
        self._filename = filename
        self._finalized = False
        self.output_path: pathlib.Path | None = None

    @property
    def frame_count(self) -> int:
        return len(self._images)

    @override
    def on_episode_start(self) -> None:
        self._images = []
        self._finalized = False
        self.output_path = None

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)

    @override
    def on_episode_end(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if not self._images:
            raise ValueError("cannot save an episode video without frames")
        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / (self._filename or f"out_{next_idx}.mp4")

        logging.info(f"Saving video to {out_path}")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=self._out_dir, delete=False) as stream:
                temporary = pathlib.Path(stream.name)
            imageio.mimwrite(
                temporary,
                [np.asarray(x) for x in self._images[:: self._subsample]],
                fps=50 // max(1, self._subsample),
            )
            os.replace(temporary, out_path)
            self.output_path = out_path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
