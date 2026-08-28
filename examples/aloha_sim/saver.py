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


class LiveDisplay(_subscriber.Subscriber):
    """Optional local-only view of the same post-step policy frame saved to video."""

    def __init__(self, *, enabled: bool, every_steps: int = 5) -> None:
        self._enabled = enabled
        self._every_steps = every_steps
        self._step = 0
        self._figure = None
        self._axes = None
        self._image = None
        self._pyplot = None

    @override
    def on_episode_start(self) -> None:
        self._step = 0
        if not self._enabled:
            return
        import matplotlib.pyplot as plt

        plt.ion()
        self._pyplot = plt
        self._figure, self._axes = plt.subplots(num="Push-PI ALOHA simulation")
        self._axes.set_axis_off()
        self._axes.set_title("Post-step policy view")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        del action
        if not self._enabled:
            return
        self._step += 1
        if self._step != 1 and self._step % self._every_steps:
            return
        frame = np.transpose(np.asarray(observation["images"]["cam_high"]), (1, 2, 0))
        if self._image is None:
            self._image = self._axes.imshow(frame)
        else:
            self._image.set_data(frame)
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._pyplot.pause(0.001)

    @override
    def on_episode_end(self) -> None:
        if self._figure is not None:
            self._pyplot.close(self._figure)
        self._figure = self._axes = self._image = self._pyplot = None
