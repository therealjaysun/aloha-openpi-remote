from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy
_BENCHMARK_NOISE_SEED = "__openpi_benchmark_noise_seed"


def _validate_benchmark_noise_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("benchmark noise seed must be a uint32")
    return seed


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        compact_masked_images: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._compact_masked_images = compact_masked_images

        if self._is_pytorch_model:
            if next(self._model.parameters()).device != torch.device(pytorch_device):
                self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        benchmark_noise_seed = None
        if _BENCHMARK_NOISE_SEED in obs:
            benchmark_noise_seed = obs[_BENCHMARK_NOISE_SEED]
            if noise is not None or not self._is_pytorch_model:
                raise ValueError("fixed benchmark noise requires PyTorch and no explicit noise")
            obs = {key: value for key, value in obs.items() if key != _BENCHMARK_NOISE_SEED}
            benchmark_noise_seed = _validate_benchmark_noise_seed(benchmark_noise_seed)
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._compact_masked_images:
            present = [name for name, mask in inputs["image_mask"].items() if bool(mask)]
            if not present:
                raise ValueError("at least one image must be present")
            inputs["image"] = {name: inputs["image"][name] for name in present}
            inputs["image_mask"] = {name: inputs["image_mask"][name] for name in present}
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            policy_started = time.monotonic()
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if self._is_pytorch_model:
            pytorch_device = torch.device(self._pytorch_device)
            if pytorch_device.type == "cuda":
                torch.cuda.synchronize(pytorch_device)
            input_transfer_ms = (time.monotonic() - policy_started) * 1000

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if benchmark_noise_seed is not None:
            generator = torch.Generator(device=self._pytorch_device).manual_seed(benchmark_noise_seed)
            sample_kwargs["noise"] = torch.normal(
                mean=0.0,
                std=1.0,
                size=(1, self._model.config.action_horizon, self._model.config.action_dim),
                generator=generator,
                dtype=torch.float32,
                device=self._pytorch_device,
            )
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if self._is_pytorch_model:
            if pytorch_device.type == "cuda":
                torch.cuda.synchronize(pytorch_device)
            model_ms = (time.monotonic() - start_time) * 1000
            cuda_timing = getattr(self._model, "last_inference_cuda_timing", dict)()
            output_started = time.monotonic()
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            output_transfer_ms = (time.monotonic() - output_started) * 1000
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            model_ms = (time.monotonic() - start_time) * 1000

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = (
            {
                "infer_ms": (time.monotonic() - policy_started) * 1000,
                "input_transfer_ms": input_transfer_ms,
                "model_ms": model_ms,
                "output_transfer_ms": output_transfer_ms,
                **cuda_timing,
            }
            if self._is_pytorch_model
            else {"infer_ms": model_ms}
        )
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def pytorch_model_device(self) -> str:
        if not self._is_pytorch_model:
            raise ValueError("JAX policies do not have a PyTorch model device")
        return str(next(self._model.parameters()).device)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
