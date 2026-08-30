import dataclasses
import enum
import logging
import subprocess
from typing import Literal

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Host to bind. The project wrapper passes loopback; the upstream-compatible default is retained.
    host: str = "0.0.0.0"
    # Record the policy's behavior for debugging.
    record: bool = False

    # Optional project profile identity and GPU requirements used by the remote ALOHA wrapper.
    policy_profile: str | None = None
    policy_backend: Literal["jax", "pytorch"] = "jax"
    pytorch_checkpoint_dir: str | None = None
    require_jax_platform: str | None = None
    require_jax_device: str | None = None
    require_torch_device: str | None = None
    compact_masked_images: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}

PROJECT_PROFILES = {
    "pi0_aloha_sim": (EnvMode.ALOHA_SIM, "pi0_aloha_sim", "pi0_aloha_sim"),
    "pi05_aloha_base": (EnvMode.ALOHA, "pi05_aloha", "pi05_base"),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, compact_masked_images: bool = False
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            compact_masked_images=compact_masked_images,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.policy_backend == "pytorch":
        if args.policy_profile is None or args.pytorch_checkpoint_dir is None:
            raise ValueError("PyTorch project policies require a profile and checkpoint directory")
        try:
            _, config_name, _ = PROJECT_PROFILES[args.policy_profile]
        except KeyError as error:
            raise ValueError("Unsupported project policy profile") from error
        train_config = _config.get_config(config_name)
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model,
                pytorch_compile_mode=None,
                pytorch_denoise_compile_mode="default" if train_config.model.pi05 else None,
            ),
        )
        return _policy_config.create_trained_policy(
            train_config,
            args.pytorch_checkpoint_dir,
            default_prompt=args.default_prompt,
            pytorch_device="cuda:0",
            compact_masked_images=args.compact_masked_images,
        )
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                compact_masked_images=args.compact_masked_images,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                compact_masked_images=args.compact_masked_images,
            )


def _profile_metadata(args: Args) -> dict:
    if args.policy_profile is None:
        return {}
    try:
        expected_env, config_name, checkpoint_label = PROJECT_PROFILES[args.policy_profile]
    except KeyError as error:
        raise ValueError("Unsupported project policy profile") from error
    if not isinstance(args.policy, Default) or args.env is not expected_env:
        raise ValueError("Project policy profile does not match the selected default environment")
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()
    return {
        "policy_profile": args.policy_profile,
        "config_name": config_name,
        "checkpoint_label": checkpoint_label,
        "checkpoint_variant": checkpoint_label + ("_pytorch" if args.policy_backend == "pytorch" else ""),
        "policy_backend": args.policy_backend,
        "action_horizon": 50,
        "action_dimension": 14,
        "source_sha": source_sha,
        "compact_masked_images": args.compact_masked_images,
    }


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = {**policy.metadata, **_profile_metadata(args)}

    if args.require_jax_platform is not None:
        import jax

        platform = jax.default_backend()
        devices = jax.devices()
        if platform != args.require_jax_platform:
            raise RuntimeError(f"Required JAX platform {args.require_jax_platform!r}, got {platform!r}")
        device_kinds = [device.device_kind for device in devices]
        if args.require_jax_device is not None and not any(args.require_jax_device in kind for kind in device_kinds):
            raise RuntimeError(f"Required JAX device containing {args.require_jax_device!r} was not found")
        selected_device = next(
            (kind for kind in device_kinds if args.require_jax_device is None or args.require_jax_device in kind),
            device_kinds[0],
        )
        policy_metadata.update({"jax_platform": platform, "jax_device": selected_device})

    if args.require_torch_device is not None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Required PyTorch CUDA device is unavailable")
        selected_device = torch.cuda.get_device_name(0)
        if args.require_torch_device not in selected_device:
            raise RuntimeError(f"Required PyTorch device containing {args.require_torch_device!r} was not found")
        model_device = policy.pytorch_model_device
        if not model_device.startswith("cuda:"):
            raise RuntimeError(f"PyTorch model is not on CUDA: {model_device}")
        policy_metadata.update(
            {"torch_platform": "cuda", "torch_device": selected_device, "torch_model_device": model_device}
        )

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info("Creating server on %s:%d", args.host, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
