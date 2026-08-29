import numpy as np
from openpi_client import action_chunk_broker
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_trim_trailing_language_padding_keeps_every_valid_position():
    inputs = {
        "tokenized_prompt": np.arange(12).reshape(2, 6),
        "tokenized_prompt_mask": np.array(
            [[True, False, True, False, False, False], [True, True, False, False, True, False]]
        ),
    }
    _policy._trim_trailing_language_padding(inputs)  # noqa: SLF001
    np.testing.assert_array_equal(inputs["tokenized_prompt"], np.arange(12).reshape(2, 6)[:, :5])
    np.testing.assert_array_equal(
        inputs["tokenized_prompt_mask"],
        [[True, False, True, False, False], [True, True, False, False, True]],
    )

    with pytest.raises(ValueError, match="at least one valid token"):
        _policy._trim_trailing_language_padding(  # noqa: SLF001
            {"tokenized_prompt": np.zeros(3), "tokenized_prompt_mask": np.zeros(3, dtype=bool)}
        )
    for tokens, mask in ((np.zeros(()), np.zeros((), dtype=bool)), (np.zeros(2), np.zeros(3, dtype=bool))):
        with pytest.raises(ValueError, match="same non-scalar shape"):
            _policy._trim_trailing_language_padding(  # noqa: SLF001
                {"tokenized_prompt": tokens, "tokenized_prompt_mask": mask}
            )


def test_pi05_policy_trims_language_padding_before_sampling():
    class Model(torch.nn.Module):
        pi05 = True

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = type("Config", (), {"action_horizon": 50, "action_dim": 14})()
            self.prompts = []

        def sample_actions(self, _device, observation, noise=None):
            self.prompts.append((observation.tokenized_prompt.clone(), observation.tokenized_prompt_mask.clone()))
            return torch.zeros((1, self.config.action_horizon, self.config.action_dim))

    model = Model()
    policy = _policy.Policy(model, is_pytorch=True, pytorch_device="cpu")
    policy.infer(
        {
            "image": {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
            "state": np.zeros(14, dtype=np.float32),
            "tokenized_prompt": np.arange(6, dtype=np.int32),
            "tokenized_prompt_mask": np.array([True, False, True, False, False, False]),
        }
    )
    tokens, mask = model.prompts[0]
    assert tokens.shape == mask.shape == (1, 3)
    torch.testing.assert_close(tokens, torch.tensor([[0, 1, 2]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(mask, torch.tensor([[True, False, True]]), rtol=0, atol=0)


def test_fixed_benchmark_noise_is_local_exact_and_shape_safe():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = type("Config", (), {"action_horizon": 50, "action_dim": 32})()
            self.noises = []

        def sample_actions(self, device, observation, noise=None):
            if noise is None:
                noise = torch.zeros((1, self.config.action_horizon, self.config.action_dim), device=device)
            self.noises.append(noise.clone())
            return noise[:, :, :14]

    model = Model()
    policy = _policy.Policy(model, is_pytorch=True, pytorch_device="cpu")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    observation = {
        "image": {"base_0_rgb": image},
        "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
        "state": np.zeros(14, dtype=np.float32),
    }
    request = {**observation, "__openpi_benchmark_noise_seed": 7}
    first = policy.infer(request)["actions"]
    second = policy.infer(request)["actions"]
    np.testing.assert_array_equal(first, second)
    assert model.noises[0].shape == (1, 50, 32)
    assert "__openpi_benchmark_noise_seed" not in observation
    with pytest.raises(ValueError, match="uint32"):
        policy.infer({**observation, "__openpi_benchmark_noise_seed": None})
    assert policy.infer(observation)["actions"].shape == (50, 14)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
