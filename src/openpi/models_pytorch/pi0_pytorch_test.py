from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def test_pi05_sampler_uses_exact_bounded_loop_and_reuses_denoise_inputs():
    class PrefixModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma = SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
            )
            self.gemma_expert = SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
            )
            self.inference_modes = []

        def forward(self, **_kwargs):
            self.inference_modes.append(torch.is_inference_mode_enabled())
            return None, object()

    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.pi05 = True
    model.config = SimpleNamespace(action_horizon=4, action_dim=2)
    model.action_in_proj = torch.nn.Linear(2, 4, bias=False)
    model.paligemma_with_expert = PrefixModel()
    state = torch.zeros((1, 14))
    model._preprocess_observation = lambda _observation, *, train: ([], [], None, None, state)  # noqa: SLF001
    prefix_embs = torch.zeros((1, 3, 4))
    prefix_pad_masks = torch.tensor([[True, False, True]])
    prefix_att_masks = torch.zeros((1, 3), dtype=torch.bool)
    model.embed_prefix = lambda *_args: (prefix_embs, prefix_pad_masks, prefix_att_masks)

    calls = []

    def denoise_step(_state, _prefix_masks, cache, x_t, timestep, **prepared):
        calls.append((timestep.clone(), prepared, cache, torch.is_inference_mode_enabled()))
        return x_t * 0.1 + timestep[:, None, None]

    model.denoise_step = denoise_step
    observation = SimpleNamespace(state=state)
    noise = torch.zeros((1, 4, 2))
    actual = model.sample_actions("cpu", observation, noise=noise, num_steps=4)

    expected = noise
    expected_times = []
    dt = torch.tensor(-0.25)
    time = torch.tensor(1.0)
    while time >= -dt / 2:
        expected_times.append(time.clone())
        expected = expected + dt * (expected * 0.1 + time.expand(1)[:, None, None])
        time += dt

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack([call[0] for call in calls]).flatten(), torch.stack(expected_times))
    assert len(calls) == 4
    assert all(call[3] for call in calls)
    assert all(model.paligemma_with_expert.inference_modes)
    assert model.paligemma_with_expert.paligemma.language_model.config._attn_implementation == "eager"  # noqa: SLF001
    assert model.paligemma_with_expert.gemma_expert.model.config._attn_implementation == "eager"  # noqa: SLF001
    assert len({call[1]["denoise_attention_mask"].data_ptr() for call in calls}) == 1
    assert len({call[1]["denoise_position_ids"].data_ptr() for call in calls}) == 1
    assert len({id(call[2]) for call in calls}) == 1

    suffix_pad_masks = torch.ones((1, 4), dtype=torch.bool)
    suffix_att_masks = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    expected_mask = model._prepare_attention_masks_4d(  # noqa: SLF001
        torch.cat(
            [
                prefix_pad_masks[:, None, :].expand(1, 4, 3),
                make_att_2d_masks(suffix_pad_masks, suffix_att_masks),
            ],
            dim=2,
        )
    )
    torch.testing.assert_close(calls[0][1]["denoise_attention_mask"], expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1]["denoise_position_ids"], torch.tensor([[2, 3, 4, 5]]), rtol=0, atol=0)

    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            model.sample_actions("cpu", observation, noise=noise, num_steps=invalid)

    calls.clear()
    model.sample_actions("cpu", observation, noise=noise)
    assert len(calls) == 10


def test_pi05_precomputed_denoise_inputs_match_legacy_path():
    class ExpertModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = SimpleNamespace(_attn_implementation=None)
            self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=config))
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            suffix_embs = kwargs["inputs_embeds"][1]
            adarms_cond = kwargs["adarms_cond"][1][:, None, :]
            return [None, suffix_embs + adarms_cond], None

    model = object.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.pi05 = True
    model.config = SimpleNamespace(action_horizon=4)
    model.gradient_checkpointing_enabled = False
    model.action_in_proj = torch.nn.Linear(2, 4)
    model.time_mlp_in = torch.nn.Linear(4, 4)
    model.time_mlp_out = torch.nn.Linear(4, 4)
    model.action_out_proj = torch.nn.Linear(4, 2)
    model.paligemma_with_expert = ExpertModel()

    state = torch.zeros((1, 14))
    prefix_pad_masks = torch.tensor([[True, False, True]])
    cache = object()
    x_t = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) / 10
    timestep = torch.tensor([0.5])
    baseline = model.denoise_step(state, prefix_pad_masks, cache, x_t, timestep)
    baseline_call = model.paligemma_with_expert.calls[-1]

    optimized = model.denoise_step(
        state,
        prefix_pad_masks,
        cache,
        x_t,
        timestep,
        denoise_attention_mask=baseline_call["attention_mask"],
        denoise_position_ids=baseline_call["position_ids"],
    )
    optimized_call = model.paligemma_with_expert.calls[-1]

    torch.testing.assert_close(optimized, baseline, rtol=0, atol=0)
    torch.testing.assert_close(optimized_call["inputs_embeds"][1], baseline_call["inputs_embeds"][1], rtol=0, atol=0)
    torch.testing.assert_close(optimized_call["adarms_cond"][1], baseline_call["adarms_cond"][1], rtol=0, atol=0)
    assert optimized_call["attention_mask"] is baseline_call["attention_mask"]
    assert optimized_call["position_ids"] is baseline_call["position_ids"]
    assert optimized_call["past_key_values"] is cache
