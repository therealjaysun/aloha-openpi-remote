from types import SimpleNamespace

import pytest
import torch

from openpi.models.pi0_config import Pi0Config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def test_pi05_compile_wraps_only_the_denoise_step(monkeypatch):
    calls = []

    class PaliGemma(torch.nn.Module):
        pass

    def compiled(*_args, **_kwargs):
        return None

    def compile_step(function, **kwargs):
        calls.append((function, kwargs))
        return compiled

    monkeypatch.setattr(
        pi0_pytorch._gemma,  # noqa: SLF001
        "get_config",
        lambda _variant: SimpleNamespace(width=4),
    )
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", lambda *_args, **_kwargs: PaliGemma())
    monkeypatch.setattr(torch, "compile", compile_step)
    config = SimpleNamespace(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
        action_dim=2,
        pytorch_compile_mode=None,
        pytorch_denoise_compile_mode="default",
    )

    model = PI0Pytorch(config)

    assert len(calls) == 1
    assert calls[0][0].__self__ is model
    assert calls[0][0].__func__ is PI0Pytorch.denoise_step
    assert calls[0][1] == {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False}
    assert model._compiled_denoise_step is compiled  # noqa: SLF001


def test_pi05_denoise_compile_requires_whole_sampler_eager():
    with pytest.raises(AssertionError):
        Pi0Config(pi05=True, pytorch_denoise_compile_mode="default")
    assert (
        Pi0Config(pi05=True, pytorch_compile_mode=None, pytorch_denoise_compile_mode="default").pytorch_compile_mode
        is None
    )


def test_pi05_sampler_uses_exact_bounded_loop_and_reuses_denoise_inputs():
    class PrefixModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = SimpleNamespace(_attn_implementation=None)
            self.paligemma = SimpleNamespace(language_model=SimpleNamespace(config=config))
            self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=config))
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

    def compiled_step(_state, _prefix_masks, cache, x_t, timestep, **prepared):
        calls.append((timestep.clone(), prepared, cache, torch.is_inference_mode_enabled(), x_t.is_inference()))
        return x_t * 0.1 + timestep[:, None, None]

    def eager_step(*_args, **_kwargs):
        raise AssertionError("compiled dispatch unexpectedly used the eager denoise step")

    model.denoise_step = eager_step
    model._compiled_denoise_step = compiled_step  # noqa: SLF001
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
    assert all(call[4] for call in calls)
    assert all(model.paligemma_with_expert.inference_modes)
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
    model.denoise_step = compiled_step
    model._compiled_denoise_step = None  # noqa: SLF001
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
