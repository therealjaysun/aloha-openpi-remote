import math
from types import SimpleNamespace

import torch
from transformers import GemmaConfig
from transformers import GemmaModel

from openpi.models_pytorch.gemma_pytorch import AdaptiveRMSNorm
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


def test_adaptive_rms_norm_matches_reference():
    norm = AdaptiveRMSNorm(4, 3, 1e-6)
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 4
    cond = torch.tensor([[0.25, -0.5, 0.75]])
    with torch.no_grad():
        norm.dense.weight.copy_(torch.arange(36, dtype=torch.float32).reshape(12, 3) / 100)
        norm.dense.bias.copy_(torch.arange(12, dtype=torch.float32) / 50)

    actual, gate = norm(inputs, cond)
    modulation = norm.dense(cond).unsqueeze(1)
    scale, shift, expected_gate = modulation.chunk(3, dim=-1)
    normalized = inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(actual, normalized * (1 + scale) + shift)
    torch.testing.assert_close(gate, expected_gate)
    assert set(norm.state_dict()) == {"dense.weight", "dense.bias"}


def test_pi05_expert_uses_local_adaptive_norms():
    config = SimpleNamespace(width=16, mlp_dim=32, num_heads=2, head_dim=8, depth=2, num_kv_heads=1)
    with torch.device("meta"):
        model = PaliGemmaWithExpertModel(config, config, use_adarms=[False, True], precision="float32")

    norms = [
        norm
        for layer in model.gemma_expert.model.layers
        for norm in (layer.input_layernorm, layer.post_attention_layernorm)
    ] + [model.gemma_expert.model.norm]
    assert all(isinstance(norm, AdaptiveRMSNorm) for norm in norms)
    assert all(set(norm.state_dict()) == {"dense.weight", "dense.bias"} for norm in norms)


def test_cached_single_model_matches_full_gemma_pass():
    torch.manual_seed(0)
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=32,
        attention_dropout=0.0,
    )
    model = GemmaModel(config).eval()
    wrapper = object.__new__(PaliGemmaWithExpertModel)
    torch.nn.Module.__init__(wrapper)
    prefix = torch.randn(1, 3, 16)
    suffix = torch.randn(1, 2, 16)
    positions = torch.arange(5).unsqueeze(0)
    minimum = torch.finfo(prefix.dtype).min
    full_mask = torch.full((1, 1, 5, 5), minimum)
    full_mask.masked_fill_(torch.ones(5, 5, dtype=torch.bool).tril()[None, None], 0)

    _, cache = wrapper._forward_single_model(  # noqa: SLF001
        model,
        prefix,
        full_mask[:, :, :3, :3],
        positions[:, :3],
        None,
        use_cache=True,
        cond=None,
    )
    actual, _ = wrapper._forward_single_model(  # noqa: SLF001
        model,
        suffix,
        full_mask[:, :, 3:, :],
        positions[:, 3:],
        cache,
        use_cache=False,
        cond=None,
    )
    expected = model(
        inputs_embeds=torch.cat([prefix, suffix], dim=1) / math.sqrt(config.hidden_size),
        attention_mask=full_mask,
        position_ids=positions,
        use_cache=False,
    ).last_hidden_state[:, 3:]

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
