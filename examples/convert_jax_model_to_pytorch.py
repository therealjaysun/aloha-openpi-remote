#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # Convert to PyTorch:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:
    # pi0_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
"""

import gc
import json
import os
import pathlib
import shutil
from typing import Literal

from flax import traverse_util
from flax.nnx import traversals
from huggingface_hub import load_torch_model
from huggingface_hub import save_torch_model
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        if "pi05" in checkpoint_dir:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # Handle final norm layer
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    # Use repository restore utility to load a pure dict of params (value suffix removed)
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


_BASE_LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"
_PROBE_SOURCE = "PaliGemma/img/embedding/kernel"


def _logical_source_key(key_path: tuple[str, ...]) -> str:
    if key_path[-1:] == ("value",):
        key_path = key_path[:-1]
    return "/".join(key_path)


def _partial_restore(
    checkpointer: ocp.PyTreeCheckpointer,
    params_path: pathlib.Path,
    metadata_tree,
    source_key: str,
    dtype,
):
    flat_metadata = traverse_util.flatten_dict(metadata_tree)
    matches = [key_path for key_path in flat_metadata if _logical_source_key(key_path) == source_key]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint leaf for {source_key!r}, found {len(matches)}")

    selected_path = matches[0]
    partial_tree = traverse_util.unflatten_dict(
        {
            key_path: metadata if key_path == selected_path else ocp.PLACEHOLDER
            for key_path, metadata in flat_metadata.items()
        }
    )
    cpu = jax.devices("cpu")
    if len(cpu) != 1:
        raise RuntimeError(f"Expected one JAX CPU device, found {len(cpu)}")
    sharding = jax.sharding.SingleDeviceSharding(cpu[0])
    item = {"params": partial_tree}
    restored = checkpointer.restore(
        params_path,
        ocp.args.PyTreeRestore(
            item=item,
            restore_args=jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=jax.Array, dtype=dtype, sharding=sharding), item
            ),
        ),
    )["params"]
    concrete = [value for value in traverse_util.flatten_dict(restored).values() if value is not ocp.PLACEHOLDER]
    if len(concrete) != 1:
        raise ValueError(f"Partial restore for {source_key!r} returned {len(concrete)} concrete leaves")
    return concrete[0]


def _mapped_tensors(source_key: str, value: torch.Tensor, model_config: openpi.models.pi0_config.Pi0Config):
    """Yield the stock converter's target tensors for one logical Orbax leaf."""
    if not source_key.startswith("PaliGemma/"):
        projection_keys = (
            {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
            if model_config.pi05
            else {
                "state_proj",
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            }
        )
        parts = source_key.split("/")
        if len(parts) == 2 and parts[0] in projection_keys and parts[1] in {"kernel", "bias"}:
            suffix = "weight" if parts[1] == "kernel" else "bias"
            yield f"{parts[0]}.{suffix}", value.T if parts[1] == "kernel" else value
            return
        raise KeyError(f"Unmapped checkpoint leaf: {source_key}")

    key = source_key.removeprefix("PaliGemma/")
    vision_prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    language_prefix = "paligemma_with_expert.paligemma.model.language_model"
    expert_prefix = "paligemma_with_expert.gemma_expert.model"
    vision_width = 1152
    vision_layers = 27
    language_width = 2048
    language_layers = 18
    language_heads = 8
    head_dim = 256
    expert_config = openpi.models.gemma.get_config("gemma_300m")

    single_mappings = {
        "img/embedding/bias": (f"{vision_prefix}.embeddings.patch_embedding.bias", lambda tensor: tensor),
        "img/pos_embedding": (
            f"{vision_prefix}.embeddings.position_embedding.weight",
            lambda tensor: tensor.reshape(-1, vision_width),
        ),
        "img/Transformer/encoder_norm/scale": (
            f"{vision_prefix}.post_layernorm.weight",
            lambda tensor: tensor,
        ),
        "img/Transformer/encoder_norm/bias": (
            f"{vision_prefix}.post_layernorm.bias",
            lambda tensor: tensor,
        ),
        "img/head/kernel": (
            "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
            lambda tensor: tensor.T,
        ),
        "img/head/bias": (
            "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
            lambda tensor: tensor,
        ),
        "llm/embedder/input_embedding": (f"{language_prefix}.embed_tokens.weight", lambda tensor: tensor),
        "llm/final_norm/scale": (f"{language_prefix}.norm.weight", lambda tensor: tensor),
    }
    if key == "img/embedding/kernel":
        yield f"{vision_prefix}.embeddings.patch_embedding.weight", value.permute(3, 2, 0, 1)
        return
    if key in single_mappings:
        target_key, transform = single_mappings[key]
        yield target_key, transform(value)
        return

    vision_stacks = {
        "img/Transformer/encoderblock/LayerNorm_0/scale": ("layer_norm1.weight", lambda tensor: tensor),
        "img/Transformer/encoderblock/LayerNorm_0/bias": ("layer_norm1.bias", lambda tensor: tensor),
        "img/Transformer/encoderblock/LayerNorm_1/scale": ("layer_norm2.weight", lambda tensor: tensor),
        "img/Transformer/encoderblock/LayerNorm_1/bias": ("layer_norm2.bias", lambda tensor: tensor),
        "img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel": ("mlp.fc1.weight", lambda tensor: tensor.T),
        "img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias": ("mlp.fc1.bias", lambda tensor: tensor),
        "img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel": ("mlp.fc2.weight", lambda tensor: tensor.T),
        "img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias": ("mlp.fc2.bias", lambda tensor: tensor),
    }
    if key in vision_stacks:
        suffix, transform = vision_stacks[key]
        for index in range(vision_layers):
            yield f"{vision_prefix}.encoder.layers.{index}.{suffix}", transform(value[index])
        return

    attention_match = key.removeprefix("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/")
    if attention_match in {
        "key/kernel",
        "key/bias",
        "value/kernel",
        "value/bias",
        "query/kernel",
        "query/bias",
        "out/kernel",
        "out/bias",
    }:
        component, parameter = attention_match.split("/")
        projection = {"key": "k_proj", "value": "v_proj", "query": "q_proj", "out": "out_proj"}[component]
        suffix = "weight" if parameter == "kernel" else "bias"
        for index in range(vision_layers):
            layer = value[index].reshape(-1, vision_width)
            transformed = layer.T if parameter == "kernel" else layer.reshape(-1)
            yield f"{vision_prefix}.encoder.layers.{index}.self_attn.{projection}.{suffix}", transformed
        return

    if key == "llm/layers/attn/q_einsum/w":
        for index in range(language_layers):
            yield (
                f"{language_prefix}.layers.{index}.self_attn.q_proj.weight",
                value[index].permute(0, 2, 1).reshape(language_heads * head_dim, language_width),
            )
        return
    if key == "llm/layers/attn/kv_einsum/w":
        for index in range(language_layers):
            yield f"{language_prefix}.layers.{index}.self_attn.k_proj.weight", value[index, 0, 0].T
            yield f"{language_prefix}.layers.{index}.self_attn.v_proj.weight", value[index, 1, 0].T
        return
    if key == "llm/layers/attn/attn_vec_einsum/w":
        for index in range(language_layers):
            yield (
                f"{language_prefix}.layers.{index}.self_attn.o_proj.weight",
                value[index].permute(2, 0, 1).reshape(language_heads * head_dim, language_width),
            )
        return
    if key == "llm/layers/mlp/gating_einsum":
        for index in range(language_layers):
            yield f"{language_prefix}.layers.{index}.mlp.gate_proj.weight", value[index, 0].T
            yield f"{language_prefix}.layers.{index}.mlp.up_proj.weight", value[index, 1].T
        return
    if key == "llm/layers/mlp/linear":
        for index in range(language_layers):
            yield f"{language_prefix}.layers.{index}.mlp.down_proj.weight", value[index].T
        return
    if key in {"llm/layers/pre_attention_norm/scale", "llm/layers/pre_ffw_norm/scale"}:
        norm = "input_layernorm" if "pre_attention" in key else "post_attention_layernorm"
        for index in range(language_layers):
            yield f"{language_prefix}.layers.{index}.{norm}.weight", value[index]
        return

    expert_attention = {
        "llm/layers/attn/q_einsum_1/w": "q",
        "llm/layers/attn/kv_einsum_1/w": "kv",
        "llm/layers/attn/attn_vec_einsum_1/w": "o",
    }
    if key in expert_attention:
        kind = expert_attention[key]
        for index in range(expert_config.depth):
            if kind == "q":
                yield (
                    f"{expert_prefix}.layers.{index}.self_attn.q_proj.weight",
                    value[index]
                    .permute(0, 2, 1)
                    .reshape(expert_config.num_heads * expert_config.head_dim, expert_config.width),
                )
            elif kind == "kv":
                yield f"{expert_prefix}.layers.{index}.self_attn.k_proj.weight", value[index, 0, 0].T
                yield f"{expert_prefix}.layers.{index}.self_attn.v_proj.weight", value[index, 1, 0].T
            else:
                yield (
                    f"{expert_prefix}.layers.{index}.self_attn.o_proj.weight",
                    value[index].reshape(expert_config.num_heads * expert_config.head_dim, expert_config.width).T,
                )
        return
    if key == "llm/layers/mlp_1/gating_einsum":
        for index in range(expert_config.depth):
            yield f"{expert_prefix}.layers.{index}.mlp.gate_proj.weight", value[index, 0].T
            yield f"{expert_prefix}.layers.{index}.mlp.up_proj.weight", value[index, 1].T
        return
    if key == "llm/layers/mlp_1/linear":
        for index in range(expert_config.depth):
            yield f"{expert_prefix}.layers.{index}.mlp.down_proj.weight", value[index].T
        return

    if not model_config.pi05 and key in {
        "llm/layers/pre_attention_norm_1/scale",
        "llm/layers/pre_ffw_norm_1/scale",
    }:
        norm = "input_layernorm" if "pre_attention" in key else "post_attention_layernorm"
        for index in range(expert_config.depth):
            yield f"{expert_prefix}.layers.{index}.{norm}.weight", value[index]
        return
    if not model_config.pi05 and key == "llm/final_norm_1/scale":
        yield f"{expert_prefix}.norm.weight", value
        return

    pi05_norms = {
        "llm/layers/pre_attention_norm_1/Dense_0/bias": ("input_layernorm.dense.bias", False),
        "llm/layers/pre_attention_norm_1/Dense_0/kernel": ("input_layernorm.dense.weight", True),
        "llm/layers/pre_ffw_norm_1/Dense_0/bias": ("post_attention_layernorm.dense.bias", False),
        "llm/layers/pre_ffw_norm_1/Dense_0/kernel": ("post_attention_layernorm.dense.weight", True),
    }
    if model_config.pi05 and key in pi05_norms:
        suffix, transpose = pi05_norms[key]
        for index in range(expert_config.depth):
            yield f"{expert_prefix}.layers.{index}.{suffix}", value[index].T if transpose else value[index]
        return
    if model_config.pi05 and key in {
        "llm/final_norm_1/Dense_0/bias",
        "llm/final_norm_1/Dense_0/kernel",
    }:
        suffix = "weight" if key.endswith("kernel") else "bias"
        yield f"{expert_prefix}.norm.dense.{suffix}", value.T if suffix == "weight" else value
        return
    raise KeyError(f"Unmapped checkpoint leaf: {source_key}")


def _bf16_digest(tensor: torch.Tensor) -> str:
    import hashlib

    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def probe_partial_bfloat16(checkpoint_dir: str, model_config: openpi.models.pi0_config.Pi0Config) -> None:
    """Prove selective BF16 restore and DLPack match the stock FP32-to-BF16 result for one real leaf."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("partial-bfloat16 restore requires JAX_PLATFORMS=cpu")
    params_path = pathlib.Path(checkpoint_dir) / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata_tree = checkpointer.metadata(params_path)["params"]
        direct = _partial_restore(checkpointer, params_path, metadata_tree, _PROBE_SOURCE, jnp.bfloat16)
        direct_torch = torch.from_dlpack(direct)
        direct_outputs = list(_mapped_tensors(_PROBE_SOURCE, direct_torch, model_config))
        if len(direct_outputs) != 1 or direct_outputs[0][1].dtype != torch.bfloat16:
            raise ValueError("BF16 probe did not produce one BF16 target tensor")
        direct_shape = tuple(direct_outputs[0][1].shape)
        direct_digest = _bf16_digest(direct_outputs[0][1])
        del direct_outputs, direct_torch, direct
        gc.collect()

        reference = _partial_restore(checkpointer, params_path, metadata_tree, _PROBE_SOURCE, jnp.float32)
        reference_torch = torch.from_dlpack(reference)
        reference_outputs = list(_mapped_tensors(_PROBE_SOURCE, reference_torch, model_config))
        reference_tensor = reference_outputs[0][1].to(torch.bfloat16)
        if tuple(reference_tensor.shape) != direct_shape or _bf16_digest(reference_tensor) != direct_digest:
            raise ValueError("Direct BF16 restore differs from the stock FP32-to-BF16 conversion")
    print(f"Partial BF16 probe passed for {_PROBE_SOURCE} with shape {direct_shape}.")


def convert_pi0_checkpoint_partial_bfloat16(
    checkpoint_dir: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
) -> None:
    """Convert one stored Orbax leaf at a time while the BF16 target stays on CUDA."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("partial-bfloat16 restore requires JAX_PLATFORMS=cpu")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("partial-bfloat16 conversion requires a CUDA GPU with BF16 support")

    output = pathlib.Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite conversion output: {output}")
    output.mkdir(mode=0o700, parents=False)
    params_path = pathlib.Path(checkpoint_dir) / "params"
    device = torch.device("cuda:0")
    with torch.device(device):
        model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    model.paligemma_with_expert.gemma_expert.lm_head = None
    model.to(dtype=torch.bfloat16)
    model.eval()
    target_state = model.state_dict()
    assigned: set[str] = set()

    with ocp.PyTreeCheckpointer() as checkpointer, torch.inference_mode():
        metadata_tree = checkpointer.metadata(params_path)["params"]
        source_keys = sorted({_logical_source_key(path) for path in traverse_util.flatten_dict(metadata_tree)})
        expected_source_count = 51 if model_config.pi05 else 50
        if len(source_keys) != expected_source_count:
            raise ValueError(f"Expected {expected_source_count} source leaves, found {len(source_keys)}")
        for source_key in source_keys:
            restored = _partial_restore(checkpointer, params_path, metadata_tree, source_key, jnp.bfloat16)
            source_tensor = torch.from_dlpack(restored)
            if source_tensor.device.type != "cpu" or source_tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"{source_key} restored as {source_tensor.device}/{source_tensor.dtype}, expected CPU/BF16"
                )
            produced = 0
            mappings = _mapped_tensors(source_key, source_tensor, model_config)
            for target_key, transformed in mappings:
                if target_key not in target_state:
                    raise KeyError(f"Unknown target parameter {target_key!r} from {source_key!r}")
                if target_key in assigned:
                    raise ValueError(f"Target parameter {target_key!r} was assigned more than once")
                target = target_state[target_key]
                if tuple(target.shape) != tuple(transformed.shape) or target.dtype != torch.bfloat16:
                    raise ValueError(
                        f"Target mismatch for {target_key}: expected {tuple(target.shape)}/{target.dtype}, "
                        f"got {tuple(transformed.shape)}/{transformed.dtype}"
                    )
                target.copy_(transformed, non_blocking=False)
                assigned.add(target_key)
                produced += 1
                del target, transformed
            if produced == 0:
                raise ValueError(f"Source leaf {source_key!r} produced no target parameters")
            torch.cuda.synchronize(device)
            del mappings, source_tensor, restored
            gc.collect()

    expected_target_count = 811 if model_config.pi05 else 776
    if len(assigned) != expected_target_count:
        raise ValueError(f"Expected {expected_target_count} mapped targets, assigned {len(assigned)}")
    allowed_unmapped = {_BASE_LM_HEAD}
    if set(target_state) - assigned != allowed_unmapped:
        raise ValueError(f"Unexpected unmapped targets: {sorted(set(target_state) - assigned)}")
    embedding_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if target_state[_BASE_LM_HEAD].data_ptr() != target_state[embedding_key].data_ptr():
        raise ValueError("The base language-model head is not tied to the mapped embedding")

    save_torch_model(
        model,
        output,
        max_shard_size="1GB",
        safe_serialization=True,
        shared_tensors_to_discard=[_BASE_LM_HEAD],
    )
    assets_source = pathlib.Path(checkpoint_dir) / "assets"
    if not assets_source.is_dir():
        raise FileNotFoundError(f"Checkpoint assets are missing: {assets_source}")
    shutil.copytree(assets_source, output / "assets")
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": "bfloat16",
        "restore_mode": "partial-bfloat16",
    }
    (output / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

    del target_state, model
    torch.cuda.empty_cache()
    with torch.device(device):
        verification_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    verification_model.paligemma_with_expert.gemma_expert.lm_head = None
    verification_model.to(dtype=torch.bfloat16)
    incompatible = load_torch_model(verification_model, output, strict=False, safe=True)
    if incompatible.unexpected_keys or not set(incompatible.missing_keys) <= {_BASE_LM_HEAD}:
        raise ValueError(
            f"Sharded checkpoint load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(f"Partial BF16 conversion completed with {len(assigned)} mapped target tensors.")


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    # Initialize checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
        model_config: Model config
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    # Process projection params
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Create configs based on checkpoint path
    # All models use the same PaliGemma config structure
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # Instantiate model
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    # Combine all parameters (no prefix needed for our model structure)
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Load state dict
    pi0_model.load_state_dict(all_params, strict=False)

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)

    # Save model weights as SafeTensors using save_model to handle tied weights
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir) / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
        "restore_mode": "full-float32",
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    restore_mode: Literal["full-float32", "partial-bfloat16"] = "full-float32",
    *,
    inspect_only: bool = False,
    partial_probe_only: bool = False,
):
    """Load JAX model and optionally convert to PyTorch.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        output_path: Path to save converted PyTorch model (required for conversion)
        precision: Precision for model conversion
        restore_mode: Keep the stock full-FP32 restore or use the bounded BF16 experiment
        inspect_only: Only inspect parameter keys, don't convert
        partial_probe_only: Validate one selective BF16 restore without writing a checkpoint
    """
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    if inspect_only:
        if partial_probe_only:
            raise ValueError("--inspect-only and --partial-probe-only are mutually exclusive")
        load_jax_model_and_print_keys(checkpoint_dir)
    elif restore_mode == "partial-bfloat16":
        if precision != "bfloat16":
            raise ValueError("partial-bfloat16 restore requires --precision bfloat16")
        if partial_probe_only:
            probe_partial_bfloat16(checkpoint_dir, model_config)
        else:
            if not output_path:
                raise ValueError("--output-path is required for partial-bfloat16 conversion")
            convert_pi0_checkpoint_partial_bfloat16(checkpoint_dir, output_path, model_config)
    else:
        if partial_probe_only:
            raise ValueError("--partial-probe-only requires --restore-mode partial-bfloat16")
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
