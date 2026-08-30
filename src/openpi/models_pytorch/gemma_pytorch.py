from typing import Literal

import torch
from torch import nn
import transformers
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

_TRANSFORMERS_VERSION = "4.53.2"


class AdaptiveRMSNorm(nn.Module):
    """The only Gemma extension required by the pi0.5 action expert."""

    def __init__(self, dim: int, cond_dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3)

    def forward(self, inputs: torch.Tensor, cond: torch.Tensor):
        normalized = inputs * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + self.eps)
        modulation = self.dense(cond)
        if inputs.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normalized = normalized * (1 + scale.float()) + shift.float()
        return normalized.to(inputs.dtype), gate.to(inputs.dtype)


def _apply_norm(norm: nn.Module, inputs: torch.Tensor, cond: torch.Tensor | None):
    result = norm(inputs, cond) if isinstance(norm, AdaptiveRMSNorm) else norm(inputs)
    return result if isinstance(result, tuple) else (result, None)


def _gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor | None):
    return residual + update if gate is None else residual + update * gate


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        if transformers.__version__ != _TRANSFORMERS_VERSION:
            raise RuntimeError(f"OpenPI requires transformers=={_TRANSFORMERS_VERSION}, got {transformers.__version__}")

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        if use_adarms[1]:
            for layer in self.gemma_expert.model.layers:
                layer.input_layernorm = AdaptiveRMSNorm(
                    action_expert_config.width, action_expert_config.width, action_expert_config_hf.rms_norm_eps
                )
                layer.post_attention_layernorm = AdaptiveRMSNorm(
                    action_expert_config.width, action_expert_config.width, action_expert_config_hf.rms_norm_eps
                )
            self.gemma_expert.model.norm = AdaptiveRMSNorm(
                action_expert_config.width, action_expert_config.width, action_expert_config_hf.rms_norm_eps
            )

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        vision_model = self.paligemma.model.vision_tower.vision_model
        hidden_states = vision_model.embeddings(image)
        encoder_dtype = vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(encoder_dtype)
        hidden_states = vision_model.encoder(inputs_embeds=hidden_states).last_hidden_state
        hidden_states = vision_model.post_layernorm(hidden_states)
        return self.paligemma.model.multi_modal_projector(hidden_states)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def _forward_single_model(
        self,
        model,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values,
        *,
        use_cache: bool,
        cond: torch.Tensor | None,
    ):
        hidden_states = inputs_embeds.to(model.layers[0].self_attn.q_proj.weight.dtype)
        cache_position = torch.arange(position_ids.shape[1], device=position_ids.device) + (
            past_key_values.get_seq_length() if past_key_values is not None and use_cache else 0
        )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(model.layers):
            normalized, gate = _apply_norm(layer.input_layernorm, hidden_states, cond)
            input_shape = normalized.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                if use_cache:
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        layer_idx,
                        {"sin": sin, "cos": cos, "cache_position": cache_position},
                    )
                else:
                    key_states = torch.cat([past_key_values[layer_idx][0], key_states], dim=2)
                    value_states = torch.cat([past_key_values[layer_idx][1], value_states], dim=2)
            attention_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=layer.self_attn.scaling,
            )
            attention_output = layer.self_attn.o_proj(attention_output.reshape(*input_shape, -1).contiguous())
            hidden_states = _gated_residual(hidden_states, attention_output, gate)
            normalized, gate = _apply_norm(layer.post_attention_layernorm, hidden_states, cond)
            hidden_states = _gated_residual(hidden_states, layer.mlp(normalized), gate)

        hidden_states, _ = _apply_norm(model.norm, hidden_states, cond)
        return hidden_states, past_key_values if use_cache else None

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output, prefix_past_key_values = self._forward_single_model(
                self.paligemma.language_model,
                inputs_embeds[0],
                attention_mask,
                position_ids,
                past_key_values,
                use_cache=bool(use_cache),
                cond=adarms_cond[0],
            )
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output, _ = self._forward_single_model(
                self.gemma_expert.model,
                inputs_embeds[1],
                attention_mask,
                position_ids,
                past_key_values,
                use_cache=bool(use_cache),
                cond=adarms_cond[1],
            )
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = _apply_norm(  # noqa: PLW2901
                        layer.input_layernorm, hidden_states, adarms_cond[i]
                    )
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = _gated_residual(hidden_states, out_emb, gates[i])
                    after_first_residual = out_emb.clone()
                    out_emb, gate = _apply_norm(layer.post_attention_layernorm, out_emb, adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = _gated_residual(after_first_residual, out_emb, gate)
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = _apply_norm(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
