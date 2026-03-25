# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OLMoE model."""

import atexit
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    ModelOutput, 
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OlmoeConfig"


@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """


    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoeModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
    """
    
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class OlmoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        OlmoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(OlmoeRMSNorm)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Olmoe
class OlmoeRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[OlmoeConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`OlmoeRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.olmo.modeling_olmo.OlmoMLP with Olmo->Olmoe
class OlmoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
 
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class OlmoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OlmoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.q_norm = OlmoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(
            (self.hidden_size // self.num_heads) * self.num_key_value_heads, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class OlmoeFlashAttention2(OlmoeAttention):
    """
    OLMoE flash attention module. This module inherits from `OlmoeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)
        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (OlmoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class OlmoeSdpaAttention(OlmoeAttention):
    """
    OLMoE attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `OlmoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from OlmoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "OlmoeModel is using OlmoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


OLMOE_ATTENTION_CLASSES = {
    "eager": OlmoeAttention,
    "flash_attention_2": OlmoeFlashAttention2,
    "sdpa": OlmoeSdpaAttention,
}


class OlmoeSparseMoeBlock(nn.Module):
    _total_assignments: int = 0
    _dropped_assignments: int = 0
    _forward_calls: int = 0
    _atexit_registered: bool = False
    _per_layer_gamma: dict = None  # {layer_idx: gamma} — set externally for per-layer capacity

    @classmethod
    def _register_atexit(cls):
        if not cls._atexit_registered:
            atexit.register(cls._print_drop_summary)
            cls._atexit_registered = True

    @classmethod
    def reset_drop_stats(cls):
        cls._total_assignments = 0
        cls._dropped_assignments = 0
        cls._forward_calls = 0

    @classmethod
    def _print_drop_summary(cls):
        total = cls._total_assignments
        dropped = cls._dropped_assignments
        if total == 0:
            return
        ratio = dropped / total * 100
        print(
            f"\n{'='*60}\n"
            f"[OlmoeSparseMoeBlock] Token-Expert Drop Summary\n"
            f"  Total assignments : {total}\n"
            f"  Dropped           : {dropped}\n"
            f"  Drop ratio        : {ratio:.2f}%\n"
            f"  Forward calls     : {cls._forward_calls}\n"
            f"{'='*60}\n",
            flush=True,
        )

    def __init__(self, config, layer_idx):
        super().__init__()
        
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

        # topk selection algorithm 
        ### 🔍🔍🔍
        self.layer_idx = layer_idx
        self.expert_capacity = config.expert_capacity \
            if hasattr(config, "expert_capacity") and isinstance(config.expert_capacity, float) else None
        self.strategy = config.strategy if hasattr(config, "strategy") else None
        self.rounds = config.rounds if hasattr(config, "rounds") else None
        ###

        self.mean_expert = None
        self._mean_expert_built = False
        if self.strategy and "mean_expert_comp" in self.strategy:
            self.mean_expert = OlmoeMLP(config)

        self.k_override = self.top_k
        if self.strategy:
            m = re.search(r'score_k(\d+)', self.strategy)
            if m:
                self.k_override = int(m.group(1))

        self.low_rank = int(getattr(config, 'low_rank', 32)) \
            if hasattr(config, 'low_rank') else 32
        self._lr_weights = None
        self._low_rank_built = False
        self._original_topk_idx = None

        layer_gammas_str = os.environ.get('LAYER_GAMMAS', None)
        if layer_gammas_str and isinstance(layer_gammas_str, str) and OlmoeSparseMoeBlock._per_layer_gamma is None:
            gammas = [float(x) for x in layer_gammas_str.split(':')]
            OlmoeSparseMoeBlock._per_layer_gamma = {i: g for i, g in enumerate(gammas)}

        OlmoeSparseMoeBlock._register_atexit()

    def _build_mean_expert(self):
        """Lazily average all expert parameters into self.mean_expert."""
        with torch.no_grad():
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                avg_weight = torch.stack(
                    [getattr(e, name).weight for e in self.experts]
                ).mean(dim=0)
                getattr(self.mean_expert, name).weight.copy_(avg_weight)
        self._mean_expert_built = True

    def _build_low_rank_experts(self):
        """Lazily SVD-decompose each expert into rank-r approximations."""
        self._lr_weights = {}
        r = self.low_rank
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                self._lr_weights[i] = {}
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    W = getattr(expert, proj_name).weight
                    U, S, V = torch.svd_lowrank(W.float(), q=r)
                    sqrt_S = S.sqrt()
                    A = (U * sqrt_S).to(dtype=W.dtype, device=W.device)
                    B = (V * sqrt_S).t().to(dtype=W.dtype, device=W.device)
                    self._lr_weights[i][proj_name] = (A, B)
        self._low_rank_built = True

    def _lr_expert_forward(self, expert_idx, x):
        """Forward pass through low-rank approximation of expert_idx."""
        ws = self._lr_weights[expert_idx]
        gate = F.linear(F.linear(x, ws['gate_proj'][1]), ws['gate_proj'][0])
        up = F.linear(F.linear(x, ws['up_proj'][1]), ws['up_proj'][0])
        hidden = self.experts[expert_idx].act_fn(gate) * up
        return F.linear(F.linear(hidden, ws['down_proj'][1]), ws['down_proj'][0])
            
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
                
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        ### 🔍🔍🔍
        total_tokens = batch_size * sequence_length
        gamma = self.expert_capacity
        if OlmoeSparseMoeBlock._per_layer_gamma is not None and self.layer_idx in OlmoeSparseMoeBlock._per_layer_gamma:
            gamma = OlmoeSparseMoeBlock._per_layer_gamma[self.layer_idx]
        if sequence_length == 1 or gamma is None: 
            expert_capacity = None
        else:
            expert_capacity = math.ceil(gamma * self.k_override * (total_tokens / self.num_experts))
        ### 
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        ### 🔍🔍🔍 select top-k experts
        self._moe_input = hidden_states
        routing_weights, selected_experts = self.adjust_tokens(expert_capacity, routing_weights)
        self._moe_input = None
        expert_idxes = range(self.num_experts)

        if expert_capacity is not None:
            n_assign = selected_experts.numel()
            n_dropped = (selected_experts == self.num_experts).sum().item()
            OlmoeSparseMoeBlock._total_assignments += n_assign
            OlmoeSparseMoeBlock._dropped_assignments += n_dropped
            OlmoeSparseMoeBlock._forward_calls += 1
        ######
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
            
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected

        ### 🔍🔍🔍 select top-k experts
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts + 1).permute(2, 1, 0)

        comp_mode = None
        if expert_capacity is not None and self.strategy:
            if "identity_comp" in self.strategy:
                comp_mode = "identity"
            elif "mean_expert_comp" in self.strategy:
                comp_mode = "mean_expert"
            elif "top1_clone_comp" in self.strategy:
                comp_mode = "top1_clone"
            elif "low_rank_comp" in self.strategy:
                comp_mode = "low_rank"

        top1_cache = None
        if comp_mode == "top1_clone":
            survived_w = routing_weights.clone()
            survived_w[selected_experts == self.num_experts] = -float('inf')
            top1_slot = survived_w.argmax(dim=-1)
            top1_expert_id = selected_experts[
                torch.arange(selected_experts.size(0), device=selected_experts.device),
                top1_slot,
            ]
            top1_cache = torch.zeros_like(hidden_states)

        if comp_mode == "low_rank":
            if not self._low_rank_built:
                self._build_low_rank_experts()
            original_topk_idx = self._original_topk_idx
            dropped_mask = (selected_experts == self.num_experts)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in expert_idxes:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            if comp_mode == "top1_clone":
                raw_output = expert_layer(current_state)
                current_hidden_states = raw_output * routing_weights[top_x, idx, None]
                is_top1 = (top1_expert_id[top_x] == expert_idx)
                if is_top1.any():
                    top1_cache[top_x[is_top1]] = raw_output[is_top1].to(hidden_states.dtype)
            else:
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

            if comp_mode == "low_rank":
                dropped_here = (original_topk_idx == expert_idx) & dropped_mask
                if dropped_here.any():
                    drop_tokens, drop_slots = dropped_here.nonzero(as_tuple=True)
                    drop_state = hidden_states[drop_tokens]
                    lr_output = self._lr_expert_forward(expert_idx, drop_state)
                    lr_weighted = lr_output * routing_weights[drop_tokens, drop_slots, None]
                    final_hidden_states.index_add_(
                        0, drop_tokens, lr_weighted.to(hidden_states.dtype))

        if comp_mode is not None and comp_mode != "low_rank":
            dropped_mask = (selected_experts == self.num_experts)
            dropped_weights = routing_weights.clone()
            dropped_weights[~dropped_mask] = 0.0
            dropped_weight_sum = dropped_weights.sum(dim=-1)
            has_drops = dropped_weight_sum > 0
            if has_drops.any():
                dw = dropped_weight_sum[has_drops].unsqueeze(-1)
                if comp_mode == "identity":
                    final_hidden_states[has_drops] += dw * hidden_states[has_drops]
                elif comp_mode == "mean_expert":
                    if not self._mean_expert_built:
                        self._build_mean_expert()
                    mean_out = self.mean_expert(hidden_states[has_drops])
                    final_hidden_states[has_drops] += dw * mean_out.to(hidden_states.dtype)
                elif comp_mode == "top1_clone":
                    final_hidden_states[has_drops] += dw * top1_cache[has_drops]

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
                    
        return final_hidden_states, router_logits

    def _compute_contrib_norms(self, scores, topk_weight, topk_idx, use_lr=False):
        """Compute ||w_i * E_i(x)||_2 for each top-k assignment.
        If use_lr=True, uses low-rank expert approximations (much cheaper)."""
        T = scores.shape[0]
        top_k = self.top_k
        flat_hidden = self._moe_input

        if use_lr:
            if not self._low_rank_built:
                self._build_low_rank_experts()

        contrib_norms = torch.zeros(T, top_k, device=scores.device, dtype=torch.float32)
        for e in range(self.num_experts):
            match = topk_idx == e
            if not match.any():
                continue
            tok, slot = torch.where(match)
            h_sub = flat_hidden[tok]
            with torch.no_grad():
                if use_lr:
                    e_out = self._lr_expert_forward(e, h_sub).float()
                else:
                    e_out = self.experts[e](h_sub).float()
            w_e = topk_weight[tok, slot].float().unsqueeze(-1)
            contrib_norms[tok, slot] = (w_e * e_out).norm(dim=-1)
        return contrib_norms

    def _reroute_by_contrib(self, scores, expert_capacity, contrib_norms, topk_idx):
        """Enforce capacity using pre-computed contrib_norms as priority."""
        T, K = scores.shape
        top_k = self.top_k

        contrib_priority = torch.full(
            (T, K), -float('inf'), dtype=torch.float32, device=scores.device)
        for slot in range(top_k):
            tok_range = torch.arange(T, device=scores.device)
            expert_col = topk_idx[:, slot]
            contrib_priority[tok_range, expert_col] = contrib_norms[:, slot]

        mask_buffer = torch.zeros(T, K, dtype=torch.bool, device=scores.device)
        mask_buffer.scatter_(-1, topk_idx, True)

        usage = mask_buffer.sum(dim=0)
        cols = (usage > expert_capacity).nonzero(as_tuple=True)[0]
        if cols.numel() > 0:
            for c in cols:
                assigned = mask_buffer[:, c].nonzero(as_tuple=True)[0]
                if assigned.numel() <= expert_capacity:
                    continue
                prio = contrib_priority[assigned, c]
                _, keep_order = prio.topk(expert_capacity)
                keep_set = assigned[keep_order]
                mask_buffer[:, c] = False
                mask_buffer[keep_set, c] = True

        top_mask = mask_buffer.gather(-1, topk_idx)
        self._original_topk_idx = topk_idx.clone()
        topk_idx_out = topk_idx.masked_fill(~top_mask, self.num_experts)
        topk_weight_out = scores.gather(-1, topk_idx_out.clamp(max=K - 1))
        topk_weight_out = topk_weight_out.masked_fill(~top_mask, 0.0)
        return topk_weight_out, topk_idx_out

    def _reroute_contrib_oracle(self, scores, expert_capacity):
        """Oracle: use actual expert output norm as dropping priority (~2x compute)."""
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        contrib_norms = self._compute_contrib_norms(scores, topk_weight, topk_idx, use_lr=False)
        return self._reroute_by_contrib(scores, expert_capacity, contrib_norms, topk_idx)

    def _reroute_contrib_lr(self, scores, expert_capacity):
        """Low-rank oracle: use SVD-approximated expert output norm as dropping priority.
        Much cheaper than full oracle — rank-r matmuls instead of full expert forwards."""
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        contrib_norms = self._compute_contrib_norms(scores, topk_weight, topk_idx, use_lr=True)
        return self._reroute_by_contrib(scores, expert_capacity, contrib_norms, topk_idx)

    def adjust_tokens(self, expert_capacity, scores):

        if expert_capacity is None: 
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
            return topk_weight, topk_idx
        
        def _reroute_common(scores, expert_capacity, top_k):
            
            T, K = scores.shape
            top_k_all = top_k  
            
            topk_weight, topk_idx = torch.topk(scores, k=top_k_all, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)

            # Step 2: Enforce expert capacity
            current_usage = mask_buffer.sum(dim=0)
            cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
            if cols.numel() > 0:
                scores_sub = scores[:, cols]
                mask_sub = mask_buffer[:, cols]
                capacity_indices = compute_indices(scores_sub, mask_sub, expert_capacity)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            self._original_topk_idx = topk_idx.clone()
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)

            return topk_weight, topk_idx
        
        def compute_indices(scores_sub, mask_sub, expert_capacity):

            masked_scores = scores_sub.masked_fill(~mask_sub, float('-inf'))  # ascending 排序，mask无效
            _, capacity_indices = torch.topk(
                                        masked_scores, 
                                        k=expert_capacity, 
                                        dim=0, 
                                        sorted=False
                                        )

            return capacity_indices
        
        def reroute_random(scores, expert_capacity, top_k):
            
            T, K = scores.shape

            # Step 1: Initial top-k_all selection
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)

            # Step 2: Enforce expert capacity
            current_usage = mask_buffer.sum(dim=0)
            cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]

            if cols.numel() > 0:
                scores_sub = torch.rand_like(scores[:, cols])
                mask_sub = mask_buffer[:, cols]
                capacity_indices = compute_indices(scores_sub, mask_sub, expert_capacity)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            # Step 4: Final top_k gathering
            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)

            return topk_weight, topk_idx

        def reroute_sequential_order(scores, expert_capacity, top_k, mode):
            
            T, K = scores.shape
            # Step 1: Initial top-k_all selection
            topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)
            
            # Step 2: Enforce expert capacity
            current_usage = mask_buffer.sum(dim=0)
            cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]

            if cols.numel() > 0:
                mask_sub = mask_buffer[:, cols]
                if mode == "last":
                    scores_sub = torch.cumsum(mask_sub, dim=-2)
                elif mode == "first":
                    reversed_mask = torch.flip(mask_sub, dims=[-2])
                    reversed_cumsum = torch.cumsum(reversed_mask, dim=-2)
                    scores_sub = torch.flip(reversed_cumsum, dims=[-2])
                else:
                    raise ValueError("Unsupported mode: should be 'first' or 'last'")
                
                capacity_indices = compute_indices(scores_sub.float(), mask_sub, expert_capacity)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            # Step 4: Final top_k gathering
            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)

            return topk_weight, topk_idx

        def reroute_margin(scores, expert_capacity, top_k):
            """Margin-based dropping: keep tokens for which this expert is
            most *irreplaceable*, measured by the gap between the token's
            gate score at the overloaded expert and its best fallback
            (the (k+1)-th highest score).  Large margin → the token has
            no good alternative → prioritize keeping it."""

            T, K = scores.shape

            topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)

            # (k+1)-th best score per token = best unselected expert score
            topk_plus1_vals, _ = torch.topk(scores, k=top_k + 1, dim=-1, sorted=True)
            next_best = topk_plus1_vals[:, -1]                      # [T]
            margin_scores = scores - next_best.unsqueeze(-1)         # [T, K]

            current_usage = mask_buffer.sum(dim=0)
            cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
            if cols.numel() > 0:
                margin_sub = margin_scores[:, cols]
                mask_sub = mask_buffer[:, cols]
                capacity_indices = compute_indices(margin_sub, mask_sub, expert_capacity)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)

            return topk_weight, topk_idx

        def reroute_vulnerability(scores, expert_capacity, top_k):
            """Vulnerability-aware dropping: protect tokens whose assigned
            experts are severely overloaded.  For each token we sum the
            *continuous overflow* (excess tokens beyond capacity) of its
            top-k assigned experts.  Higher sum → the token is served by
            experts under heavier contention → it is more likely to lose
            assignments elsewhere → prioritize keeping it here.
            Gate score is used as a secondary tiebreaker."""

            T, K = scores.shape

            topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)

            current_usage = mask_buffer.sum(dim=0).float()                 # [K]
            overflow = torch.clamp(current_usage - expert_capacity, min=0) # [K]

            # Per-token vulnerability: sum of continuous overflow at its assigned experts.
            # Varies across tokens even when all experts are overloaded.
            token_vuln = overflow[topk_idx].sum(dim=-1)                    # [T]

            # Combine: vulnerability as primary, gate score as tiebreaker.
            # Normalize vulnerability to [0, 1] so the tiebreaker stays meaningful.
            max_vuln = token_vuln.max()
            if max_vuln > 0:
                norm_vuln = token_vuln / max_vuln
            else:
                norm_vuln = token_vuln
            vuln_scores = norm_vuln.unsqueeze(-1).expand_as(scores) + scores * 0.1

            cols = (overflow > 0).nonzero(as_tuple=True)[0]
            if cols.numel() > 0:
                vuln_sub = vuln_scores[:, cols]
                mask_sub = mask_buffer[:, cols]
                capacity_indices = compute_indices(vuln_sub, mask_sub, expert_capacity)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)

            return topk_weight, topk_idx

        def _redistribute_weights(topk_weight, topk_idx):
            """Redistribute dropped routing weight to surviving assignments.
            For each token, the surviving experts' weights are scaled up so
            their sum equals the original total, compensating for the lost
            contribution of dropped experts."""
            survived = (topk_idx != self.num_experts)               # [T, top_k]
            survived_sum = (topk_weight * survived).sum(dim=-1, keepdim=True)
            original_sum = topk_weight.sum(dim=-1, keepdim=True)
            scale = torch.where(
                survived_sum > 0,
                original_sum / survived_sum,
                torch.ones_like(survived_sum),
            )
            return topk_weight * scale * survived, topk_idx

        def _apply_soft_reroute(scores, expert_capacity, topk_idx, mask_buffer):
            """Re-route dropped slots to next-best available experts.

            Outer loop: assignment rounds (one slot per token per round).
            Inner loop: over experts (K) with vectorised ops per expert.
            A token with multiple dropped slots is handled across rounds so
            no expert is assigned twice to the same token."""
            T, K = scores.shape

            top_mask = mask_buffer.gather(-1, topk_idx)
            dropped_mask = ~top_mask

            if not dropped_mask.any():
                topk_weight = scores.gather(-1, topk_idx)
                return topk_weight, topk_idx

            remaining_cap = (expert_capacity - mask_buffer.sum(dim=0)).clamp(min=0).long()

            candidate_scores = scores.clone()
            candidate_scores.scatter_(-1, topk_idx, float('-inf'))

            dropped_t, dropped_s = dropped_mask.nonzero(as_tuple=True)
            n_dropped = dropped_t.shape[0]
            cand = candidate_scores[dropped_t]          # [n_dropped, K]
            assigned = torch.full((n_dropped,), self.num_experts,
                                  dtype=torch.long, device=scores.device)
            still_open = torch.ones(n_dropped, dtype=torch.bool,
                                    device=scores.device)

            max_slots_per_token = dropped_mask.sum(dim=-1).max().item()
            for _ in range(max_slots_per_token):
                if not still_open.any():
                    break

                round_cand = cand.clone()
                round_cand[:, remaining_cap <= 0] = float('-inf')
                round_cand[~still_open] = float('-inf')

                best = round_cand.argmax(dim=-1)
                best_val = round_cand.gather(-1, best.unsqueeze(-1)).squeeze(-1)
                seekers = still_open & (best_val > float('-inf'))
                if not seekers.any():
                    break

                # One slot per token per round to avoid duplicate assignments.
                seeker_idx = seekers.nonzero(as_tuple=True)[0]
                seeker_tok = dropped_t[seeker_idx]
                _, sorted_order = seeker_tok.sort(stable=True)
                sorted_tok = seeker_tok[sorted_order]
                first_occ = torch.cat([
                    torch.tensor([True], device=scores.device),
                    sorted_tok[1:] != sorted_tok[:-1],
                ])
                dedup = torch.zeros_like(seekers)
                dedup[seeker_idx[sorted_order[first_occ]]] = True
                seekers = seekers & dedup

                assigned_this_round = torch.zeros_like(still_open)
                for e in range(K):
                    if remaining_cap[e] <= 0:
                        continue
                    wanting = seekers & (best == e)
                    n_want = wanting.sum()
                    if n_want == 0:
                        continue
                    cap = remaining_cap[e]
                    if n_want <= cap:
                        assigned[wanting] = e
                        still_open[wanting] = False
                        assigned_this_round[wanting] = True
                        remaining_cap[e] -= n_want
                    else:
                        idxs = wanting.nonzero(as_tuple=True)[0]
                        gate = scores[dropped_t[idxs], e]
                        _, keep = gate.topk(cap)
                        winners = idxs[keep]
                        assigned[winners] = e
                        still_open[winners] = False
                        assigned_this_round[winners] = True
                        remaining_cap[e] = 0
                    cand[wanting, e] = float('-inf')

                # Block newly-assigned experts for other slots of the same token.
                new_idx = assigned_this_round.nonzero(as_tuple=True)[0]
                if new_idx.numel() > 0:
                    new_tok = dropped_t[new_idx]
                    new_exp = assigned[new_idx]
                    for tok, exp in zip(new_tok.tolist(), new_exp.tolist()):
                        same = (dropped_t == tok) & still_open
                        cand[same, exp] = float('-inf')

            topk_idx[dropped_t, dropped_s] = assigned

            safe_idx = topk_idx.clamp(max=K - 1)
            topk_weight = scores.gather(-1, safe_idx)
            topk_weight[topk_idx == self.num_experts] = 0.0
            return topk_weight, topk_idx

        def _drop_with_priority(scores, expert_capacity, top_k, priority_scores):
            """Shared drop logic: select top-k, enforce capacity using the
            given priority_scores (higher = more likely to keep), return
            (topk_idx, mask_buffer) for optional soft-reroute post-processing,
            or finalise as hard drop."""
            T, K = scores.shape

            topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            mask_buffer = torch.zeros((T, K), dtype=torch.bool, device=scores.device)
            mask_buffer.scatter_(-1, topk_idx, True)

            current_usage = mask_buffer.sum(dim=0)
            cols = (current_usage > expert_capacity).nonzero(as_tuple=True)[0]
            if cols.numel() > 0:
                prio_sub = priority_scores[:, cols]
                mask_sub = mask_buffer[:, cols]
                capacity_indices = compute_indices(prio_sub, mask_sub, expert_capacity)
                mask_buffer[:, cols] = torch.zeros_like(mask_sub).scatter(0, capacity_indices, True)

            return topk_idx, mask_buffer

        def _finalise_hard_drop(scores, topk_idx, mask_buffer):
            """Convert mask_buffer into hard-dropped (topk_weight, topk_idx)."""
            topk_weight = scores.gather(-1, topk_idx)
            top_mask = mask_buffer.gather(-1, topk_idx)
            topk_idx = topk_idx.masked_fill(~top_mask, self.num_experts)
            return topk_weight, topk_idx

        def _compute_margin_priority(scores, top_k):
            """Margin priority: gap between gate score and best fallback."""
            topk_plus1_vals, _ = torch.topk(scores, k=top_k + 1, dim=-1, sorted=True)
            next_best = topk_plus1_vals[:, -1]
            return scores - next_best.unsqueeze(-1)

        strategy = self.strategy or ""
        strategy_list = ["score", "last", "first", "random", "overselect",
                         "margin", "vulnerability", "score_redist", "soft_drop",
                         "identity_comp", "mean_expert_comp", "top1_clone_comp",
                         "low_rank_comp", "score_k", "rank", "expert_centric",
                         "score_sq", "score_rank_boost", "score_margin_add",
                         "score_sqrt", "score_margin_mul",                          "score_rank_boost2",
                         "score_top1_heavy", "expert_centric_margin",
                         "score_margin_blend", "score_rank_boost_soft",
                         "score_pow15", "score_rank_gentle", "expert_centric_hybrid",
                         "contrib_oracle", "contrib_lr"]
        if expert_capacity is None or not any(s in strategy for s in strategy_list):
            return torch.topk(scores, self.top_k, dim=-1, sorted=False)

        use_soft = "_soft" in strategy

        if "margin" in strategy:
            priority = _compute_margin_priority(scores, self.top_k)
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            if use_soft:
                return _apply_soft_reroute(scores, expert_capacity, topk_idx, mask_buffer)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "vulnerability" in strategy:
            return reroute_vulnerability(scores, expert_capacity, self.top_k)
        elif "soft_drop" in strategy:
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, scores)
            return _apply_soft_reroute(scores, expert_capacity, topk_idx, mask_buffer)
        elif "score_redist" in strategy:
            topk_weight, topk_idx = _reroute_common(scores, expert_capacity, self.top_k)
            return _redistribute_weights(topk_weight, topk_idx)
        elif any(s in strategy for s in ("identity_comp", "mean_expert_comp", "top1_clone_comp", "low_rank_comp")):
            return _reroute_common(scores, expert_capacity, self.top_k)
        elif "expert_centric_margin" in strategy:
            margin = _compute_margin_priority(scores, self.top_k)
            T, K = scores.shape
            cap = min(int(expert_capacity), T)
            cap = max(cap, 1)
            top_tokens_per_expert = torch.topk(
                margin.t(), k=cap, dim=1, largest=True
            ).indices
            selected_by = torch.zeros(
                T, self.num_experts, dtype=torch.bool, device=scores.device)
            exp_idx = torch.arange(
                self.num_experts, device=scores.device
            ).unsqueeze(1).expand(-1, cap)
            selected_by[top_tokens_per_expert, exp_idx] = True
            scores_masked = scores.clone()
            scores_masked[~selected_by] = -float('inf')
            topk_weight, topk_idx = torch.topk(
                scores_masked, k=self.top_k, dim=1, largest=True)
            valid = topk_weight > -float('inf')
            topk_idx = topk_idx.masked_fill(~valid, self.num_experts)
            topk_weight = topk_weight.masked_fill(~valid, 0.0)
            return topk_weight, topk_idx
        elif "expert_centric" in strategy:
            T, K = scores.shape
            cap = min(int(expert_capacity), T)
            cap = max(cap, 1)
            top_tokens_per_expert = torch.topk(
                scores.t(), k=cap, dim=1, largest=True
            ).indices
            selected_by = torch.zeros(
                T, self.num_experts, dtype=torch.bool, device=scores.device
            )
            exp_idx = torch.arange(
                self.num_experts, device=scores.device
            ).unsqueeze(1).expand(-1, cap)
            selected_by[top_tokens_per_expert, exp_idx] = True
            scores_masked = scores.clone()
            scores_masked[~selected_by] = -float('inf')
            topk_weight, topk_idx = torch.topk(
                scores_masked, k=self.top_k, dim=1, largest=True
            )
            valid = topk_weight > -float('inf')
            topk_idx = topk_idx.masked_fill(~valid, self.num_experts)
            topk_weight = topk_weight.masked_fill(~valid, 0.0)
            return topk_weight, topk_idx
        elif "rank" in strategy:
            T, K = scores.shape
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
            _, score_order = topk_weight.sort(dim=-1, descending=True)
            rank_per_slot = torch.zeros_like(topk_weight)
            rank_per_slot.scatter_(
                1, score_order,
                torch.arange(1, self.top_k + 1, device=scores.device,
                             dtype=scores.dtype).unsqueeze(0).expand(T, -1))
            rank_matrix = torch.full(
                (T, K), self.top_k + 1, dtype=scores.dtype, device=scores.device)
            rank_matrix.scatter_(1, topk_idx, rank_per_slot)
            rank_priority = (self.top_k + 1 - rank_matrix) * 10.0 + scores
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, rank_priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_sq" in strategy:
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, scores ** 2)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_pow15" in strategy:
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, scores ** 1.5)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_rank_gentle" in strategy:
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
            rank_mat = torch.full(
                (scores.size(0), scores.size(1)), 0,
                dtype=scores.dtype, device=scores.device)
            for s in range(self.top_k):
                rank_mat.scatter_(1, topk_idx[:, s:s+1], s + 1)
            boost = 1.0 + 0.3 * (self.top_k - rank_mat).clamp(min=0) / self.top_k
            priority = scores * boost
            priority[rank_mat == 0] = -float('inf')
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "expert_centric_hybrid" in strategy:
            margin = _compute_margin_priority(scores, self.top_k)
            pick_scores = 0.7 * scores + 0.3 * margin.clamp(min=0)
            T, K = scores.shape
            cap = min(int(expert_capacity), T)
            cap = max(cap, 1)
            top_tokens_per_expert = torch.topk(
                pick_scores.t(), k=cap, dim=1, largest=True
            ).indices
            selected_by = torch.zeros(
                T, self.num_experts, dtype=torch.bool, device=scores.device)
            exp_idx = torch.arange(
                self.num_experts, device=scores.device
            ).unsqueeze(1).expand(-1, cap)
            selected_by[top_tokens_per_expert, exp_idx] = True
            scores_masked = scores.clone()
            scores_masked[~selected_by] = -float('inf')
            topk_weight, topk_idx = torch.topk(
                scores_masked, k=self.top_k, dim=1, largest=True)
            valid = topk_weight > -float('inf')
            topk_idx = topk_idx.masked_fill(~valid, self.num_experts)
            topk_weight = topk_weight.masked_fill(~valid, 0.0)
            return topk_weight, topk_idx
        elif "score_sqrt" in strategy:
            eps = 1e-8
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k,
                torch.sqrt(scores.clamp(min=eps)))
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_margin_mul" in strategy:
            margin = _compute_margin_priority(scores, self.top_k)
            priority = scores * (1.0 + margin.clamp(min=0))
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_rank_boost2" in strategy:
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
            rank_mat = torch.full(
                (scores.size(0), scores.size(1)), 0,
                dtype=scores.dtype, device=scores.device)
            for s in range(self.top_k):
                rank_mat.scatter_(1, topk_idx[:, s:s+1], s + 1)
            boost = 1.0 + 2.0 * (self.top_k - rank_mat).clamp(min=0) / self.top_k
            priority = scores * boost
            priority[rank_mat == 0] = -float('inf')
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_rank_boost_soft" in strategy:
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
            rank_mat = torch.full(
                (scores.size(0), scores.size(1)), 0,
                dtype=scores.dtype, device=scores.device)
            for s in range(self.top_k):
                rank_mat.scatter_(1, topk_idx[:, s:s+1], s + 1)
            boost = 1.0 + (self.top_k - rank_mat).clamp(min=0) / self.top_k
            priority = scores * boost
            priority[rank_mat == 0] = -float('inf')
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _apply_soft_reroute(scores, expert_capacity, topk_idx, mask_buffer)
        elif "score_rank_boost" in strategy:
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
            rank_mat = torch.full(
                (scores.size(0), scores.size(1)), 0,
                dtype=scores.dtype, device=scores.device)
            for s in range(self.top_k):
                rank_mat.scatter_(1, topk_idx[:, s:s+1], s + 1)
            boost = 1.0 + (self.top_k - rank_mat).clamp(min=0) / self.top_k
            priority = scores * boost
            priority[rank_mat == 0] = -float('inf')
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_top1_heavy" in strategy:
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
            rank_mat = torch.full(
                (scores.size(0), scores.size(1)), 0,
                dtype=scores.dtype, device=scores.device)
            for s in range(self.top_k):
                rank_mat.scatter_(1, topk_idx[:, s:s+1], s + 1)
            boost = torch.where(rank_mat == 1, 3.0, 1.0)
            priority = scores * boost
            priority[rank_mat == 0] = -float('inf')
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_margin_add" in strategy:
            margin = _compute_margin_priority(scores, self.top_k)
            priority = scores + margin
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "score_margin_blend" in strategy:
            margin = _compute_margin_priority(scores, self.top_k)
            priority = scores + 0.5 * margin.clamp(min=0)
            topk_idx, mask_buffer = _drop_with_priority(
                scores, expert_capacity, self.top_k, priority)
            return _finalise_hard_drop(scores, topk_idx, mask_buffer)
        elif "contrib_oracle" in strategy:
            return self._reroute_contrib_oracle(scores, expert_capacity)
        elif "contrib_lr" in strategy:
            return self._reroute_contrib_lr(scores, expert_capacity)
        elif "score_k" in strategy:
            return _reroute_common(scores, expert_capacity, self.k_override)
        elif "score" in strategy:
            return _reroute_common(scores, expert_capacity, self.top_k)
        elif "random" in strategy:
            return reroute_random(scores, expert_capacity, self.top_k)
        elif "first" in strategy:
            return reroute_sequential_order(scores, expert_capacity, self.top_k, mode="first")
        elif "last" in strategy:
            return reroute_sequential_order(scores, expert_capacity, self.top_k, mode="last")
        else:
            return _reroute_common(scores, expert_capacity, self.top_k)


class OlmoeDecoderLayer(nn.Module):
    def __init__(self, config: OlmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OLMOE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = OlmoeSparseMoeBlock(config, layer_idx)
        
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, router_logits = self.mlp(hidden_states, attention_mask)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
        
        return outputs


OLMOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OlmoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Olmoe Model outputting raw hidden-states without any specific head on top.",
    OLMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->Olmoe
class OlmoePreTrainedModel(PreTrainedModel):
    config_class = OlmoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OlmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OLMOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Olmoe Model outputting raw hidden-states without any specific head on top.",
    OLMOE_START_DOCSTRING,
)
# TODO: re-enable check: Copied from transformers.models.llama.modeling_llama.LlamaModel with Llama->Olmoe
class OlmoeModel(OlmoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OlmoeDecoderLayer`]

    Args:
        config: OlmoeConfig
    """

    def __init__(self, config: OlmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OlmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(OLMOE_INPUTS_DOCSTRING)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ✨✨✨        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            # ✨✨✨ 
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class OlmoeForCausalLM(OlmoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OlmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 🔍🔍🔍        
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(OLMOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoeForCausalLM

        >>> model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output
        
        # ✨✨✨            
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
