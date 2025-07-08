import math
import copy
import warnings
from typing import List, Optional, Tuple, Union, Callable
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import logging
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from fla.modules import RMSNorm
from fla.modules.feature_map import (
    DPFPFeatureMap, 
    HadamardFeatureMap,
    HedgehogFeatureMap, 
    T2RFeatureMap,
)
from fla.ops.linear_attn.utils import normalize_output
from lolcats.models.lolcats.configuration_lolcats import LolcatsConfig

logger = logging.get_logger(__name__)

# class LolcatsHedgehogFeatureMap(nn.Module):
#     class FeatureMapMLP(nn.Module):
#         def __init__(
#             self, 
#             num_heads: int = 32,
#             head_dim: int = 128,     # input dim
#             feature_dim: int = 128,  # output dim
#             dtype: torch.dtype = torch.bfloat16,
#             device: torch.device = 'cuda:0',
#             skip_connection: bool = False,
#             bias: bool = False,
#             zero_init: bool = False, 
#             normal_init: bool = True,
#         ):
#             super().__init__()
#             self.num_heads = num_heads
#             self.head_dim = head_dim
#             self.feature_dim = feature_dim
#             self.dtype = dtype
#             self.device = device
#             self.skip_connection = skip_connection
#             self.bias = bias
#             self.zero_init = zero_init
#             self.normal_init = normal_init
#             self.init_weights_()

#             if self.zero_init:  # Zero-out weights or set as identity post-initialization
#                 self.zero_init_with_skip_() if self.skip_connection else self.zero_init_()
            
#             if self.normal_init:
#                 with torch.no_grad():
#                     nn.init.normal_(self.layer, std=.02)
        
#             if self.skip_connection:
#                 assertion_fail = f'If self.skip_connection we need self.head_dim == self.feature_dim but self.head_dim is {self.head_dim} != self.feature_dim is {self.feature_dim}'
#                 assert self.head_dim == self.feature_dim, assertion_fail

#         def init_weights_(self):
#             """
#             Initialize (W)eights and (b)iases
#             """
#             self.layer = nn.Parameter(torch.zeros(
#                 (self.num_heads, self.head_dim, self.feature_dim),
#                 dtype=self.dtype, # device=self.device,
#             ))
#             nn.init.kaiming_uniform_(self.layer)

#             if self.bias:
#                 self.bias = nn.Parameter(torch.zeros(
#                     (1, self.num_heads, 1, 1),  # self.feature_dim),
#                     dtype=self.dtype, # device=self.device,
#                 ))
#                 nn.init.kaiming_uniform_(self.bias)
#             else:
#                 self.bias = 0.  # hack

#         def zero_init_with_skip_(self):
#             """
#             Initialize weights to zero matrix if skip connection
#             """
#             with torch.no_grad():
#                 nn.init.zeros_(self.layer)
        
#         def zero_init_(self):
#             """
#             Initialize weights to identity matrix if no skip connection
#             """
#             pass
#             # with torch.no_grad():
#             #     for i in range(self.layer.shape[0]):
#             #         try:
#             #             nn.init.eye_(self.layer[i])
#             #         except RuntimeError:
#             #             with torch.no_grad():
#             #                 dtype = self.layer[i].dtype
#             #                 weight = torch.eye(*self.layer[i].shape,
#             #                                 requires_grad=self.layer[i].requires_grad,
#             #                                 device=self.layer[i].device)
#             #                 self.layer[i] = weight.to(dtype=dtype)
        
#         def forward(self, x: torch.Tensor):
#             """
#             Assume x.shape is (batch_size, num_heads, seq_len, head_dim)
#             """
#             _x = torch.einsum('hdf,bhld->bhlf', self.layer, x) + self.bias
#             return x + _x if self.skip_connection else _x
    
#     class ReLU(nn.Module):
#         """
#         ReLU activation as in https://arxiv.org/abs/2103.13076
#         """
#         def __init__(self, eps=1e-12):
#             super().__init__()
#             self.eps = eps

#         def forward(self, x: torch.Tensor, *args: any, **kwargs: any):
#             return F.relu(x).clamp(min=self.eps)
    
#     class SoftmaxDim(nn.Module):
#         """
#         Softmax activation as in https://arxiv.org/abs/2402.04347
#         """
#         def __init__(self, eps=1e-12):
#             super().__init__()
#             self.eps = eps

#         def forward(self, x: torch.Tensor, *args: any, **kwargs: any):
#             return torch.cat([
#                 torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
#             ], dim=-1).clamp(min=self.eps)
        
#     def __init__(self, head_dim: int, num_heads: int):
#         super().__init__()
#         self.head_dim = head_dim
#         self.eps = 1e-12
#         self.mlp = self.FeatureMapMLP(head_dim=head_dim, num_heads=num_heads)
#         self.activation = self.SoftmaxDim(eps=self.eps)
    
#     def forward(self, x):
#         return self.activation(self.mlp(x), x)
    
#     def q_map(self, *args: any, **kwargs: any):
#         """
#         Use for inference in case q and k feature maps differ
#         """
#         return self.forward(*args, **kwargs)

#     def k_map(self, *args: any, **kwargs: any):
#         """
#         Use for inference in case q and k feature maps differ
#         """
#         return self.forward(*args, **kwargs)

class LolcatsHedgehogFeatureMap(nn.Module):
    class FeatureMapMLP(nn.Module):
        def __init__(
            self,
            num_heads: int = 32,
            head_dim: int = 128,
            feature_dim: int = 128,
            dtype: torch.dtype = torch.bfloat16,
            device: torch.device = 'cuda:0',
            skip_connection: bool = False,
            bias: bool = False,
            zero_init: bool = False,
            normal_init: bool = False,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.feature_dim = feature_dim
            self.dtype = dtype
            self.device = device
            self.skip_connection = skip_connection
            self.bias_flag = bias  # Rename to avoid conflict
            self.zero_init = zero_init
            self.normal_init = normal_init

            # Create one Linear layer per head
            self.linears = nn.ModuleList([
                nn.Linear(head_dim, feature_dim, bias=False)
                for _ in range(num_heads)
            ])
            # Move linears to target device/dtype
            # for linear in self.linears:
            #     linear.to(device=device, dtype=dtype)

            # Initialize weights
            self.init_weights_()

            # Post-init adjustments
            if zero_init:
                if skip_connection:
                    self.zero_init_with_skip_()
                else:
                    self.zero_init_()
            if normal_init:
                self.normal_init_()

            # Skip connection check
            if skip_connection:
                assert head_dim == feature_dim, (
                    f"head_dim ({head_dim}) != feature_dim ({feature_dim})"
                )

            # Bias term (shared per head)
            if self.bias_flag:
                self.bias = nn.Parameter(torch.zeros(
                    (1, num_heads, 1, 1),  # Broadcastable shape
                    dtype=dtype, device=device
                ))
                nn.init.kaiming_uniform_(self.bias)
            else:
                self.bias = 0.0

        def init_weights_(self):
            """Initialize Linear weights with kaiming_uniform"""
            for linear in self.linears:
                nn.init.kaiming_uniform_(linear.weight)

        def zero_init_with_skip_(self):
            """Zero all Linear weights"""
            for linear in self.linears:
                nn.init.zeros_(linear.weight)

        def zero_init_(self):
            """Identity initialization when head_dim == feature_dim"""
            if self.head_dim != self.feature_dim:
                raise ValueError("Identity init requires head_dim == feature_dim")
            for linear in self.linears:
                nn.init.eye_(linear.weight)

        def normal_init_(self):
            """Normal initialization for Linear weights"""
            for linear in self.linears:
                nn.init.normal_(linear.weight, std=0.02)

        def forward(self, x: torch.Tensor):
            # Stack weights: (num_heads, out_dim, in_dim)
            weights = torch.stack([l.weight for l in self.linears], dim=0)
            # Transpose to match original einsum format: (h, d, f)
            weights_t = weights.transpose(1, 2)
            # Original computation
            _x = torch.einsum('hdf,bhld->bhlf', weights_t, x) + self.bias
            return x + _x if self.skip_connection else _x

    class ReLU(nn.Module):
        def __init__(self, eps=1e-12):
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor):
            return F.relu(x).clamp(min=self.eps)

    class SoftmaxDim(nn.Module):
        def __init__(self, eps=1e-12):
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor):
            return torch.cat([
                torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
            ], dim=-1).clamp(min=self.eps)

    def __init__(self, head_dim: int, num_heads: int):
        super().__init__()
        self.head_dim = head_dim
        self.eps = 1e-12
        self.mlp = self.FeatureMapMLP(head_dim=head_dim, num_heads=num_heads)
        self.activation = self.SoftmaxDim(eps=self.eps)

    def forward(self, x):
        return self.activation(self.mlp(x))

def causal_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Causal linear attention dot product
    - If available, use CUDA kernel from fast-transformers
    """
    # if fast_causal_dot_product is None:
    #     kv = torch.einsum('bhlf,bhld->bhlfd', k, v)
    #     return torch.einsum('bhlf,bhlfd->bhld', q, kv.cumsum(dim=2))
    # return fast_causal_dot_product(q, k, v)
    kv = torch.einsum('bhlf,bhld->bhlfd', k, v)
    return torch.einsum('bhlf,bhlfd->bhld', q, kv.cumsum(dim=2))

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     fp32_attention: bool = False, eps: float = 1e-12,
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute linear attention with CUDA kernel implementation from fast-transformers
    - https://github.com/idiap/fast-transformers
    - Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); 
      v is shape (b, h, l, head_dim)
    """
    dtype = q.dtype
    # Causal mask already applied
    y = causal_dot_product(q.contiguous().to(dtype=torch.float32),
                           k.contiguous().to(dtype=torch.float32),
                           v.contiguous().to(dtype=torch.float32))
    if fp32_attention:
        y = (y / (torch.einsum(
            "bhld,bhld->bhl", q.float(), k.float().cumsum(dim=2)
        ) + eps)[..., None]).to(dtype=dtype)
    else:
        y = y.to(dtype=dtype)
        k = k.float().cumsum(dim=2).to(dtype=dtype)
        # k = k.cumsum(dim=2) 
        y = y / (torch.einsum("bhld,bhld->bhl", q, k) + eps)[..., None]
    return y, None, None


def softmax_attention(q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor] = None, 
                      causal: bool = True, fp32_attention: bool = True,
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Standard softmax attention; only compute outputs if v is not None
    -> Assume q, k, v are shape (batch_size, num_heads, seq_len, head_dim)
    """
    y = None
    a = torch.einsum('bhmd,bhnd->bhmn', q, k) * (k.shape[-1] ** -0.5)
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, -torch.finfo(a.dtype).max)
    if fp32_attention:
        a = torch.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
    else:
        a = torch.softmax(a, dim=-1)
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a, None


def quadratic_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor = None,
                        causal: bool = True, fp32_attention: bool = False, eps: float = 1e-12,
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute attention with feature maps by instantiating L x L matrix of attention weights
    -> Use for attention distillation
    -> Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); v is shape (b, h, l, head_dim)
    """
    y = None
    dtype = q.dtype
    if fp32_attention:
        q, k = q.float(), k.float()
    a = torch.einsum('bhmd,bhnd->bhmn', q, k)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    # Normalize to compute attention
    a = a / (a.sum(dim=-1, keepdim=True) + eps)
    a = a.to(dtype=dtype) if fp32_attention else a
    if torch.isnan(a).sum() > 0:
        breakpoint()
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a, None

# ----------------------
# Sliding window helpers
# ----------------------
def get_masks(window_size: int, q_len: int, k_len: int, 
              device: torch.device) -> tuple[torch.Tensor]:
    """
    Return masks for softmax and linear attention terms
    -> 1 is include, 0 is ignore
    """
    kwargs = {'device': device, 'dtype': int}
    causal_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len)
    linear_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len - window_size)
    window_mask = causal_mask - linear_mask
    # Return softmax mask (window), linear attention mask
    # -> shapes broadcast over (b, h, q_len, k_len)
    return window_mask[None, None, ...], linear_mask[None, None, ...]


def hybrid_attention_quadratic(q: torch.Tensor, k: torch.Tensor, 
                               f_q: torch.Tensor, f_k: torch.Tensor,
                               v: torch.Tensor,
                               window_factor: torch.Tensor,
                               linear_factor: torch.Tensor,
                               window_size: int,
                               kv_state: torch.Tensor = None,
                               k_state: torch.Tensor = None,
                               eps: float = 1e-12,
                               mask_value: float=-1e8):
    """
    Hybrid attention combining sliding window and linear attentions
    """

    mask_window, mask_linear = get_masks(window_size, q.shape[-2], k.shape[-2], q.device)

    # 1. Sliding window (softmax attention)
    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (k.shape[-1] ** -0.5)
    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    # torch.softmax(a_sm, dim=-1), but we account for the max when combining
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm   = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # 2. Under window (linear attention)
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
    a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # 3. Combine
    a = ((a_sm + a_ln) / (sum_sm + sum_ln)).to(q.dtype)  # Save attention weights
    # Allow outputs to also depend on prior kv_state and k_state
    y = torch.einsum('bhmn,bhnd->bhmd', a_sm + a_ln, v.float())
    if kv_state is not None:  # Combine with prior kv_state and k_state
        y += linear_factor * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float())
        sum_ln += linear_factor * torch.einsum(
            'bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]
    y = (y / (sum_sm + sum_ln + 1e-6)).to(q.dtype)
    return y, a  # attention weights only for the last chunk

class LinearAttention(nn.Module):
    def __init__(
        self, 
        config,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads # 8
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 32/8=4
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # linear attention settings
        self.attn_mode = config.attn_mode
        self.key_dim = int(self.hidden_size * config.expand_k)
        self.value_dim = int(self.hidden_size * config.expand_v)
        self.key_dim_per_group = self.key_dim // self.num_key_value_groups
        self.value_dim_per_group = self.value_dim // self.num_key_value_groups

        assert self.attn_mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.attn_mode}`."
        assert self.key_dim % self.num_heads == 0, f"key dim must be divisible by num_heads of {self.num_heads}"
        assert self.value_dim % self.num_heads == 0, f"value dim must be divisible by num_heads of {self.num_heads}"

        self.head_qk_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads
        self.do_feature_map_norm = config.norm_feature_map

        feature_map = config.feature_map
        tie_feature_map_qk = config.tie_feature_map_qk
        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim, bias=True)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim, bias=True)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim, bias=True)
        elif feature_map == 'lolcats_hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = LolcatsHedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = LolcatsHedgehogFeatureMap(head_dim=self.head_qk_dim, num_heads=self.num_heads)
                self.feature_map_k = copy.deepcopy(self.feature_map_q)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.norm_q = config.norm_q
        self.norm_k = config.norm_k

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # _init_weights
        self.q_shape = [self.num_heads, self.head_dim]
        self.k_shape = [self.num_key_value_heads, self.head_dim]
        self.v_shape = [self.num_key_value_heads, self.head_dim]

        self.decode_window_size = 64
        self.window_size = 64
        init_window_factor = -2.1972245773362196
        self.affine_attention_factors = False
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        self.register_buffer(
                "window_factors", init_window_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype)
            )

    def _process_qkv(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        b, l, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        kv_seq_len = k.shape[-2]

        # Shape is (batch_size, seq_len, num_heads, head_dim)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)

        if past_key_value is not None:  #  and k.shape[2] > q.shape[2]:  # e.g., when generating
            past_key_value.window_size = getattr(self, 'decode_window_size', None)  # self.decode_window_size
            if isinstance(past_key_value, Cache):  # In Transformers v4.36+ this is a DynamicCache object
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value[0].shape[-2]

        # Apply rotary embeddings and repeat for GQA
        if position_ids is not None and kv_seq_len <= position_ids[0, -1]:
            kv_seq_len = position_ids[0, -1] + 1  # hack for adjusting position ids
        try: # As in Transformers v4.36
            cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        except TypeError:  # As in Transformers v4.39+
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # "legacy" cache approach
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self._process_qkv(hidden_states, attention_mask, position_ids, past_key_value)

        attn = None

        if output_attentions:
            with torch.no_grad():
                _o_true, attn_true, _ = softmax_attention(q, k, v, causal=True)
                o_true = _o_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                o = self.o_proj(o_true)
                
            q = self.feature_map_q(q)
            k = self.feature_map_k(k)
            o_pred, attn_pred, _ = quadratic_attention(q, k, v, causal=True)
            attn = (o_pred, _o_true)
        else:
            q = self.feature_map_q(q)
            k = self.feature_map_k(k)
            # Apply prefill mask
            if attention_mask is not None and q.shape[2] > 1:
                if len(attention_mask.shape) == 4:
                    lin_attn_mask = (attention_mask == 0)[:, :1, -1, :l][..., None]  # b, 1, k_len, 1
                else:
                    lin_attn_mask = attention_mask[:, None, :, None]  # b, 1, k_len, 1
                k = k.masked_fill(~lin_attn_mask, 0)
            
            if past_key_value is not None:  # Initialize states
                if len(past_key_value.kv_states) == self.layer_idx:
                    b, h, _, f = k.shape
                    past_key_value.kv_states.append(
                        torch.zeros(b, h, f, self.head_dim, dtype=q.dtype, device=q.device)
                    )
                    past_key_value.k_states.append(
                        torch.zeros(b, h, 1, f, dtype=q.dtype, device=q.device)
                    )
                # Generating
                if q.shape[2] == 1 and kv_seq_len > 1 and past_key_value is not None:
                    assert use_cache is True
                    kv_state, k_state = past_key_value.update(k, v, self.layer_idx,
                                                              accumulate_in_fp32=self.fp32_attention)
                    if self.fp32_attention:
                        q = q.float()
                        o = (torch.einsum('bhlf,bhfd->bhld', q, kv_state.float()) /
                                  torch.einsum('bhlf,bhlf->bhl', q, k_state.float())[..., None]).to(dtype=k.dtype)
                    else:
                        o = (torch.einsum('bhlf,bhfd->bhld', q, kv_state) /
                                  torch.einsum('bhlf,bhlf->bhl', q, k_state)[..., None])
                else:
                    kv_state = past_key_value.kv_states[self.layer_idx]
                    k_state  = past_key_value.k_states[self.layer_idx]
                    o, _, _ = linear_attention(q, k, v)  # Ordinarily the states are ignored
                    past_key_value.update(k.detach(), v.detach(), self.layer_idx) # doing some unnecessary recomputation here
            else:
                o, _, _ = linear_attention(q, k, v)

            # Concatenate heads and apply output projection
            o = o.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            o = self.o_proj(o)
        
        return o, attn, _
    
class LolcatsAttention(LinearAttention):
      def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # "legacy" cache approach
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self._process_qkv(hidden_states, attention_mask, 
                                               position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)  # Have to do after repeat for grouped-query attn if we use same fmap

        if output_attentions:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                _y_true, a_true = softmax_attention(q, k, v)[:2]
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention outputs
            # compute attn weights under sliding window
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred, a_pred = hybrid_attention_quadratic(q, k, f_q, f_k, v,
                                                      window_factors, linear_factors,
                                                      window_size=self.window_size)
            # attn_weights = ((a_pred, a_true), (y_pred, _y_true))
            attn_weights = (_y_true, y_pred, )
        else:
            attn_weights = None
            # attention_mask = None  # For now this is always True
            if past_key_value is None:  # Regular training
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                y_true, a_pred = hybrid_attention_quadratic(q, k, f_q, f_k, v,
                                                          window_factors, linear_factors,
                                                          window_size=self.window_size)
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
                if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:  # Generating
                    assert use_cache is True
                    _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                                             self.feature_map_k,
                                                             dtype=q.dtype)
                    k_cache, v_cache, f_kv_state, f_k_state = _kv

                    # Sliding window + linear attention decode
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1

                    # Softmax attention terms
                    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm   = window_factors * torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)

                    # Combine with linear attention terms
                    y_true = (torch.einsum('bhmn,bhnd->bhmd', a_sm, v_cache.float())
                              + linear_factors * torch.einsum('bhlf,bhfd->bhld', f_q.float(), f_kv_state.float()))
                    sum_ln = linear_factors * torch.einsum(
                        'bhlf,bhnf->bhl', f_q.float(), f_k_state.float())[..., None]
                    y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype) 

                else:  # Stateful training
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state  = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                    y_true, _ = hybrid_attention_quadratic(q, k, f_q, f_k, v,
                                                         window_factors, linear_factors,
                                                         window_size=self.window_size,
                                                         kv_state=kv_state,
                                                         k_state=k_state)
                    # Save and update KV cache and states
                    # past_key_value.update(k, v.detach(), self.layer_idx,
                    #                       fmap_key_states=f_k.detach(),
                    #                       accumulate_in_fp32=True)
                    past_key_value.update(k, v, self.layer_idx,
                                          fmap_key_states=f_k,
                                          accumulate_in_fp32=True)
            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)
        return y_true, attn_weights, past_key_value

class LolcatsDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LolcatsConfig, layer_idx: int):
        super().__init__(config, layer_idx) # layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = LolcatsAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if output_attentions == True:
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            hidden_states, attns, present_key_value = self.self_attn(
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
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (attns, )

            if use_cache:
                outputs += (present_key_value,)

            return outputs
            
        else:
            return super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kwargs)
        

class LolcatsPreTrainedModel(LlamaPreTrainedModel):

    config_class = LolcatsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ['LolcatsAttentionDecoderLayer']
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(
        self, 
        module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # if rescale_prenorm_residual:
        #     # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #     #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #     #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #     #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #     #
        #     # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        #     for name, p in module.named_parameters():
        #         if name in ["o_proj.weight", "down_proj.weight"]:
        #             # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
        #             # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
        #             # We need to reinit p since this code could be called multiple times
        #             # Having just p *= scale would repeatedly scale it down
        #             with torch.no_grad():
        #                 p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class LolcatsModel(LlamaModel, LolcatsPreTrainedModel):

    def __init__(self, config: LolcatsConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LolcatsDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        # if use_cache and not isinstance(past_key_values, Cache):
        #     return_legacy_cache = True
        #     if past_key_values is None:
        #         past_key_values = DynamicCache()
        #     else:
        #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #         logger.warning_once(
        #             "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
        #             "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
        #             "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
        #         )
        if use_cache:
            if past_key_values is None or isinstance(past_key_values, DynamicCache): # Determine and setup our KV cache or state
                attention_type = getattr(self.layers[0].self_attn, 'attention_type', None)
                past_key_values = LinearAttentionSlidingWindowCache() # LinearAttentionState() # get_attention_cache(attention_type)
            else:
                past_key_values.get_usable_length(input_ids.shape[-2])  

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
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if output_attentions:
            all_softmax_hidden_states = () 

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                if all_softmax_hidden_states is not None:
                    all_softmax_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
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
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LolcatsModelForCausalLM(LlamaForCausalLM, LolcatsPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LolcatsModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
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
            return_dict=return_dict,
            cache_position=cache_position,
            # **kwargs,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class LinearAttentionState(Cache):
    """
    Handle the KV and K states for linear attention
    - Adopts HF Transformers `past_key_values` convention
    - Inherits from `Cache` class
    - Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self) -> None:
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states. A layer index can be optionally passed.
        """
        if len(self._seen_tokens_by_layer) <= layer_idx:  # Initializing kv and k states
            self._seen_tokens_by_layer.append(0)
        return self._seen_tokens_by_layer[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = True, **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad ():
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            dtype = key_states.dtype
            if accumulate_in_fp32:
                key_states, value_states = key_states.float(), value_states.float()

            kv_state = torch.einsum('bhlf,bhld->bhfd', key_states, value_states).detach()
            k_state  = key_states.sum(dim=-2, keepdim=True).detach()  # b, h, 1, f; note the 1
            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                print('if len(self.k_states) <= layer_idx:  # Initializing kv and k states')
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))
            else:
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state  = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx]  = k_state
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2] 
        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def to_legacy_cache(self):
        """Hack, but just return self"""
        return self

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        Reorders the cache for beam search, given the selected beam indices.
        -> Copied from transformers/src/transformers/cache_utils.py
        """
        raise NotImplementedError('Reordering cache not implemented for LinearAttentionState')

class LinearAttentionSlidingWindowCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []

        # Account for sliding windows
        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.window_size = window_size

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = False, 
               fmap_key_states: torch.Tensor = None,  # should not be None
               grad_enabled: bool = False,
               **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV, K states; and KV cache during training
        - For decoding, use `self.decode_kv_states` to keep track of KV states 
          up to sliding window terms
        - For (chunked) training, use `self.kv_states` to keep track of KV states
          up to end of sequence
        - Likewise for `self.decode_k_states` and `self.k_states`
        """
        with torch.set_grad_enabled(grad_enabled):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            dtype = key_states.dtype
            if accumulate_in_fp32:
                # key_states = key_states.float()
                fmap_key_states = fmap_key_states.float()
                value_states = value_states.float()

            # Decoding KV state (KV terms up to last window_size)
            decode_kv_state = torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states[:, :, :-self.window_size], value_states[:, :, :-self.window_size]
            )
            # KV state
            kv_state = decode_kv_state + torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states[:, :, -self.window_size:], value_states[:, :, -self.window_size:]
            )
            # shape is b, h, 1, f; note the 1
            decode_k_state = fmap_key_states[:, :, :-self.window_size].sum(dim=-2, keepdim=True)
            k_state = (decode_k_state + fmap_key_states[:, :, -self.window_size:].sum(dim=-2, keepdim=True))

            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))

                self.decode_kv_states.append(decode_kv_state.to(dtype))
                self.decode_k_states.append(decode_k_state.to(dtype))

                self.k_cache.append(key_states[:, :, -self.window_size:, :])
                self.v_cache.append(value_states[:, :, -self.window_size:, :].to(dtype))
                # self._seen_tokens_by_layer[layer_idx].append(key_states.shape[-2])
            else:
                # Update kv and k states recurrently
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state  = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx]  = k_state

                decode_kv_state = (self.decode_kv_states[layer_idx].to(kv_state.dtype) 
                                   + decode_kv_state).to(dtype)
                decode_k_state  = (self.decode_k_states[layer_idx].to(kv_state.dtype) 
                                   + decode_k_state).to(dtype)
                self.decode_kv_states[layer_idx] = decode_kv_state
                self.decode_k_states[layer_idx]  = decode_k_state

                self.k_cache[layer_idx] = key_states[:, :, -self.window_size:, :]
                self.v_cache[layer_idx] = value_states[:, :, -self.window_size:, :]
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def update_for_decoding(self, keys: torch.Tensor, values: torch.Tensor, 
                            layer_idx: int, feature_map_k: Callable, dtype: torch.dtype):
        """
        Update the decoding KV and K states, and KV cache, during decodeing
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]

            if k_cache.shape[-2] < self.window_size:  # build window-size cache
                self.k_cache[layer_idx] = torch.cat([k_cache, keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache, values], dim=-2)
            else:
                # MZ 6/3: handle short inputs; zero-out padding when initial k.shape[2] < self.window_size
                # if k_cache[:, :, :1, :].sum() == 0:   # heuristic for zeroing out padding in cache
                #     f_k_state = torch.zeros(k_cache[:, :, :1, :].shape, dtype=dtype, device=k_cache.device)
                # else:
                #     f_k_state = feature_map_k(k_cache[:, :, :1, :])
                # -> MZ (later): above only relevant if we zero-pad in our hybrid attention computation
                k_state = feature_map_k(k_cache[:, :, :1, :])
                v_state = v_cache[:, :, :1, :]
                kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype) # b, h, f, d
                self.decode_kv_states[layer_idx] += kv_state
                self.decode_k_states[layer_idx] += k_state
                
                self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)
            
            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]
            return (self.k_cache[layer_idx], self.v_cache[layer_idx], 
                    self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx])