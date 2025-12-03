"""Simplified TransformerConfig for unit tests."""
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    tensor_model_parallel_size: int = 1
    seq_length: int = 1
    hidden_size: int = 1
    ffn_hidden_size: int = 4
    num_attention_heads: int = 1
    add_bias_linear: bool = False
    gated_linear_unit: bool = False
    hidden_act: str = "relu"
    normalization: str = "LayerNorm"
    num_layers: int = 1
    use_flash_attention: bool = False
    compute_dtype: str = "fp32"
    params_dtype: str = "fp32"
    layernorm_epsilon: float = 1e-5
    apply_residual_connection_post_layernorm: bool = False
    attn_allreduce: bool = True
    attn_reduce_scatter: bool = False
    attn_allgather: bool = False
    attn_delay_allreduce: bool = False
    ffn_allreduce: bool = True
    ffn_reduce_scatter: bool = False
    ffn_allgather: bool = False
    use_alltoall: bool = False
    offset: int | tuple | list = 0
    num_blocks: int = 0
    block_size: int = 0

