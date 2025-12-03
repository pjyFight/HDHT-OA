"""MF model config placeholder."""
from dataclasses import dataclass


@dataclass
class MFModelConfig:
    """Minimal representation of MF model kwargs."""
    pad_token_id: int = 0
    block_size: int = 1
    num_blocks: int = 1
    moe_router_score_function: str = "softmax"
    moe_router_enable_expert_bias: bool = False
    normalization: str = "LayerNorm"
    add_bias_linear: bool = False
    gated_linear_unit: bool = False

