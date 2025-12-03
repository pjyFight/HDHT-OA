# coding=utf-8
# Copyright 2025 The Z.ai team and the HuggingFace Inc. team. All rights reserved.
#
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
"""Glm4 Config API."""

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.model_config_utils import (
    register_mf_model_parameter,
    ignore_and_delete_parameter
)
from mindformers.parallel_core.mf_model_config import MFModelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CONFIG, legacy=False, search_names='glm4')
class Glm4Config(PretrainedConfig):
    r"""
    Glm4 configuration definition. Mirrors defaults from THUDM/GLM-4-9B-0414.
    """

    model_type = "glm4"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_rep",
        "layers.*.mlp.down_proj": "rowwise_rep",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    @register_mf_model_parameter(
        mf_model_kwargs=MFModelConfig(
            block_size=32,
            num_blocks=1024,
            normalization='RMSNorm',
            add_bias_linear=False,
            gated_linear_unit=True,
            sandwich_norm=True,
            rotary_cos_format='interleaved'))
    @ignore_and_delete_parameter()
    def __init__(self,
                 vocab_size=151552,
                 hidden_size=4096,
                 intermediate_size=13696,
                 num_hidden_layers=40,
                 num_attention_heads=32,
                 num_key_value_heads=2,
                 partial_rotary_factor=0.5,
                 head_dim=128,
                 hidden_act="silu",
                 attention_dropout=0.0,
                 max_position_embeddings=131072,
                 initializer_range=0.02,
                 rms_norm_eps=0.00000015625,
                 use_cache=True,
                 tie_word_embeddings=False,
                 rope_theta=10000.0,
                 pad_token_id=151329,
                 eos_token_id=(151329, 151336, 151338),
                 bos_token_id=None,
                 attention_bias=True,
                 position_embedding_type="partial_rope",
                 **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.position_embedding_type = position_embedding_type

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

