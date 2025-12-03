# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for Glm4MoeConfig."""
import pytest

from mindformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from mindformers.parallel_core.mf_model_config import MFModelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


class TestGlm4MoeConfig:
    """Tests covering the Glm4Moe configuration helper."""

    @pytest.mark.level0
    def test_default_configuration_values(self):
        """Ensure defaults from the spec are propagated to attributes."""
        config = Glm4MoeConfig()

        assert config.vocab_size == 151552
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 46
        assert config.num_attention_heads == 96
        assert config.moe_intermediate_size == 1408
        assert config.num_experts_per_tok == 8
        assert config.norm_topk_prob is True
        assert config.model_type == "glm4_moe"
        assert "layers.*.self_attn.q_proj" in Glm4MoeConfig.base_model_tp_plan

    @pytest.mark.level0
    def test_rope_scaling_type_key_is_renamed(self):
        """When rope scaling contains 'type', it should be copied to 'rope_type'."""
        rope_scaling = {"type": "linear", "factor": 2.0}
        config = Glm4MoeConfig(rope_scaling=rope_scaling)

        assert config.rope_scaling["rope_type"] == "linear"
        assert config.rope_scaling["factor"] == 2.0

    @pytest.mark.level0
    def test_model_parameter_registration_metadata(self):
        """Decorator should expose MF model metadata on the __init__ function."""
        mf_kwargs = Glm4MoeConfig.__init__._mf_model_kwargs  # pylint: disable=protected-access
        assert isinstance(mf_kwargs, MFModelConfig)
        assert mf_kwargs.pad_token_id == 151643
        assert mf_kwargs.moe_router_score_function == "sigmoid"
        assert mf_kwargs.gated_linear_unit is True

    @pytest.mark.level0
    def test_config_registered_in_global_registry(self):
        """MindFormerRegister should contain the config with the provided search name."""
        registry_key = (MindFormerModuleType.CONFIG.value, "Glm4MoeConfig")
        assert registry_key in MindFormerRegister._registry  # pylint: disable=protected-access
        entry = MindFormerRegister._registry[registry_key]  # pylint: disable=protected-access
        assert entry["search_names"] == 'glm4_moe'
        assert entry["object"] is Glm4MoeConfig

