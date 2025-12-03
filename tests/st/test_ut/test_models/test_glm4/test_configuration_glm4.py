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
"""Unit tests for Glm4Config."""
import pytest

from mindformers.models.glm4.configuration_glm4 import Glm4Config
from mindformers.parallel_core.mf_model_config import MFModelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


class TestGlm4Config:
    """Validates default behaviors for the Glm4 configuration."""

    @pytest.mark.level0
    def test_default_configuration_fields(self):
        config = Glm4Config()

        assert config.vocab_size == 151552
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 2
        assert config.position_embedding_type == "partial_rope"
        assert config.model_type == "glm4"
        assert "layers.*.self_attn.q_proj" in Glm4Config.base_model_tp_plan

    @pytest.mark.level0
    def test_override_arguments_apply(self):
        config = Glm4Config(vocab_size=10, num_attention_heads=8, eos_token_id=(1,))

        assert config.vocab_size == 10
        assert config.num_attention_heads == 8
        assert config.eos_token_id == (1,)

    @pytest.mark.level0
    def test_registered_model_parameter_metadata(self):
        mf_kwargs = Glm4Config.__init__._mf_model_kwargs  # pylint: disable=protected-access
        assert isinstance(mf_kwargs, MFModelConfig)
        assert mf_kwargs.block_size == 32
        assert mf_kwargs.sandwich_norm is True

    @pytest.mark.level0
    def test_registry_contains_glm4_config(self):
        registry_key = (MindFormerModuleType.CONFIG.value, "Glm4Config")
        assert registry_key in MindFormerRegister._registry  # pylint: disable=protected-access
        assert MindFormerRegister._registry[registry_key]["search_names"] == 'glm4'  # pylint: disable=protected-access

