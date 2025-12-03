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
"""UTs for Glm4Moe modeling API."""

import pytest

from mindformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from mindformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM
from mindformers.models.glm4_moe.modeling_glm4_moe_infer import InferenceGlm4MoeForCausalLM
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


class TestGlm4MoeForCausalLM:
    """Ensure Glm4MoeForCausalLM routes to the proper implementation."""

    @pytest.mark.level0
    def test_predict_mode_returns_inference_impl(self, monkeypatch):
        """When RUN_MODE is unset/predict, the inference model should be instantiated."""
        monkeypatch.delenv("RUN_MODE", raising=False)
        config = Glm4MoeConfig()

        model = Glm4MoeForCausalLM(config)

        assert isinstance(model, InferenceGlm4MoeForCausalLM)
        assert model.config is config

    @pytest.mark.level0
    def test_train_mode_raises_not_implemented(self, monkeypatch):
        """RUN_MODE=train should raise an explicit NotImplementedError."""
        monkeypatch.setenv("RUN_MODE", "train")

        with pytest.raises(NotImplementedError):
            Glm4MoeForCausalLM(Glm4MoeConfig())

    @pytest.mark.level0
    def test_model_registered_with_registry(self):
        """MindFormerRegister should have the model recorded under MODELS."""
        registry_key = (MindFormerModuleType.MODELS.value, "Glm4MoeForCausalLM")
        assert registry_key in MindFormerRegister._registry  # pylint: disable=protected-access
        entry = MindFormerRegister._registry[registry_key]  # pylint: disable=protected-access
        assert entry["object"] is Glm4MoeForCausalLM

