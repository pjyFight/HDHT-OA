# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Unit tests for transformer utils helpers."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.transformer import utils as transformer_utils


class DummySubCell:
    """Subcell exposing a sharded state dict."""

    def __init__(self):
        self.param = Parameter(
            Tensor(np.ones((2, 2), dtype=np.float32), dtype=mstype.float32), name="sub.param"
        )

    def sharded_state_dict(self):
        return {
            "sub.param": {
                "shape": self.param.shape,
                "shard": (1, 2),
            }
        }

    def name_cells(self):
        return {"self": self}


class DummyNetwork:
    """Minimal network exposing parameters and cells."""

    def __init__(self):
        self.sub = DummySubCell()
        self.head = Parameter(Tensor(np.ones((2,), dtype=np.float32), dtype=mstype.float32), name="head.bias")

    def name_cells(self):
        return {"self": self, "sub": self.sub}

    def parameters_dict(self):
        return {"sub.param": self.sub.param, "head.bias": self.head}


class TestAttnMaskHelpers:
    """Tests for attention mask helpers."""

    @pytest.mark.level0
    def test_attn_mask_fill_applies_value(self):
        func = transformer_utils.get_attn_mask_func("attn_mask_fill")
        scores = Tensor(np.ones((1, 2), dtype=np.float32), dtype=mstype.float32)
        mask = Tensor(np.array([[False, True]]), dtype=mstype.bool_)
        output = func(scores, mask, fill_value=-9.0)

        output_np = output.asnumpy()
        assert output_np[0, 0] == pytest.approx(1.0, rel=1e-6)
        assert output_np[0, 1] == pytest.approx(-9.0, rel=1e-6)

    @pytest.mark.level0
    def test_attn_mask_add_casts_mask(self):
        func = transformer_utils.get_attn_mask_func("attn_mask_add")
        scores = Tensor(np.zeros((1, 2), dtype=np.float32), dtype=mstype.float32)
        mask = Tensor(np.array([[0.0, -5.0]], dtype=np.float32), dtype=mstype.float32)
        output = func(scores, mask)

        np.testing.assert_allclose(output.asnumpy(), np.array([[0.0, -5.0]], dtype=np.float32))

    @pytest.mark.level0
    def test_get_attn_mask_func_with_invalid_name(self):
        with pytest.raises(KeyError):
            transformer_utils.get_attn_mask_func("unknown")


class TestStateDictGeneration:
    """Tests for sharded state dict utilities."""

    @pytest.mark.level0
    def test_generate_state_dict_includes_sharded_and_full_params(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_group_size", lambda: 2)
        state_dict = transformer_utils.generate_state_dict(DummyNetwork())

        assert state_dict["total_rank"] == 2
        assert "sub.param" in state_dict["model"]
        assert "head.bias" in state_dict["model"]
        assert state_dict["model"]["head.bias"]["shard"] == (1,)


class TestCommAndTopologyHelpers:
    """Tests targeting communication helper utilities."""

    @pytest.mark.level0
    def test_update_comm_config_single_tp_multi_dp(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(transformer_utils, "get_data_parallel_world_size", lambda: 2)
        monkeypatch.setattr(transformer_utils, "get_moe_tensor_parallel_world_size", lambda: 1)

        config = TransformerConfig()
        updated = transformer_utils.update_comm_config(config)

        assert updated.use_alltoall is True
        assert updated.attn_allreduce is False
        assert updated.ffn_allreduce is False

    @pytest.mark.level0
    def test_update_comm_config_moe_tp_enabled(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(transformer_utils, "get_data_parallel_world_size", lambda: 2)
        monkeypatch.setattr(transformer_utils, "get_moe_tensor_parallel_world_size", lambda: 2)

        config = TransformerConfig()
        updated = transformer_utils.update_comm_config(config)

        assert updated.attn_allgather is True
        assert updated.ffn_reduce_scatter is True
        assert updated.ffn_allreduce is False

    @pytest.mark.level0
    def test_get_num_layers_and_offset_with_pp_offsets(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_pipeline_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(transformer_utils, "get_pipeline_model_parallel_rank", lambda: 1)

        config = TransformerConfig(num_layers=5, offset=[1, 0])

        layers, offset = transformer_utils.get_num_layers_and_offset(config)

        assert layers == 2
        assert offset == 3

    @pytest.mark.level0
    def test_get_num_layers_and_offset_raises_for_small_model(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_pipeline_model_parallel_world_size", lambda: 8)

        config = TransformerConfig(num_layers=4)

        with pytest.raises(RuntimeError):
            transformer_utils.get_num_layers_and_offset(config)

    @pytest.mark.level0
    def test_get_num_layers_and_offset_invalid_offset_shape(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_pipeline_model_parallel_world_size", lambda: 2)

        config = TransformerConfig(num_layers=6, offset=[1, 0, 0])

        with pytest.raises(ValueError):
            transformer_utils.get_num_layers_and_offset(config)


class TestMathHelpers:
    """Tests for small math helpers."""

    @pytest.mark.level0
    def test_divide_checks_divisibility(self):
        assert transformer_utils.divide(6, 3) == 2
        with pytest.raises(ValueError):
            transformer_utils.divide(5, 3)


class TestCustomOpsToggle:
    """Tests for custom ops toggling."""

    @pytest.mark.level0
    def test_use_ms_custom_ops_false_when_module_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ms_custom_ops", None, raising=False)
        assert transformer_utils.use_ms_custom_ops() is False

    @pytest.mark.level0
    def test_use_ms_custom_ops_true_when_module_present(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ms_custom_ops", SimpleNamespace(), raising=False)
        monkeypatch.setattr(transformer_utils, "is_310p", lambda: False)
        assert transformer_utils.use_ms_custom_ops() is True

    @pytest.mark.level0
    def test_use_ms_custom_ops_false_when_310p(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ms_custom_ops", SimpleNamespace(), raising=False)
        monkeypatch.setattr(transformer_utils, "is_310p", lambda: True)
        assert transformer_utils.use_ms_custom_ops() is False


class TestParameterUtility:
    """Covers helpers related to parameter creation."""

    @pytest.mark.level0
    def test_create_empty_parameter_returns_expected_shape(self):
        param = transformer_utils.create_empty_parameter((2, 3), dtype=mstype.float32, name="dummy")
        assert param.shape == (2, 3)
        assert param.dtype == mstype.float32


class TestWorldSizeFallbacks:
    """Ensure fallback logic returns non-zero defaults."""

    @pytest.mark.level0
    def test_world_size_helpers_default_to_one(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_tensor_model_parallel_world_size", lambda: 0)
        monkeypatch.setattr(transformer_utils, "get_moe_tensor_parallel_world_size", lambda: 0)
        monkeypatch.setattr(transformer_utils, "get_moe_expert_parallel_world_size", lambda: 0)
        monkeypatch.setattr(transformer_utils, "get_data_parallel_world_size", lambda: 0)

        assert transformer_utils.get_tp_world_size() == 1
        assert transformer_utils.get_moe_tp_world_size() == 1
        assert transformer_utils.get_moe_ep_world_size() == 1
        assert transformer_utils.get_dp_world_size() == 1


class TestPaddingIndexGeneration:
    """Tests for generate_padding_index helper."""

    @pytest.mark.level0
    def test_generate_padding_index_single_dp(self, monkeypatch):
        monkeypatch.setattr(transformer_utils, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(transformer_utils, "get_data_parallel_world_size", lambda: 1)
        monkeypatch.setattr(transformer_utils, "get_data_parallel_group",
                            lambda: SimpleNamespace(rank=0, group=None))

        q_seq_lens = Tensor(np.array([[2]], dtype=np.int32))
        attn_pad, attn_unpad, ffn_pad, ffn_unpad = transformer_utils.generate_padding_index(q_seq_lens)

        assert attn_pad.shape == (2,)
        assert attn_unpad.shape == (2,)
        assert ffn_pad.shape == (2,)
        assert ffn_unpad.shape == (2,)

