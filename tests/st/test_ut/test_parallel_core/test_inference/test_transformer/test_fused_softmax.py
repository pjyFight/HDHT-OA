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
"""UTs for FusedScaleMaskSoftmax."""
import numpy as np
import pytest

from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.transformer.fused_softmax import FusedScaleMaskSoftmax


def simple_mask(tensor, mask):
    """Mask function for tests that multiplies by mask."""
    return tensor + mask


class TestFusedScaleMaskSoftmax:
    """Tests covering the fused softmax helper."""

    @pytest.mark.level0
    def test_forward_pass_with_scale_and_mask(self):
        layer = FusedScaleMaskSoftmax(mask_func=simple_mask, scale=0.5, softmax_compute_type=mstype.float32)
        x = Tensor(np.array([[2.0, 0.0]], dtype=np.float32), dtype=mstype.float32)
        mask = Tensor(np.array([[0.0, -1.0]], dtype=np.float32), dtype=mstype.float32)

        output = layer.construct(x, mask)

        assert output.shape == (1, 2)
        np.testing.assert_allclose(output.asnumpy().sum(axis=-1), np.ones(1), atol=1e-6)

    @pytest.mark.level0
    def test_precision_casts_to_fp32_when_needed(self):
        layer = FusedScaleMaskSoftmax(mask_func=simple_mask, scale=None, softmax_compute_type=mstype.float32)
        x = Tensor(np.array([[1.0, 1.0]], dtype=np.float16), dtype=mstype.float16)

        output = layer.construct(x, mask=None)

        assert output.dtype == mstype.float16
        np.testing.assert_allclose(output.asnumpy(), np.array([[0.5, 0.5]], dtype=np.float16), atol=1e-3)

    @pytest.mark.level0
    def test_invalid_scale_precision_combination_raises(self):
        with pytest.raises(ValueError):
            FusedScaleMaskSoftmax(mask_func=simple_mask, scale=0.1, softmax_compute_type=mstype.float16)
