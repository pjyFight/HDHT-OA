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
"""Unit tests for LowerTriangularMaskWithDynamic."""
import numpy as np
import pytest

from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.transformer.lower_triangular_mask import (
    LowerTriangularMaskWithDynamic,
)


class TestLowerTriangularMask:
    """Validates lower-triangular mask generation."""

    @pytest.mark.level0
    def test_prefill_returns_fixed_attention_mask(self):
        """Prefill path should directly return the static fa mask."""
        module = LowerTriangularMaskWithDynamic(seq_length=4, compute_type=mstype.float16)
        module.is_prefill = True

        mask = module.construct(positions=Tensor(np.zeros((1, 4)), dtype=mstype.int32))

        np.testing.assert_array_equal(mask.asnumpy(), module.fa_lower_triangle_mask.asnumpy())

    @pytest.mark.level0
    def test_decode_gathers_rows_from_preallocated_mask(self):
        """Decode path should gather using provided positions."""
        module = LowerTriangularMaskWithDynamic(seq_length=4, compute_type=mstype.float16)
        module.is_prefill = False
        positions = Tensor(np.array([0, 2], dtype=np.int32))

        mask = module.construct(positions=positions)

        expected = module.pa_lower_triangle_mask.asnumpy()[positions.asnumpy()]
        np.testing.assert_array_equal(mask.asnumpy(), expected)

    @pytest.mark.level0
    def test_bfloat16_prefill_mask_has_positive_coeff(self):
        """When using bf16 compute type, mask coefficient becomes +1."""
        module = LowerTriangularMaskWithDynamic(seq_length=4, compute_type=mstype.bfloat16)
        mask = module.prefill().asnumpy()
        coeff_region = mask[np.triu_indices(128, 1)]

        np.testing.assert_allclose(coeff_region, np.ones_like(coeff_region), rtol=1e-4, atol=1e-4)

