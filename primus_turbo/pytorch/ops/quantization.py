###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus_turbo.pytorch.core.float8 import ScalingGranularity
from primus_turbo.pytorch.kernels.quantization_impl import (
    quantize_fp8_rowwise,
    quantize_fp8_tensorwise,
)

__all__ = ["quantize_fp8"]


def quantize_fp8(
    x: torch.Tensor,
    dtype: torch.dtype,
    granularity: ScalingGranularity,
    axis: Optional[int] = None,
    # block_size: Optional[Tuple[int, ...]] = None,
):
    """
    FP8 Quantize
    """
    if granularity == ScalingGranularity.TENSORWISE:
        return quantize_fp8_tensorwise(x, dtype)
    elif granularity == ScalingGranularity.ROWWISE:
        if axis is None:
            raise ValueError("axis must be specified for rowwise FP8 quantization")
        return quantize_fp8_rowwise(x, dtype, axis)
    elif granularity == ScalingGranularity.BLOCKWISE:
        raise NotImplementedError("Blockwise FP8 quantization is not supported yet")
    else:
        raise NotImplementedError(f"Unknown granularity {granularity}")


"""
FP4 Quantize
"""
