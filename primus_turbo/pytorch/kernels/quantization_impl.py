###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch

"""
Quantize FP8 Tensor-Wise
"""


def quantize_fp8_tensorwise(x: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(dtype).max

    # Compute Scale
    # TODO: x_max too low, 1e-12
    x_max = x.abs().amax().to(torch.float32)
    scale = fp8_max / x_max
    scale_inv = 1.0 / scale

    # Quantize
    x_scaled = x * scale
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.to(dtype)

    return x_fp8, scale_inv


"""
Quantize FP8 Row-Wise
"""


def quantize_fp8_rowwise(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = axis if axis >= 0 else x.dim() + axis
    if axis < 0 or axis >= x.dim():
        raise ValueError(f"axis={axis} is out of bounds for tensor of dimension {x.dim()}")

    fp8_max = torch.finfo(dtype).max

    # Compute Scale
    x_max = torch.amax(x.abs(), dim=axis, keepdim=True).to(torch.float32)
    scale = fp8_max / x_max
    scale_inv = 1.0 / scale

    # Quantize
    x_scaled = x * scale
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    x_fp8 = x_clamped.to(dtype)
    return x_fp8, scale_inv
