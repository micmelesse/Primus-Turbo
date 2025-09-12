###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

import torch


def quantize_fp8_tensorwise(x: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Tensor-Wise
    """
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(x, dtype)
    return x_fp8, scale_inv


def quantize_fp8_rowwise(
    x: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP8 Row-Wise
    """
    if not x.is_contiguous():
        x = x.contiguous()
    x_fp8, scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_rowwise(x, dtype, axis)
    return x_fp8, scale_inv
