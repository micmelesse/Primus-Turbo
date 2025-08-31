###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import torch


def quantize_fp8_tensorwise_ref(x, dtype):
    fp8_max = torch.finfo(dtype).max
    # Compute Scale & Scale-Inv
    x_max = x.abs().amax().to(torch.float32)
    scale = fp8_max / x_max
    scale_inv = 1.0 / scale
    # Quantize
    x_scaled = x * scale
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    return x_clamped.to(dtype), scale_inv.to(torch.float32)


def quantize_fp8_rowwise_ref(x, dtype, axis):
    axis = axis if axis >= 0 else x.dim() + axis
    if axis < 0 or axis >= x.dim():
        raise ValueError(f"axis={axis} is out of bounds for tensor of dimension {x.dim()}")
    fp8_max = torch.finfo(dtype).max
    # Compute Scale
    x_max = torch.amax(x.abs(), dim=axis, keepdim=True).to(torch.float32)
    x_max = torch.clamp(x_max, min=1e-8)

    scale = fp8_max / x_max
    scale_inv = 1.0 / scale
    # Quantize
    x_scaled = x * scale
    x_clamped = torch.clamp(x_scaled, -fp8_max, fp8_max)
    return x_clamped.to(dtype), scale_inv.to(torch.float32)
