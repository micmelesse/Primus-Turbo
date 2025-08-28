###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Union

import torch

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_csrc_fp8_row_impl,
    grouped_gemm_csrc_fp8_tensor_impl,
    grouped_gemm_fp8_blockwise_impl,
    grouped_gemm_variable_k_fp8_blockwise_impl,
    grouped_gemm_variable_k_fp8_row_csrc_impl,
    grouped_gemm_variable_k_fp8_tensor_csrc_impl,
)
from primus_turbo.pytorch.kernels.quantize import (
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quant_fp8_rowwise_impl,
    quant_fp8_tensorwise_impl,
)

__all__ = [
    "grouped_gemm_fp8",
    "grouped_gemm_fp8_blockwise",
]


class BlockwiseFP8GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        seg_lens: torch.Tensor,
        block_size: int = 128,
        dtype=float8_e4m3,
    ):
        batch_size = seg_lens.size(0)
        assert batch_size == weight.size(0)
        seg_indptr = torch.cat([torch.tensor([0], device=seg_lens.device), seg_lens.cumsum(0)])

        # Quantize input activation (row): shape [M, K] → FP8
        x_fp8_row, x_scales_row = quant_fp8_blockwise_impl(x, dtype, axis=1, block_size=block_size)
        # Quantize weight blockwise: shape [B, N, K] → FP8
        w_fp8, w_scales = quant_fp8_blockwise_for_weight_impl(weight, dtype, block_size)

        # TODO: Opt
        out = grouped_gemm_fp8_blockwise_impl(
            x_fp8_row,
            w_fp8,
            x_scales_row,
            w_scales,
            batch_size,
            seg_indptr,
            out_dtype=x.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            trans_a=False,
            trans_b=True,
        )

        ctx.save_for_backward(x, w_fp8, w_scales, seg_lens, seg_indptr)
        ctx.batch_size = batch_size
        ctx.block_size = block_size
        ctx.dtype = dtype

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_fp8, w_scales, seg_lens, seg_indptr = ctx.saved_tensors
        batch_size = ctx.batch_size
        block_size = ctx.block_size
        dtype = ctx.dtype

        # quant grad_out
        grad_out_fp8_row, grad_out_scales_row = quant_fp8_blockwise_impl(
            grad_out, dtype, axis=1, block_size=block_size
        )

        # TODO: Opt
        # DGrad NN:
        grad_x = grouped_gemm_fp8_blockwise_impl(
            grad_out_fp8_row,
            w_fp8,
            grad_out_scales_row,
            w_scales,
            batch_size,
            seg_indptr,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            trans_a=False,
            trans_b=False,
        )

        # TODO: Opt
        # WGrad TN
        # grad_w = grad_out.T @ x
        # [b, n, k] = [m, n] * [m, k]
        act_scales_col_seg_lens = torch.ceil(seg_lens.float() / block_size).to(seg_lens.dtype)
        act_scales_col_seg_indptr = torch.cat(
            [
                torch.tensor([0], device=seg_lens.device),
                act_scales_col_seg_lens.cumsum(0),
            ]
        )

        x_fp8_col, x_scales_col = quant_fp8_blockwise_segment_m_impl(
            x,
            batch_size,
            seg_lens,
            seg_indptr,
            act_scales_col_seg_indptr,
            dtype,
            block_size,
        )
        grad_out_fp8_col, grad_out_scales_col = quant_fp8_blockwise_segment_m_impl(
            grad_out,
            batch_size,
            seg_lens,
            seg_indptr,
            act_scales_col_seg_indptr,
            dtype,
            block_size,
        )

        grad_w = grouped_gemm_variable_k_fp8_blockwise_impl(
            grad_out_fp8_col,
            x_fp8_col,
            grad_out_scales_col,
            x_scales_col,
            batch_size,
            seg_indptr,
            act_scales_col_seg_indptr,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=1,
            scale_group_size_k=block_size,
            trans_a=True,
            trans_b=False,
        )

        return (
            grad_x,
            grad_w,
            None,
            None,
            None,
            None,
            None,
        )


def grouped_gemm_fp8_blockwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    seg_lens: torch.Tensor,
    block_size: int = 128,
    dtype=float8_e4m3,
):
    """
    Grouped GEMM with FP8 blockwise quantization.

    Args:
        x (torch.Tensor): Input tensor of shape [M, K], float16/bfloat16.
        weight (torch.Tensor): Weight tensor of shape [B, N, K], float16/bfloat16.
        seg_lens (torch.Tensor): Segment lengths of shape [B], int64. Sum should equal M.
        block_size (int): Block size for quantization. Default: 128.
        dtype (torch.dtype): FP8 type. Default: turbo.float8_e4m3.

    Returns:
        torch.Tensor: Output tensor of shape [M, N], same dtype as x.

    Example:
        >>> x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
        >>> weight = torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16)
        >>> seg_lens = torch.tensor([64, 64, 64, 64], dtype=torch.long, device="cuda")
        >>> out = grouped_gemm_fp8_blockwise(x, weight, seg_lens)
        >>> print(out.shape)  # torch.Size([256, 64])
    """
    return BlockwiseFP8GroupedGemmFunc.apply(x, weight, seg_lens, block_size, dtype)


@torch.compile
def calc_scale_and_scale_inv_rowwise(x: torch.Tensor, fp8_max: float, row_wise: bool = True):
    if row_wise:
        if x.dim() == 2:
            amax = x.abs().amax(dim=1, keepdim=True)
        elif x.dim() == 3:
            amax = x.abs().amax(dim=2, keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    else:
        if x.dim() == 2:
            amax = x.abs().amax(dim=0, keepdim=True)
        elif x.dim() == 3:
            amax = x.abs().amax(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    scale = torch.full_like(amax, fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
    scale_inv = 1.0 / scale

    return scale, scale_inv


class GroupedGemmFP8RowFunc(torch.autograd.Function):
    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.ROWWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)

        a_scale, a_scale_inv = calc_scale_and_scale_inv_rowwise(a, torch.finfo(a_dtype).max, row_wise=True)
        b_scale, b_scale_inv = calc_scale_and_scale_inv_rowwise(b, torch.finfo(b_dtype).max, row_wise=trans_b)

        a_fp8_row = quant_fp8_rowwise_impl(a, a_scale, a_dtype, row_quant=True)
        b_fp8_row = quant_fp8_rowwise_impl(b, b_scale, b_dtype, row_quant=trans_b)

        out = grouped_gemm_csrc_fp8_row_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            num_cu=num_cu,
        )

        # we need a/b do col quant for backward.
        a_scale, a_scale_inv = calc_scale_and_scale_inv_rowwise(a, torch.finfo(a_dtype).max, row_wise=False)
        b_scale, b_scale_inv = calc_scale_and_scale_inv_rowwise(
            b, torch.finfo(b_dtype).max, row_wise=not trans_b
        )

        a_fp8_col = quant_fp8_rowwise_impl(a, a_scale, a_dtype, row_quant=False)
        b_fp8_col = quant_fp8_rowwise_impl(b, b_scale, b_dtype, row_quant=not trans_b)

        ctx.save_for_backward(a_fp8_col, b_fp8_col, a_scale_inv, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_fp8_col, b_fp8_col, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors
        grad_out_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(ctx.config.format, False)

        grad_out_scale, grad_out_scale_inv = calc_scale_and_scale_inv_rowwise(
            grad_out, torch.finfo(grad_out_dtype).max, row_wise=True
        )

        grad_out_fp8 = quant_fp8_rowwise_impl(grad_out, grad_out_scale, grad_out_dtype, row_quant=True)

        grad_a = grouped_gemm_csrc_fp8_row_impl(
            grad_out_fp8,
            b_fp8_col,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            num_cu=ctx.num_cu,
        )

        grad_out_scale, grad_out_scale_inv = calc_scale_and_scale_inv_rowwise(
            grad_out, torch.finfo(grad_out_dtype).max, row_wise=False
        )
        grad_out_fp8 = quant_fp8_rowwise_impl(grad_out, grad_out_scale, grad_out_dtype, row_quant=False)

        lhs, rhs = (grad_out_fp8, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv, a_scale_inv) if ctx.trans_b else (a_scale_inv, grad_out_scale_inv)
        )

        grad_b = grouped_gemm_variable_k_fp8_row_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


@torch.compile
def calc_scale_and_scale_inv_tensorwise(x: torch.Tensor, fp8_max: float):
    amax = x.abs().amax()
    scale = torch.full([1], fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
    scale_inv = 1.0 / scale

    return scale, scale_inv


class GroupedGemmFP8TensorFunc(torch.autograd.Function):

    @staticmethod
    def get_fp8_dtype(format: Format, is_fwd_stage: bool):
        if format == Format.E4M3:
            return float8_e4m3
        elif format == Format.E5M2:
            return float8_e5m2
        else:
            raise ValueError(f"Unsupported FP8 format: {format}")

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool,
        config: Float8QuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.TENSORWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)

        a_scale, a_scale_inv = calc_scale_and_scale_inv_tensorwise(a, torch.finfo(a_dtype).max)
        b_scale, b_scale_inv = calc_scale_and_scale_inv_tensorwise(b, torch.finfo(b_dtype).max)

        a_fp8 = quant_fp8_tensorwise_impl(a, a_scale, a_dtype)
        b_fp8 = quant_fp8_tensorwise_impl(b, b_scale, b_dtype)

        out = grouped_gemm_csrc_fp8_tensor_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            num_cu=num_cu,
        )

        ctx.save_for_backward(a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = a.dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_fp8_col, b_fp8_col, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        grad_out_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(ctx.config.format, False)

        grad_out_scale, grad_out_scale_inv = calc_scale_and_scale_inv_tensorwise(
            grad_out, torch.finfo(grad_out_dtype).max
        )

        grad_out_fp8 = quant_fp8_tensorwise_impl(grad_out, grad_out_scale, grad_out_dtype)

        grad_a = grouped_gemm_csrc_fp8_tensor_impl(
            grad_out_fp8,
            b_fp8_col,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            num_cu=ctx.num_cu,
        )

        lhs, rhs = (grad_out_fp8, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv, a_scale_inv) if ctx.trans_b else (a_scale_inv, grad_out_scale_inv)
        )

        grad_b = grouped_gemm_variable_k_fp8_tensor_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[Float8QuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """ """
    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)
    if config is None:
        config = Float8QuantConfig()

    args = (a, b, group_lens, group_offs, trans_b, config, num_cu)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return GroupedGemmFP8TensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return GroupedGemmFP8RowFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")
