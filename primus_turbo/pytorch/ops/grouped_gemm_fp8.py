###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from typing import Union

import torch

from primus_turbo.pytorch.core.float8 import (
    Format,
    ScalingGranularity,
    TensorwiseQuantConfig,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_blockwise_impl,
    grouped_gemm_fp8_csrc_impl,
    grouped_gemm_fp8_variable_k_csrc_impl,
    grouped_gemm_variable_k_fp8_blockwise_impl,
)
from primus_turbo.pytorch.kernels.quantize import (
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8

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
        config: TensorwiseQuantConfig,
        num_cu: int | None,
    ):
        assert config.granularity == ScalingGranularity.ROWWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        out_dtype = a.dtype
        assert out_dtype in [torch.float16, torch.bfloat16]

        a_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(config.format, True)
        a_fp8_row, a_scale_inv_row = quantize_fp8(a, a_dtype, config.granularity, axis=-1)
        b_fp8_row, b_scale_inv_row = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-1 if trans_b else -2)
        )
        out = grouped_gemm_fp8_csrc_impl(
            a_fp8_row,
            b_fp8_row,
            a_scale_inv_row,
            b_scale_inv_row,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=out_dtype,
            granularity=config.granularity,
            num_cu=num_cu,
        )

        # we need a/b do col quant for backward.
        a_fp8_col, a_scale_inv_col = quantize_fp8(a, a_dtype, config.granularity, axis=-2)
        b_fp8_col, b_scale_inv_col = quantize_fp8(
            b, b_dtype, config.granularity, axis=(-2 if trans_b else -1)
        )

        ctx.save_for_backward(a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        ctx.config = config
        ctx.out_dtype = out_dtype
        ctx.num_cu = num_cu
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a_fp8_col, b_fp8_col, a_scale_inv_col, b_scale_inv_col, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = GroupedGemmFP8RowFunc.get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8_row, grad_out_scale_inv_row = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-1
        )

        grad_a = grouped_gemm_fp8_csrc_impl(
            grad_out_fp8_row,
            b_fp8_col,
            grad_out_scale_inv_row,
            b_scale_inv_col,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        # For grad_b
        grad_out_fp8_col, grad_out_scale_inv_col = quantize_fp8(
            grad_out, grad_out_dtype, ctx.config.granularity, axis=-2
        )

        lhs, rhs = (grad_out_fp8_col, a_fp8_col) if ctx.trans_b else (a_fp8_col, grad_out_fp8_col)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv_col, a_scale_inv_col)
            if ctx.trans_b
            else (a_scale_inv_col, grad_out_scale_inv_col)
        )

        grad_b = grouped_gemm_fp8_variable_k_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


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
        config: TensorwiseQuantConfig,
        num_cu: int | None,
    ):

        assert config.granularity == ScalingGranularity.TENSORWISE
        assert a.ndim == 2, "Input tensor must be 3-dimensions."
        assert b.ndim == 3, "Weight tensor must be 3-dimensional."
        a_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)
        b_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(config.format, True)
        a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, config.granularity)
        b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, config.granularity)

        out = grouped_gemm_fp8_csrc_impl(
            a_fp8,
            b_fp8,
            a_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
            out_dtype=a.dtype,
            granularity=config.granularity,
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
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs = ctx.saved_tensors

        # For grad_a
        grad_out_dtype = GroupedGemmFP8TensorFunc.get_fp8_dtype(ctx.config.format, False)
        grad_out_fp8, grad_out_scale_inv = quantize_fp8(grad_out, grad_out_dtype, ctx.config.granularity)
        grad_a = grouped_gemm_fp8_csrc_impl(
            grad_out_fp8,
            b_fp8,
            grad_out_scale_inv,
            b_scale_inv,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        # For grad_b
        lhs, rhs = (grad_out_fp8, a_fp8) if ctx.trans_b else (a_fp8, grad_out_fp8)
        lhs_scale, rhs_scale = (
            (grad_out_scale_inv, a_scale_inv) if ctx.trans_b else (a_scale_inv, grad_out_scale_inv)
        )
        grad_b = grouped_gemm_fp8_variable_k_csrc_impl(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
            out_dtype=ctx.out_dtype,
            granularity=ctx.config.granularity,
            num_cu=ctx.num_cu,
        )

        return grad_a, grad_b, None, None, None, None, None


def grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = True,
    config: Union[TensorwiseQuantConfig, None] = None,
    num_cu: int | None = None,
) -> torch.Tensor:
    """ """
    supported_dtypes = [torch.bfloat16, torch.float16]
    assert a.dtype in supported_dtypes, f"Unsupported dtype {a.dtype}, expected one of {supported_dtypes}"
    assert b.dtype in supported_dtypes, f"Unsupported dtype {b.dtype}, expected one of {supported_dtypes}"

    if group_offs is None:
        group_offs = grouped_gemm_compute_offs(group_lens)
    if config is None:
        config = TensorwiseQuantConfig()

    args = (a, b, group_lens, group_offs, trans_b, config, num_cu)

    if config.granularity == ScalingGranularity.TENSORWISE:
        return GroupedGemmFP8TensorFunc.apply(*args)
    elif config.granularity == ScalingGranularity.ROWWISE:
        return GroupedGemmFP8RowFunc.apply(*args)
    else:
        raise ValueError(f"Unsupported FP8 ScalingGranularity: {config.granularity}")


"""
TODO: MXFP8, MXFP4
"""
