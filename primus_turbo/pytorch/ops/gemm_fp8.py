from typing import Optional

import torch

from primus_turbo.pytorch.core.float8 import Format, MXQuantConfig, ScalingGranularity
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_blockwise_impl,
    quant_fp8_blockwise_for_act_grad_impl,
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.quantize import quant_fp8_blockwise_impl

__all__ = ["gemm_fp8_blockwise"]


# TODO: opt and refact
class BlockwiseFP8GemmFunction(torch.autograd.Function):
    """
    Autograd function for FP8 blockwise GEMM.

    This class implements the forward and backward pass for a GEMM using blockwise FP8 quantization.
    It is not intended to be used directly — use `gemm_fp8_blockwise` instead.

    Forward:
        - Quantize input `x` row-wise into FP8 blockwise.
        - Quantize weight `weight` blockwise.
        - Perform matrix multiplication: Y = x @ weight.T (NT layout).
        - Output result in high precision (same dtype as `x`).

    Backward Pass:
        - Quantize `grad_out` both row-wise and column-wise for dgrad/wgrad respectively.
        - Re-quantize input `x` column-wise for use in wgrad.
        - Compute:
            * grad_x = grad_out @ weight      (NN layout)
            * grad_w = grad_out.T @ x         (TN layout)
        - Both gradients are returned in high precision.

    Inputs:
        x (Tensor): Input tensor of shape [M, K], typically bf16/fp16.
        weight (Tensor): Weight tensor of shape [N, K], typically bf16/fp16.
        config (MXQuantConfig): Quantization configuration, specifying dtype, block size,
            scaling strategy, and granularity.

    Returns:
        out (Tensor): Output tensor of shape [M, N], same dtype as `x`.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        config: MXQuantConfig,
    ):
        # Check
        assert config != None
        assert config.granularity == ScalingGranularity.BLOCKWISE
        assert config.block_size != None
        assert config.dtype != Format.HYBRID
        block_size = config.block_size
        dtype = config.dtype.value.fwd_dtype

        # TODO: move to low level and clean
        orig_shape = x.shape  # [B, M, K] or [M, K]
        need_reshape = x.ndim == 3
        if need_reshape:
            x = x.view(-1, x.shape[-1])  # → [B*M, K]

        # Quantize input activation (row): shape [M, K] → FP8
        x_fp8_row, x_scales_row = quant_fp8_blockwise_impl(x, dtype, axis=1, block_size=block_size)
        # Quantize weight blockwise: shape [N, K] → FP8
        w_fp8, w_scales = quant_fp8_blockwise_for_weight_impl(weight, dtype, block_size)

        # Perform NT GEMM in quantized domain:
        # out = x_fp8_row @ w_fp8.T → shape [M, N]
        # out = gemm_fp8_blockwise_nt_impl(
        #     x_fp8_row,
        #     w_fp8,
        #     x_scales_row,
        #     w_scales,
        #     out_dtype=x.dtype,
        #     block_size=block_size,
        # )
        out = gemm_fp8_blockwise_impl(
            x_fp8_row,
            w_fp8,
            x_scales_row,
            w_scales,
            out_dtype=x.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            transA=False,
            transB=True,
        )

        # Save tensors for backward
        ctx.save_for_backward(x, w_fp8, w_scales)
        ctx.config = config
        ctx.orig_shape = orig_shape if need_reshape else None

        if need_reshape:
            out = out.view(orig_shape[0], orig_shape[1], -1)  # [B, M, N]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        x, w_fp8, w_scales = ctx.saved_tensors
        config = ctx.config
        block_size = config.block_size
        dtype = config.dtype.value.bwd_dtype
        orig_shape = ctx.orig_shape

        need_reshape = grad_out.ndim == 3
        if need_reshape:
            grad_out = grad_out.view(-1, grad_out.shape[-1])  # → [B*M, N]

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        grad_out_fp8_row, grad_out_scales_row, grad_out_fp8_col, grad_out_scales_col = (
            quant_fp8_blockwise_for_act_grad_impl(grad_out, dtype, block_size)
        )

        # TODO: dequant + quant kernel
        x_fp8_col, x_scales_col = quant_fp8_blockwise_impl(x, dtype, axis=0, block_size=block_size)

        # DGrad NN:
        # grad_x = grad_out @ weight
        # [m, k] = [m, n] * [n, k]
        # grad_x = gemm_fp8_blockwise_nn_impl(
        #     grad_out_fp8_row,
        #     w_fp8,
        #     grad_out_scales_row,
        #     w_scales,
        #     out_dtype=grad_out.dtype,
        #     block_size=block_size,
        # )
        grad_x = gemm_fp8_blockwise_impl(
            grad_out_fp8_row,
            w_fp8,
            grad_out_scales_row,
            w_scales,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=block_size,
            scale_group_size_k=block_size,
            transA=False,
            transB=False,
        )

        # WGrad TN:
        # grad_w = grad_out.T @ x
        # [n,k] = [m, n] * [m, k]
        # grad_w = gemm_fp8_blockwise_tn_impl(
        #     grad_out_fp8_col,
        #     x_fp8_col,
        #     grad_out_scales_col,
        #     x_scales_col,
        #     out_dtype=grad_out.dtype,
        #     block_size=block_size,
        # )
        grad_w = gemm_fp8_blockwise_impl(
            grad_out_fp8_col,
            x_fp8_col,
            grad_out_scales_col,
            x_scales_col,
            out_dtype=grad_out.dtype,
            scale_group_size_m=1,
            scale_group_size_n=1,
            scale_group_size_k=block_size,
            transA=True,
            transB=False,
        )

        if orig_shape is not None:
            grad_x = grad_x.view(orig_shape)  # [B, M, K]
        return grad_x, grad_w, None, None


# TODO: weight support Float8Tensor
def gemm_fp8_blockwise(
    x,
    weight,
    config: Optional[MXQuantConfig] = None,
):
    """
    Blockwise GEMM using FP8 quantization.

    This function applies blockwise FP8 quantization to both the input `x` and the `weight`,
    performs matrix multiplication in the FP8 domain, and returns the result in the original
    input precision (e.g., bf16 or fp16).

    Args:
        x (torch.Tensor): Input tensor of shape [M, K], typically in bf16 or fp16.
        weight (torch.Tensor): Weight tensor of shape [N, K], typically in bf16 or fp16.
        config (MXQuantConfig, optional): Quantization configuration that controls FP8 dtype,
            scaling strategy, granularity, and block size. If None, uses default MXQuantConfig.

    Returns:
        torch.Tensor: Output tensor of shape [M, N], in the same dtype as `x`.

    Example:
        >>> x = torch.randn(512, 1024, dtype=torch.bfloat16).cuda()
        >>> w = torch.randn(768, 1024, dtype=torch.bfloat16).cuda()
        >>> y = gemm_fp8_blockwise(x, w)
    """
    if config is None:
        config = MXQuantConfig()
    return BlockwiseFP8GemmFunction.apply(x, weight, config)
