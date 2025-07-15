from typing import Optional, Union

import torch

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    MXQuantConfig,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
    gemm_fp8_blockwise_impl,
    gemm_fp8_tensorwise_impl,
    quant_fp8_blockwise_for_weight_impl,
)
from primus_turbo.pytorch.kernels.quantize import (
    quant_fp8_blockwise_impl,
    quant_fp8_tensorwise_impl,
)

__all__ = ["gemm_fp8_blockwise", "gemm_fp8_tensorwise"]


# TODO: opt and refact
class BlockwiseFP8GemmFunction(torch.autograd.Function):
    """
    Autograd function for FP8 blockwise GEMM.

    This class implements the forward and backward pass for a GEMM using blockwise FP8 quantization.

    Forward:
        - Quantize input `x` row-wise into FP8 blockwise.
        - Quantize weight `weight` blockwise.
        - Perform matrix multiplication: Y = x @ weight.T (NT layout).
        - Output result in high precision (same dtype as `x`).

    Backward:
        - Quantize `grad_out` both row-wise and column-wise for dgrad/wgrad respectively.
        - Re-quantize input `x` column-wise for use in wgrad.
        - Compute:
            * grad_x = grad_out @ weight      (NN layout)
            * grad_w = grad_out.T @ x         (TN layout)
        - Both gradients are returned in high precision.

    Inputs:
        x (Tensor): Input tensor of shape [B0,...,Bn, M, K], typically bf16/fp16.
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
        assert x.ndim >= 2, "Input tensor x must have at least 2 dimensions."
        assert weight.ndim == 2, "Weight tensor must be 2-dimensional."
        assert config.format == Format.E4M3

        x_dtype = float8_e4m3
        w_dtype = float8_e4m3

        block_size = config.block_size

        *batch_shape, M, K = x.shape
        x = x.view(-1, K)  # flatten shape [BM, K]

        # Quantize input activation (row): shape [M, K] → FP8
        x_fp8_row, x_scales_row = quant_fp8_blockwise_impl(x, x_dtype, axis=1, block_size=block_size)
        # Quantize weight blockwise: shape [N, K] → FP8
        w_fp8, w_scales = quant_fp8_blockwise_for_weight_impl(weight, w_dtype, block_size)

        # Perform NT GEMM in quantized domain:
        # out = x_fp8_row @ w_fp8.T → shape [M, N]
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
        out = out.view(*batch_shape, M, -1)  # restore shape

        # Save tensors for backward
        ctx.save_for_backward(x, w_fp8, w_scales)
        ctx.config = config
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        x, w_fp8, w_scales = ctx.saved_tensors
        config = ctx.config
        block_size = config.block_size

        out_dtype = float8_e4m3
        x_dtype = float8_e4m3

        *batch_shape, M, N = grad_out.shape
        grad_out = grad_out.view(-1, N)

        # Quantize grad_out in both row-wise and column-wise directions:
        # - row-wise: for dgrad (grad_x)
        # - col-wise: for wgrad (grad_w)
        grad_out_fp8_row, grad_out_scales_row = quant_fp8_blockwise_impl(grad_out, out_dtype, -1, block_size)
        grad_out_fp8_col, grad_out_scales_col = quant_fp8_blockwise_impl(grad_out, out_dtype, -2, block_size)
        # TODO: dequant + quant kernel
        x_fp8_col, x_scales_col = quant_fp8_blockwise_impl(x, x_dtype, axis=0, block_size=block_size)

        # DGrad NN:
        # grad_x = grad_out @ weight
        # [m, k] = [m, n] * [n, k]
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

        grad_x = grad_x.view(*batch_shape, M, -1)
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


@torch.compile
def calc_scale_and_scale_inv(x: torch.Tensor, fp8_max: float):
    amax = x.abs().amax()
    scale = torch.full([1], fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
    scale_inv = 1.0 / scale

    return scale, scale_inv


class TensorwiseFP8GemmFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        out_dtype: torch.dtype,
        config: Float8QuantConfig,
    ):
        assert config.granularity == ScalingGranularity.TENSORWISE
        assert config.format == Format.E4M3 or config.format == Format.HYBRID

        x_dtype = float8_e4m3
        y_dtype = float8_e4m3

        x_scale, x_scale_inv = calc_scale_and_scale_inv(x, torch.finfo(x_dtype).max)
        y_scale, y_scale_inv = calc_scale_and_scale_inv(y, torch.finfo(y_dtype).max)

        x_fp8 = quant_fp8_tensorwise_impl(x, x_scale, x_dtype)
        y_fp8 = quant_fp8_tensorwise_impl(y, y_scale, y_dtype)

        # NN
        out = gemm_fp8_tensorwise_impl(
            x_fp8, x_scale_inv, False, y_fp8.T.contiguous(), y_scale_inv, True, out_dtype, False
        )

        ctx.save_for_backward(x_fp8, x_scale_inv, y_fp8, y_scale_inv)

        ctx.out_dtype = out_dtype
        ctx.config = config

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x_fp8, x_scale_inv, y_fp8, y_scale_inv) = ctx.saved_tensors
        config = ctx.config

        # For HYBRID format. data type of grad_out is e5m2.
        grad_out_dtype = float8_e4m3 if config.format == Format.E4M3 else float8_e5m2

        grad_out_scale, grad_out_scale_inv = calc_scale_and_scale_inv(
            grad_out, torch.finfo(grad_out_dtype).max
        )

        grad_out_fp8 = quant_fp8_tensorwise_impl(grad_out, grad_out_scale, grad_out_dtype)

        # NT
        x_grad = gemm_fp8_tensorwise_impl(
            grad_out_fp8,
            grad_out_scale_inv,
            False,
            y_fp8,
            y_scale_inv,
            True,
            ctx.out_dtype,
            False,
        )

        # TN
        y_grad = gemm_fp8_tensorwise_impl(
            x_fp8.T.contiguous(),
            x_scale_inv,
            False,
            grad_out_fp8.T.contiguous(),
            grad_out_scale_inv,
            True,
            ctx.out_dtype,
            False,
        )

        return (x_grad, y_grad, None, None)


def gemm_fp8_tensorwise(
    A: torch.Tensor,
    B: torch.Tensor,
    transA: bool = False,
    transB: bool = False,
    out_dtype: Union[torch.dtype, None] = None,
    config: Union[Float8QuantConfig, None] = None,
) -> torch.Tensor:
    assert A.ndim == 2 and B.ndim == 2, "Only 2D tensors are supported"
    # TODO(ruibzhan): We only support NN layout on gfx942.
    assert not transA and not transB, f"Only support NN layout on FP8 tensorwise GEMM."

    if out_dtype is None:
        out_dtype = torch.result_type(A, B)

    if config is None:
        config = Float8QuantConfig()

    args = (A, B, out_dtype, config)

    return TensorwiseFP8GemmFunction.apply(*args)
