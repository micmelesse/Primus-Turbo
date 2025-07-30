import torch

from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_csrc_impl,
    grouped_gemm_variable_k_csrc_impl,
)

__all__ = ["grouped_gemm"]


class GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,  # [B * M, K],
        b: torch.Tensor,  # [B, N, K]
        seg_lens: torch.Tensor,  # [B,] int64
    ):
        # [B * M, N]
        out = grouped_gemm_csrc_impl(
            a,
            b,
            seg_lens,
            transA=False,
            transB=True,
        )
        ctx.save_for_backward(a, b, seg_lens)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b, seg_lens = ctx.saved_tensors
        grad_a = grouped_gemm_csrc_impl(
            grad_out,
            b,
            seg_lens,
            transA=False,
            transB=False,
        )

        grad_b = grouped_gemm_variable_k_csrc_impl(
            grad_out,
            a,
            seg_lens,
            transA=True,
            transB=False,
        )
        return grad_a, grad_b, None


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    seg_lens: torch.Tensor,
):
    return GroupedGemmFunc.apply(a, b, seg_lens)
