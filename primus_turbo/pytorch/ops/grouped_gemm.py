import torch

from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_csrc_impl,
    grouped_gemm_csrc_init,
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
        init_ptr: int = 0,
    ):
        B = b.size(0)
        init_ptr_inner = init_ptr
        if init_ptr == 0:
            init_ptr_inner = grouped_gemm_csrc_init(B)
        # [B * M, N]
        out = grouped_gemm_csrc_impl(
            a,
            b,
            seg_lens,
            transA=False,
            transB=True,
            init_ptr=init_ptr_inner,
        )
        ctx.save_for_backward(a, b, seg_lens)
        ctx.init_ptr = init_ptr_inner
        return (out, init_ptr_inner) if init_ptr == 0 else out

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) == 1:
            grad_out = grad_outputs[0]
        elif len(grad_outputs) == 2:
            grad_out, _ = grad_outputs
        else:
            raise ValueError("Unexpected number of gradients")
        a, b, seg_lens = ctx.saved_tensors
        init_ptr = ctx.init_ptr
        grad_a = grouped_gemm_csrc_impl(
            grad_out,
            b,
            seg_lens,
            transA=False,
            transB=False,
            init_ptr=init_ptr,
        )

        grad_b = grouped_gemm_variable_k_csrc_impl(
            grad_out,
            a,
            seg_lens,
            transA=True,
            transB=False,
            init_ptr=init_ptr,
        )
        return (grad_a, grad_b, None, None)


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    seg_lens: torch.Tensor,
    init_ptr: int = 0,
):
    return GroupedGemmFunc.apply(a, b, seg_lens, init_ptr)
