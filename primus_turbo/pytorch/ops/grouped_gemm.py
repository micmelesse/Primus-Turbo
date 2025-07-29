import torch

from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_csrc_impl,
    grouped_gemm_csrc_init,
    grouped_gemm_variable_k_csrc_impl,
)

__all__ = ["grouped_gemm", "grouped_gemm_init"]


class GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,  # [B * M, K],
        b: torch.Tensor,  # [B, N, K]
        seg_lens: torch.Tensor,  # [B,] int64
        init_ptr: torch.Tensor = None,
    ):
        B = b.size(0)
        if init_ptr is None:
            init_ptr_inner = grouped_gemm_csrc_init(torch.tensor(B, dtype=torch.int64, device="cuda"))
        else:
            init_ptr_inner = init_ptr
        # [B * M, N]
        out = grouped_gemm_csrc_impl(
            a,
            b,
            seg_lens,
            transA=False,
            transB=True,
            init_ptr=init_ptr_inner,
        )
        ctx.save_for_backward(a, b, seg_lens, init_ptr_inner)
        return out, init_ptr_inner

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) == 2:
            grad_out, _ = grad_outputs
        else:
            raise ValueError("Unexpected number of gradients")
        a, b, seg_lens, init_ptr = ctx.saved_tensors
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
    init_ptr: torch.Tensor = None,
):
    return GroupedGemmFunc.apply(a, b, seg_lens, init_ptr)


def grouped_gemm_init(group_size: torch.Tensor) -> torch.Tensor:
    init_ptr = grouped_gemm_csrc_init(group_size)
    return init_ptr
