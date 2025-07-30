import torch


def grouped_gemm_csrc_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    seg_lens: torch.Tensor,
    transA: bool,
    transB: bool,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert (transA == False and transB == True) or (
        transA == False and transB == False
    ), f"Only NT (transA=False, transB=True) and NN (transA=False, transB=False) modes are supported, got transA={transA}, transB={transB}"
    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, seg_lens, transA, transB)
    return out


def grouped_gemm_variable_k_csrc_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    seg_lens: torch.Tensor,
    transA: bool,
    transB: bool,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [torch.float16, torch.bfloat16], f"a must be float16 or bfloat16, got {a.dtype}"
    assert b.dtype in [torch.float16, torch.bfloat16], f"b must be float16 or bfloat16, got {b.dtype}"
    assert transA == True and transB == False, "Only transA=True and transB=False are supported."
    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_variable_k(a, b, seg_lens, transA, transB)
    return out
