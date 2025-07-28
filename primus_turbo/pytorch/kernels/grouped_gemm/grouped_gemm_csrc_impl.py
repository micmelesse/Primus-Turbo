import torch

_torch_custom_op_wrapper = torch.library.custom_op


def grouped_gemm_csrc_init(group_size: int) -> int:
    return torch.ops.primus_turbo_cpp_extension.init_grouped_gemm(group_size)


@_torch_custom_op_wrapper("primus_turbo::grouped_gemm_csrc_impl", mutates_args=(), device_types="cuda")
def grouped_gemm_csrc_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    seg_lens: torch.Tensor,
    transA: bool,
    transB: bool,
    init_ptr: int,  # must do grouped_gemm_csrc_init before grouped_gemm_csrc_impl
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert (transA == False and transB == True) or (
        transA == False and transB == False
    ), f"Only NT (transA=False, transB=True) and NN (transA=False, transB=False) modes are supported, got transA={transA}, transB={transB}"
    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, seg_lens, transA, transB, init_ptr)
    return out


@_torch_custom_op_wrapper(
    "primus_turbo::grouped_gemm_variable_k_csrc_impl", mutates_args=(), device_types="cuda"
)
def grouped_gemm_variable_k_csrc_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    seg_lens: torch.Tensor,
    transA: bool,
    transB: bool,
    init_ptr: int,  # must do grouped_gemm_csrc_init before grouped_gemm_variable_k_fp8_blockwise_impl
) -> torch.Tensor:
    assert transA == True and transB == False, "Only transA=True and transB=False are supported."
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_variable_k(
        a, b, seg_lens, transA, transB, init_ptr
    )
    return out
