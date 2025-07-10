import torch


def _empty_tensor():
    return torch.Tensor()


def gemm_impl(
    A: torch.Tensor,
    transA: bool,
    B: torch.Tensor,
    transB: bool,
    out_dtype: torch.dtype,
    transC: bool,
    backend="hipblaslt",
) -> torch.Tensor:
    assert backend in ("hipblaslt")

    args = (
        A,
        _empty_tensor(),
        B,
        _empty_tensor(),
        out_dtype,
        transA,
        transB,
        transC,
    )

    if backend == "hipblaslt":
        # TODO(ruibzhan): support more backends.
        out = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(*args)

    return out
