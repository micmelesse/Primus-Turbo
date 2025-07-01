import torch

__all__ = ["gemm"]


def gemm(A: torch.Tensor, B: torch.Tensor, out_dtype: torch.dtype, layout: str) -> torch.Tensor:
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    args = (
        A,
        B,
        out_dtype,
        transa,
        transb,
    )

    # TODO(ruibzhan): support more backends.
    out = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(*args)

    return out
