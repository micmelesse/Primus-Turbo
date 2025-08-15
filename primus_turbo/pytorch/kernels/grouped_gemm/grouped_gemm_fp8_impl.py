###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import triton

import primus_turbo.pytorch as turbo
from primus_turbo.triton.grouped_gemm.grouped_gemm_fp8_kernel import (
    compute_m_num_tiles_indptr,
    grouped_gemm_fp8_blockwise_kernel,
    grouped_gemm_variable_k_fp8_blockwise_tn_kernel,
)


def grouped_gemm_fp8_blockwise_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    batch_size: int,
    seg_indptr: torch.Tensor,  # [B+1,] int64
    out_dtype: torch.dtype,
    scale_group_size_m: int,
    scale_group_size_n: int,
    scale_group_size_k: int,
    transA: bool,
    transB: bool,
):
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a_scales.dim() == 2, f"a scales must be 2D, got {a_scales.shape}"
    assert b_scales.dim() == 3, f"b scales must be 3D, got {b_scales.shape}"

    M = a.shape[1] if transA else a.shape[0]
    Ka = a.shape[0] if transA else a.shape[1]
    Kb = b.shape[-1] if transB else b.shape[-2]
    N = b.shape[-2] if transB else b.shape[-1]
    B = b.shape[0]

    assert Ka == Kb, f"K mismatch: Ka={Ka}, Kb={Kb}"
    assert B == batch_size

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": scale_group_size_k,
        "num_stages": 2,
        "num_warps": 4,
    }

    c = torch.empty(M, N, dtype=out_dtype, device=a.device)

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_fp8_blockwise_kernel[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        batch_size,
        M,
        N,
        Ka,
        seg_indptr,
        m_num_tiles_indptr,
        transA,
        transB,
        scale_group_size_m,
        scale_group_size_n,
        scale_group_size_k,
        **config,
    )
    return c


def grouped_gemm_variable_k_fp8_blockwise_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    batch_size: int,
    seg_indptr: torch.Tensor,  # [B+1,] int64
    scales_seg_indptr: torch.Tensor,
    out_dtype: torch.dtype,
    scale_group_size_m: int,
    scale_group_size_n: int,
    scale_group_size_k: int,
    transA: bool,
    transB: bool,
):
    assert transA == True and transB == False, "Only transA=True and transB=False are supported."
    assert (
        seg_indptr.shape[0] == batch_size + 1
    ), f"Expected seg_indptr shape [{batch_size + 1}], got {seg_indptr.shape}"
    assert (
        scales_seg_indptr.shape[0] == batch_size + 1
    ), f"Expected scales_seg_indptr shape [{batch_size + 1}], got {scales_seg_indptr.shape}"

    assert (
        scale_group_size_m == 1 and scale_group_size_n == 1
    ), f"Only scale_group_size_m == 1 and scale_group_size_n == 1 are supported, got {scale_group_size_m}, {scale_group_size_n}"

    # a_view = a.transpose(-1, -2) if transA else a
    # a_scales_view = a_scales.transpose(-1, -2) if transA else a_scales
    # b_view = b.transpose(-1, -2) if transB else b
    # b_scales_view = b_scales.transpose(-1, -2) if transB else b_scales

    M = a.shape[1] if transA else a.shape[0]
    Ka = a.shape[0] if transA else a.shape[1]
    Kb = b.shape[-1] if transB else b.shape[-2]
    N = b.shape[-2] if transB else b.shape[-1]
    assert Ka == Kb, f"K mismatch: KA={Ka}, KB={Kb}"

    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": scale_group_size_k,
        "num_stages": 2,
        "num_warps": 4,
    }

    c = torch.empty(batch_size, M, N, dtype=out_dtype, device=a.device)

    grid = lambda META: (
        batch_size,
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_variable_k_fp8_blockwise_tn_kernel[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        batch_size,
        M,
        N,
        Ka,
        seg_indptr,
        scales_seg_indptr,
        scale_group_size_m,
        scale_group_size_n,
        scale_group_size_k,
        **config,
    )
    return c


def grouped_gemm_compute_offs(group_lens: torch.Tensor) -> torch.Tensor:
    group_offs = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(group_lens)
    return group_offs


# row-wise
def grouped_gemm_csrc_fp8_row_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in [turbo.float8_e4m3, turbo.float8_e5m2], f"a must be float8, got {a.dtype}"
    assert b.dtype in [turbo.float8_e4m3, turbo.float8_e5m2], f"b must be float8, got {b.dtype}"
    assert trans_a == False

    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_fp8(
        a, b, group_lens, group_offs, trans_a, trans_b
    )
    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_fp8_dequant(
        out, group_lens, group_offs, a_scales, b_scales
    )
    return out


def grouped_gemm_variable_k_fp8_row_csrc_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    assert a.dim() == 2, f"a must be 2D, got {a.shape}"
    assert b.dim() == 2, f"b must be 2D, got {b.shape}"
    assert a.dtype in [turbo.float8_e4m3, turbo.float8_e5m2], f"a must be float8, got {a.dtype}"
    assert b.dtype in [turbo.float8_e4m3, turbo.float8_e5m2], f"b must be float8, got {b.dtype}"
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."
    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_fp8_variable_k(
        a, b, group_lens, group_offs, trans_a, trans_b
    )
    out = dequantize_grouped_gemm_fp8_output(
        out,
        a_scales,
        b_scales,
    )
    return out


def dequantize_grouped_gemm_fp8_output(
    input: torch.Tensor,  # [B, N, K] (FP8 quantized)
    a_scales: torch.Tensor,  # [1, N] (row-wise scales for A)
    b_scales: torch.Tensor,  # [1, K] (per-column scales for B)
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize the result of grouped GEMM FP8 operation.

    Args:
        input: The output from grouped GEMM FP8 operation with shape [B, N, K]
        a_scales: Scale factors for tensor A with shape [1, N]
        b_scales: Scale factors for tensor B with shape [1, K]
        group_lens: Length of each group with shape [B]
        group_offs: Offset of each group with shape [B]
        output_dtype: The output data type

    Returns:
        torch.Tensor: Dequantized result with shape [B, N, K]
    """
    # Apply the scaling: out = out * a_scales * b_scales
    B, N, K = input.shape

    # Create output tensor
    dequantized_out = torch.empty_like(input, dtype=output_dtype)

    # Expand a_scales from [1, N] to [B, N, 1] and then to [B, N, K]
    a_scales_expanded = a_scales.unsqueeze(-1).expand(B, N, 1).expand(B, N, K)

    # Expand b_scales from [1, K] to [B, N, K]
    b_scales_expanded = b_scales.expand(B, N, K)

    # Combine both scales
    combined_scales = a_scales_expanded * b_scales_expanded

    # Apply scales to the entire output
    dequantized_out = input * combined_scales

    return dequantized_out
