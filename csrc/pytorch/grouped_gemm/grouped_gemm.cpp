#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "../type_traits.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                        at::Tensor &group_offs, const bool transA, const bool transB) {
    // TODO:
    auto out_dtype = a.scalar_type();

    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");

    // Alloc args workspace
    const int64_t args_sizes = get_workspace_size(group_lens.numel());
    at::Tensor    args_tensor =
        at::empty({args_sizes}, at::TensorOptions().dtype(at::kByte).device(group_lens.device()));

    // Determine output tensor size based on transA and transB
    const int32_t bs = b.size(0);
    const int32_t m  = transA ? a.size(1) : a.size(0);
    const int32_t n  = transB ? b.size(1) : b.size(2);
    const int32_t k  = transA ? a.size(0) : a.size(1);
    at::Tensor    c  = at::empty({m, n}, at::dtype(out_dtype).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();
    if (a.dtype() == at::kHalf) {
        using AType = typename TorchToCKTileType<at::kHalf>::type;
        using BType = AType;
        using CType = AType;
        ck_grouped_gemm<AType, BType, CType>(
            args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
            reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
            reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
            reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n, k,
            stream);
    } else if (a.dtype() == at::kBFloat16) {
        using AType = typename TorchToCKTileType<at::kBFloat16>::type;
        using BType = AType;
        using CType = AType;
        ck_grouped_gemm<AType, BType, CType>(
            args_tensor.data_ptr(), reinterpret_cast<const AType *>(a.data_ptr()),
            reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
            reinterpret_cast<const int64_t *>(group_lens.data_ptr()),
            reinterpret_cast<const int64_t *>(group_offs.data_ptr()), transA, transB, bs, m, n, k,
            stream);
    } else {
        PRIMUS_TURBO_CHECK(false, "GroupedGemm only support float16 and bfloat16");
    }
    return c;
}

at::Tensor grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &seg_lens,
                                   const bool transA, const bool transB) {
    TORCH_CHECK(a.dtype() == b.dtype(), "All tensors must have the same dtype, got ", a.dtype(),
                ", ", b.dtype());

    TORCH_CHECK(a.dtype() == at::kHalf || a.dtype() == at::kBFloat16,
                "Only fp16 and bf16 are supported, got ", a.dtype());

    const int64_t args_workspace_sizes  = get_workspace_size(seg_lens.numel());
    at::Tensor    args_workspace_tensor = at::empty(
        {args_workspace_sizes}, at::TensorOptions().dtype(at::kByte).device(seg_lens.device()));

    at::Tensor c;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    const int64_t                B           = seg_lens.numel();
    const int                    group_count = B;
    auto                         stream      = at::cuda::getCurrentCUDAStream();
    ck_tile::GemmTransKernelArg *kargs_ptr =
        reinterpret_cast<ck_tile::GemmTransKernelArg *>(args_workspace_tensor.data_ptr());

    if (a.dtype() == at::kHalf) {
        using AType = ck_tile::half_t;
        using BType = ck_tile::half_t;
        using CType = ck_tile::half_t;
        if (transA && !transB) // TN
        {
            const int M = a.size(1);
            const int N = b.size(1);
            c           = at::empty({B, M, N}, a.options());
            ck_grouped_gemm_variable_k_kernel<AType, BType, CType, Col, Row, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(seg_lens.data_ptr()), group_count, M, N, M, N, N,
                1, stream);
        } else {
            TORCH_CHECK(false, "Unsupported: transA = ", transA, ", transB = ", transB);
        }
    } else if (a.dtype() == at::kBFloat16) {
        using AType = ck_tile::bfloat16_t;
        using BType = ck_tile::bfloat16_t;
        using CType = ck_tile::bfloat16_t;
        if (transA && !transB) // TN
        {
            const int M = a.size(1);
            const int N = b.size(1);
            c           = at::empty({B, M, N}, a.options());
            ck_grouped_gemm_variable_k_kernel<AType, BType, CType, Col, Row, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int64_t *>(seg_lens.data_ptr()), group_count, M, N, M, N, N,
                1, stream);
        } else {
            TORCH_CHECK(false, "Unsupported: transA = ", transA, ", transB = ", transB);
        }
    }

    return c;
}

} // namespace primus_turbo::pytorch
