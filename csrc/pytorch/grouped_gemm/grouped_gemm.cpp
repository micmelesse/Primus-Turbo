#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "../type_traits.h"
#include "ck_tile/core/numeric/half.hpp"
// #include "ck_tile/ops/common/tensor_layout.hpp"
namespace primus_turbo::pytorch {

at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &c, at::Tensor &seg_lens,
                        const bool transA, const bool transB) {

    TORCH_CHECK(a.dtype() == b.dtype() && b.dtype() == c.dtype(),
                "All tensors must have the same dtype, got ", a.dtype(), ", ", b.dtype(), ", and ",
                c.dtype());

    TORCH_CHECK(a.dtype() == at::kHalf || a.dtype() == at::kBFloat16,
                "Only fp16 and bf16 are supported, got ", a.dtype());

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    const int                    B           = b.size(0);
    const int                    group_count = B;
    auto                         stream      = at::cuda::getCurrentCUDAStream();
    void                        *temp_ptr    = ck_grouped_gemm_init(group_count, stream);
    ck_tile::GemmTransKernelArg *kargs_ptr   = static_cast<ck_tile::GemmTransKernelArg *>(temp_ptr);

    if (a.dtype() == at::kHalf) {
        using AType = ck_tile::half_t;
        using BType = ck_tile::half_t;
        using CType = ck_tile::half_t;
        if (!transA && transB) // NT
        {
            const int N = b.size(1);
            const int K = b.size(2);
            ck_grouped_gemm_kernel<AType, BType, CType, Row, Col, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int *>(seg_lens.data_ptr()), group_count, N, K, K, K, N, 1,
                stream);
        } else if (!transA && !transB) // NN
        {
            const int N = b.size(2);
            const int K = b.size(1);
            ck_grouped_gemm_kernel<AType, BType, CType, Row, Row, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int *>(seg_lens.data_ptr()), group_count, N, K, K, N, N, 1,
                stream);
        } else {
            TORCH_CHECK(false, "Unsupported: transA = ", transA, ", transB = ", transB);
        }
    } else if (a.dtype() == at::kBFloat16) {
        using AType = ck_tile::bfloat16_t;
        using BType = ck_tile::bfloat16_t;
        using CType = ck_tile::bfloat16_t;
        if (!transA && transB) // NT
        {
            const int N = b.size(1);
            const int K = b.size(2);
            ck_grouped_gemm_kernel<AType, BType, CType, Row, Col, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int *>(seg_lens.data_ptr()), group_count, N, K, K, K, N, 1,
                stream);
        } else if (!transA && !transB) // NN
        {
            const int N = b.size(2);
            const int K = b.size(1);
            ck_grouped_gemm_kernel<AType, BType, CType, Row, Row, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int *>(seg_lens.data_ptr()), group_count, N, K, K, N, N, 1,
                stream);
        } else {
            TORCH_CHECK(false, "Unsupported: transA = ", transA, ", transB = ", transB);
        }
    }

    return c;
}

at::Tensor grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &c,
                                   at::Tensor &seg_lens, const bool transA, const bool transB) {
    TORCH_CHECK(a.dtype() == b.dtype() && b.dtype() == c.dtype(),
                "All tensors must have the same dtype, got ", a.dtype(), ", ", b.dtype(), ", and ",
                c.dtype());

    TORCH_CHECK(a.dtype() == at::kHalf || a.dtype() == at::kBFloat16,
                "Only fp16 and bf16 are supported, got ", a.dtype());

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    const int64_t                B           = seg_lens.numel();
    const int                    group_count = B;
    auto                         stream      = at::cuda::getCurrentCUDAStream();
    void                        *temp_ptr    = ck_grouped_gemm_init(group_count, stream);
    ck_tile::GemmTransKernelArg *kargs_ptr   = static_cast<ck_tile::GemmTransKernelArg *>(temp_ptr);

    if (a.dtype() == at::kHalf) {
        using AType = ck_tile::half_t;
        using BType = ck_tile::half_t;
        using CType = ck_tile::half_t;
        if (transA && !transB) // TN
        {
            const int M = a.size(1);
            const int N = b.size(1);
            ck_grouped_gemm_variable_k_kernel<AType, BType, CType, Col, Row, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int *>(seg_lens.data_ptr()), group_count, M, N, M, N, N, 1,
                stream);
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
            ck_grouped_gemm_variable_k_kernel<AType, BType, CType, Col, Row, Row>(
                kargs_ptr, reinterpret_cast<const AType *>(a.data_ptr()),
                reinterpret_cast<const BType *>(b.data_ptr()),
                reinterpret_cast<CType *>(c.data_ptr()),
                reinterpret_cast<const int *>(seg_lens.data_ptr()), group_count, M, N, M, N, N, 1,
                stream);
        } else {
            TORCH_CHECK(false, "Unsupported: transA = ", transA, ", transB = ", transB);
        }
    }

    return c;
}

} // namespace primus_turbo::pytorch
