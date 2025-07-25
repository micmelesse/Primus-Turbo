#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "../type_traits.h"
// #include "ck_tile/core/numeric/half.hpp"
// #include "ck_tile/ops/common/tensor_layout.hpp"
namespace primus_turbo::pytorch {

at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &c, at::Tensor &seg_lens,
                        const bool transA, const bool transB) {
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using AType = ck_tile::half_t;
    using BType = ck_tile::half_t;
    using CType = ck_tile::half_t;
    const int B = b.size(0);
    const int N = b.size(1);
    const int K = b.size(2);

    ck_grouped_gemm_kernel<AType, BType, CType, Row, Col, Row>(
        reinterpret_cast<const AType *>(a.data_ptr()),
        reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
        reinterpret_cast<const int *>(seg_lens.data_ptr()), B, N, K);
    return c;
}
} // namespace primus_turbo::pytorch
