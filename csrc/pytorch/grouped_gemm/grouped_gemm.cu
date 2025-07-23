#include "../type_traits.h"
#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
namespace primus_turbo::pytorch {

torch::Tensor grouped_gemm(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c,
                           torch::Tensor &seg_lens, const bool transA, const bool transB) {
    using Row            = ck_tile::tensor_layout::gemm::RowMajor;
    using Col            = ck_tile::tensor_layout::gemm::ColumnMajor;
    using AType          = typename TorchToCKType<torch::kHalf>::type;
    using BType          = AType;
    using CType          = AType;
    const int32_t M      = transA ? a.size(1) : a.size(0);
    const int32_t K      = transA ? a.size(0) : a.size(1);
    const int32_t N      = transB ? b.size(0) : b.size(1);
    auto          stream = at::cuda::getCurrentCUDAStream();
    ck_grouped_gemm_kernel<AType, BType, CType, Row, Col, Row>(
        reinterpret_cast<const AType *>(a.data_ptr()),
        reinterpret_cast<const BType *>(b.data_ptr()), reinterpret_cast<CType *>(c.data_ptr()),
        reinterpret_cast<const int *>(seg_lens.data_ptr()), M, N, K, stream);
    return c;
}
} // namespace primus_turbo::pytorch
