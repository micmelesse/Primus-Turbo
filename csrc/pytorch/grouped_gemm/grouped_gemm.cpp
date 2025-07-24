#include "primus_turbo/grouped_gemm.h"
#include "../extensions.h"
#include "../type_traits.h"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
namespace primus_turbo::pytorch {

at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &c, at::Tensor &seg_lens,
                        const bool transA, const bool transB) {
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    // using AType = float;
    // using BType = float;
    // using CType = float;
    const int    B          = b.size(0);
    const int    N          = b.size(1);
    const int    K          = b.size(2);
    const float *p_a        = static_cast<const float *>(a.data_ptr());
    const float *p_b        = static_cast<const float *>(b.data_ptr());
    float       *p_c        = static_cast<float *>(c.data_ptr());
    const int   *p_seg_lens = static_cast<const int *>(seg_lens.data_ptr());
    // const int32_t M      = transA ? a.size(1) : a.size(0);
    // const int32_t K      = transA ? a.size(0) : a.size(1);
    // const int32_t N      = transB ? b.size(0) : b.size(1);
    // auto          stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // printf("hihihi\n");
    ck_grouped_gemm_kernel<float, float, float>(p_a, p_b, p_c, p_seg_lens, B, N, K);

    ck_test();
    return c;
}
} // namespace primus_turbo::pytorch
