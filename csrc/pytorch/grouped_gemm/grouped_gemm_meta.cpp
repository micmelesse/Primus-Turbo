#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

int64_t init_grouped_gemm_meta(const at::Tensor &group_count) {
    return 0;
}

at::Tensor grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &seg_lens, const bool transA,
                             const bool transB, int64_t temp_ptr) {
    int output_rows = 0, output_cols = 0;
    if (!transA && !transB) { // NN
        output_rows = a.size(0);
        output_cols = b.size(2);
    } else if (!transA && transB) { // NT
        output_rows = a.size(0);
        output_cols = b.size(1);
    }
    return at::empty({output_rows, output_cols}, a.options().device(at::kMeta));
}

at::Tensor grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &seg_lens,
                                        const bool transA, const bool transB, int64_t temp_ptr) {
    // set for NT
    const int64_t B = seg_lens.numel();
    const int64_t M = a.size(1);
    const int64_t N = b.size(1);

    return at::empty({B, M, N}, a.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
