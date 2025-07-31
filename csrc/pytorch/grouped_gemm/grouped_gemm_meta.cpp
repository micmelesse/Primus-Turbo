#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                             at::Tensor &group_offs, const bool transA, const bool transB) {
    // int output_rows, output_cols;
    // if (!transA && !transB) { // NN
    //     output_rows = a.size(0);
    //     output_cols = b.size(2);
    // } else if (!transA && transB) { // NT
    //     output_rows = a.size(0);
    //     output_cols = b.size(1);
    // }
    // return at::empty({output_rows, output_cols}, a.options().device(at::kMeta));

    // TODO: out-datatype
    const int64_t m = transA ? a.size(1) : a.size(0);
    const int64_t n = transB ? b.size(1) : b.size(2);
    return at::empty(std::vector<int64_t>{m, n}, a.options().device(at::kMeta));
}

at::Tensor grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &seg_lens,
                                        const bool transA, const bool transB) {
    // set for NT
    const int64_t B = seg_lens.numel();
    const int64_t M = a.size(1);
    const int64_t N = b.size(1);

    return at::empty({B, M, N}, a.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
