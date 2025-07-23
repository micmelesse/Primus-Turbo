#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch {

torch::Tensor grouped_gemm_meta(torch::Tensor &a, torch::Tensor &b, torch::Tensor &c,
                                torch::Tensor &seg_lens, const bool transA, const bool transB) {
    // tmp set for TN
    const int64_t B_M = a.size(0);
    const int64_t N   = b.size(1);

    return at::empty({B_M, N}, c.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
