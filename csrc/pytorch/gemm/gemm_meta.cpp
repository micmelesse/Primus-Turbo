#include <torch/extension.h>

namespace primus_turbo::pytorch {

at::Tensor gemm_meta(const at::Tensor A, const at::Tensor B, at::ScalarType out_dtype,
                     const bool transA, bool transB) {
    const int64_t m = transA ? A.size(1) : A.size(0);
    const int64_t n = transB ? B.size(0) : B.size(1);

    return at::empty({m, n}, torch::dtype(out_dtype).device(at::kMeta));
}

torch::Tensor gemm_fp8_blockwise_meta(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                      torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                      const bool transB, const int64_t block_size) {
    const int32_t m = transA ? a.size(1) : a.size(0);
    const int32_t n = transB ? b.size(0) : b.size(1);
    return at::empty({m, n}, c.options().device(at::kMeta));
}

} // namespace primus_turbo::pytorch
