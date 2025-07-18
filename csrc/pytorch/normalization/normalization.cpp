#include "primus_turbo/normalization.h"

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor rmsnorm_fwd(const at::Tensor &input, const at::Tensor &gamma, const double eps) {
    TORCH_CHECK(input.is_contiguous(), "rmsnorm_fwd: input must be contiguous.");
    TORCH_CHECK(gamma.is_contiguous(), "rmsnorm_fwd: gamma must be contiguous.");

    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;
    auto          output    = at::empty_like(input);

    TORCH_CHECK(input.numel() % inner_len == 0, "input.numel() must be divisible by gamma.numel()");

    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_fwd_impl<float>(input.data_ptr<float>(), gamma.data_ptr<float>(),
                            output.data_ptr<float>(), inner_len, outer_len, static_cast<float>(eps),
                            stream);

    return output;
}

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &input, const at::Tensor &gamma,
                                    const at::Tensor &grad_output, const double eps) {
    TORCH_CHECK(input.is_contiguous(), "rmsnorm_bwd: input must be contiguous.");
    TORCH_CHECK(gamma.is_contiguous(), "rmsnorm_bwd: gamma must be contiguous.");
    TORCH_CHECK(grad_output.is_contiguous(), "rmsnorm_bwd: grad_output must be contiguous.");

    const int64_t inner_len = gamma.numel();
    const int64_t outer_len = input.numel() / inner_len;

    TORCH_CHECK(input.numel() % inner_len == 0, "input.numel() must be divisible by gamma.numel()");

    auto input_grad = at::empty_like(input);
    auto gamma_grad = at::empty_like(input);

    auto stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_bwd_impl<float>(input.data_ptr<float>(), gamma.data_ptr<float>(),
                            grad_output.data_ptr<float>(), input_grad.data_ptr<float>(),
                            gamma_grad.data_ptr<float>(), inner_len, outer_len,
                            static_cast<float>(eps), stream);

    return {input_grad, gamma_grad};
}

} // namespace primus_turbo::pytorch
