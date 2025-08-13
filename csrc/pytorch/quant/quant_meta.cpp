// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../extensions.h"

namespace primus_turbo::pytorch {

at::Tensor fp8_quantize_meta(const at::Tensor input, const at::Tensor scale,
                             const at::ScalarType dest_dtype) {
    return torch::empty_like(input, torch::dtype(dest_dtype).device(at::kMeta));
}

at::Tensor fp8_dequantize_meta(const at::Tensor input, const at::Tensor scale_inv,
                               const at::ScalarType dest_dtype) {
    return torch::empty_like(input, torch::dtype(dest_dtype).device(at::kMeta));
}

at::Tensor fp8_quantize_row_col_meta(at::Tensor &input, at::Tensor &scale,
                                     const bool is_row_major) {
    int64_t    b = 1, m = 0, k = 0;
    at::Tensor output;
    int64_t    dims = input.ndimension();
    if (dims == 2) {
        m = input.size(0);
        k = input.size(1);

        output = at::empty({m, k}, at::dtype(at::kFloat8_e4m3fnuz).device(at::kCUDA));
    }

    else if (dims == 3) {
        b      = input.size(0);
        m      = input.size(1);
        k      = input.size(2);
        output = at::empty({b, m, k}, at::dtype(at::kFloat8_e4m3fnuz).device(at::kCUDA));
    }
    return output;
}

at::Tensor grouped_gemm_fp8_dequant_meta(at::Tensor &input, at::Tensor &group_lens,
                                         at::Tensor &scale_a, at::Tensor &scale_b) {
    int64_t    b = 1, m = 0, n = 0;
    at::Tensor output;

    m = input.size(0);
    n = input.size(1);

    output = at::empty({m, n}, at::dtype(input.dtype()).device(at::kCUDA));

    return output;
}

} // namespace primus_turbo::pytorch
