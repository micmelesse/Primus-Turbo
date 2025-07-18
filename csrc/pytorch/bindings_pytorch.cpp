#include <torch/extension.h>

#include "extensions.h"

namespace primus_turbo::pytorch {

/********************************************/

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    m.def("hipblaslt_gemm(Tensor A, Tensor B, ScalarType out_dtype, bool transA, bool transB, bool "
          "transC) -> "
          "Tensor");
    m.def("gemm_fp8_blockwise("
          "Tensor a, Tensor a_scales, "
          "Tensor b, Tensor b_scales, "
          "Tensor c, "
          "bool transA, bool transB, "
          "int block_size"
          ") -> Tensor");
    m.def("fp8_quantize(Tensor input, Tensor scale, ScalarType dest_dtype) -> Tensor");
    m.def("fp8_dequantize(Tensor input, Tensor scale_inv, ScalarType dest_dtype) -> Tensor");
    m.def("rmsnorm_fwd(Tensor input, Tensor gamma, float eps) -> Tensor");
    m.def("rmsnorm_bwd(Tensor input, Tensor gamma, Tensor grad_out, float eps) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    m.impl("hipblaslt_gemm", hipblaslt_gemm);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise);
    m.impl("fp8_quantize", fp8_quantize);
    m.impl("fp8_dequantize", fp8_dequantize);
    m.impl("rmsnorm_fwd", rmsnorm_fwd);
    m.impl("rmsnorm_bwd", rmsnorm_bwd);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    m.impl("hipblaslt_gemm", hipblaslt_gemm_meta);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise_meta);
    m.impl("fp8_quantize", fp8_quantize_meta);
    m.impl("fp8_dequantize", fp8_dequantize_meta);
    m.impl("rmsnorm_fwd", rmsnorm_fwd_meta);
    m.impl("rmsnorm_bwd", rmsnorm_bwd_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

/********************************************/

} // namespace primus_turbo::pytorch
