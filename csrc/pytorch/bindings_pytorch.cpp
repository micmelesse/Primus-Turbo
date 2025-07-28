#include <torch/extension.h>

#include "extensions.h"

namespace primus_turbo::pytorch {

/********************************************/

TORCH_LIBRARY(primus_turbo_cpp_extension, m) {
    m.def("hipblaslt_gemm(Tensor A, Tensor scaleA_inv, Tensor B, Tensor scaleB_inv,"
          "ScalarType out_dtype, bool transA, bool transB, bool "
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
    m.def("fused_all_gather_matmul(Tensor A_shard, Tensor[] Bs, int gather_dim, str group_name, "
          "Tensor? A_out, Tensor[]? mm_outs, ScalarType[]? out_dtypes, bool return_A=True, bool "
          "skip_copy_local_ag_out=False)->(Tensor[])");
    // m.def("fused_all_gather_scaled_matmul(Tensor A_shard, Tensor[] Bs, Tensor A_scale, Tensor[] "
    //       "B_scales, int gather_dim, str group_name, bool[] use_fast_accum, Tensor[]? biases, "
    //       "Tensor[]? result_scales, ScalarType[]? out_dtypes,"
    //       "Tensor? A_out, Tensor[]? mm_outs, str? comm_algo, int? num_splits, bool? "
    //       "skip_copy_local_ag_out, bool? enable_sdma)->(Tensor[])");
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, CUDA, m) {
    m.impl("hipblaslt_gemm", hipblaslt_gemm);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise);
    m.impl("fp8_quantize", fp8_quantize);
    m.impl("fp8_dequantize", fp8_dequantize);
    m.impl("fused_all_gather_matmul", fused_all_gather_matmul);
    // m.impl("fused_all_gather_scaled_matmul", fused_all_gather_scaled_matmul);
}

TORCH_LIBRARY_IMPL(primus_turbo_cpp_extension, Meta, m) {
    m.impl("hipblaslt_gemm", hipblaslt_gemm_meta);
    m.impl("gemm_fp8_blockwise", gemm_fp8_blockwise_meta);
    m.impl("fp8_quantize", fp8_quantize_meta);
    m.impl("fp8_dequantize", fp8_dequantize_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

/********************************************/

} // namespace primus_turbo::pytorch
