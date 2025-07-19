#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/miopen/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/hip/HIPGraphsUtils.cuh>

#include "primus_turbo/common.h"

namespace primus_turbo::pytorch {

/* Quantize */

at::Tensor fp8_quantize(const at::Tensor input, const at::Tensor scale,
                        const at::ScalarType dest_dtype);

at::Tensor fp8_quantize_meta(const at::Tensor input, const at::Tensor scale,
                             const at::ScalarType dest_dtype);

at::Tensor fp8_dequantize(const at::Tensor input, const at::Tensor scale_inv,
                          const at::ScalarType dest_dtype);

at::Tensor fp8_dequantize_meta(const at::Tensor input, const at::Tensor scale_inv,
                               const at::ScalarType dest_dtype);

/* GEMM */

at::Tensor hipblaslt_gemm(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B, at::Tensor scaleB_inv,
                          const at::ScalarType out_dtype, bool transA, bool transB, bool transC);

at::Tensor hipblaslt_gemm_meta(at::Tensor A, at::Tensor scaleA_inv, at::Tensor B,
                               at::Tensor scaleB_inv, const at::ScalarType out_dtype, bool transA,
                               bool transB, bool transC);

torch::Tensor gemm_fp8_blockwise(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                 torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                 const bool transB, const int64_t block_size);

torch::Tensor gemm_fp8_blockwise_meta(torch::Tensor &a, torch::Tensor &a_scales, torch::Tensor &b,
                                      torch::Tensor &b_scales, torch::Tensor &c, const bool transA,
                                      const bool transB, const int64_t block_size);

std::vector<torch::Tensor> fused_all_gather_matmul(
    const torch::Tensor &A_shard, const std::vector<torch::Tensor> &Bs, int64_t gather_dim,
    const std::string &group_name, std::optional<bool> return_A, std::optional<torch::Tensor> A_out,
    std::optional<std::vector<torch::Tensor>>  mm_outs,
    std::optional<std::vector<at::ScalarType>> out_dtypes, std::optional<std::string> comm_algo,
    std::optional<int> num_splits, std::optional<bool> skip_copy_local_ag_out,
    std::optional<bool> enable_sdma);

std::vector<torch::Tensor> fused_all_gather_scaled_matmul(
    const torch::Tensor &A_shard, const std::vector<torch::Tensor> &Bs,
    const torch::Tensor &A_scale, const std::vector<torch::Tensor> &B_scales, int64_t gather_dim,
    const std::string &group_name, const std::vector<bool> &use_fast_accum,
    const std::optional<std::vector<torch::Tensor>>  &biases,
    const std::optional<std::vector<torch::Tensor>>  &result_scales,
    const std::optional<std::vector<at::ScalarType>> &out_dtypes,
    std::optional<torch::Tensor> A_out, std::optional<std::vector<torch::Tensor>> mm_outs,
    std::optional<std::string> comm_algo, std::optional<int> num_splits,
    std::optional<bool> skip_copy_local_ag_out, std::optional<bool> enable_sdma);

} // namespace primus_turbo::pytorch
