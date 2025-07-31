#pragma once
#include "../kernels/grouped_gemm/grouped_gemm.hpp"
#include "ck_tile/core.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType = float>
void ck_grouped_gemm(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                     CDataType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
                     const bool transA, const bool transB, const ck_tile::index_t group_num,
                     const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
                     hipStream_t stream);

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout>
void ck_grouped_gemm_variable_k_kernel(ck_tile::GemmTransKernelArg *kargs_ptr,
                                       const ADataType *a_ptr, const BDataType *b_ptr,
                                       CDataType *c_ptr, const int64_t *p_seg_lens,
                                       ck_tile::index_t B, ck_tile::index_t M, ck_tile::index_t N,
                                       ck_tile::index_t stride_A, ck_tile::index_t stride_B,
                                       ck_tile::index_t stride_C, ck_tile::index_t k_batch,
                                       hipStream_t stream_id);

} // namespace primus_turbo
