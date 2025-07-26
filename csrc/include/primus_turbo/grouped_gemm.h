#pragma once
#include "../kernels/grouped_gemm/grouped_gemm.hpp"
#include "ck_tile/core.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
namespace primus_turbo {

void *ck_grouped_gemm_init(const int B, hipStream_t stream_id);

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout>
void ck_grouped_gemm_kernel(ck_tile::GemmTransKernelArg *kargs_ptr, const ADataType *a_ptr,
                            const BDataType *b_ptr, CDataType *c_ptr, const int *p_seg_lens,
                            ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K,
                            ck_tile::index_t stride_A, ck_tile::index_t stride_B,
                            ck_tile::index_t stride_C, ck_tile::index_t k_batch,
                            hipStream_t stream_id);
} // namespace primus_turbo
