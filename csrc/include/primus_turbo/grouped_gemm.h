#pragma once
#include "../kernels/grouped_gemm/grouped_gemm.hpp"
#include "primus_turbo/dtype.h"
#include <cstdint>
#include <hip/hip_runtime.h>
namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType>
void ck_grouped_gemm_kernel(const ADataType *p_a, // p_a p_b p_c from gpu src
                            const BDataType *p_b, CDataType *p_c,
                            const int *p_seg_lens, // p_seg_lens from gpu src
                            const int B, const int N, const int K);
// template <typename ADataType, typename BDataType, typename CDataType>
// void ck_grouped_gemm_kernel(const ck_tile::half_t *p_a, // p_a p_b p_c from gpu src
//                             const ck_tile::half_t *p_b, ck_tile::half_t *p_c,
//                             const int *p_seg_lens, // p_seg_lens from gpu src
//                             const int B, const int N, const int K);
void ck_test();
} // namespace primus_turbo
