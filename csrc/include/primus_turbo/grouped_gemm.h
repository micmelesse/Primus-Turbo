#pragma once
#include "primus_turbo/dtype.h"
#include <cstdint>
#include <hip/hip_runtime.h>
#include "../kernels/grouped_gemm/grouped_gemm.hpp"
namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout>
void ck_grouped_gemm_kernel(const ADataType *a_ptr, // a_ptr b_ptr c_ptr from gpu src
                            const BDataType *b_ptr, CDataType *c_ptr,
                            const int *seg_lens, // seg_lens from gpu src
                            const int B, const int N, const int K, hipStream_t stream);

}
