#include "ck_tile/host/hip_check_error.hpp"
#include "grouped_gemm.hpp"
#include "primus_turbo/grouped_gemm.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {

template <typename ADataType, typename BDataType, typename CDataType>
__global__ void
compute_grouped_gemm_args(ck_tile::GemmTransKernelArg *args_ptr, const ADataType *a_ptr,
                          const BDataType *b_ptr, CDataType *c_ptr, const int64_t *group_lens_ptr,
                          const int64_t *group_offs_ptr, const ck_tile::index_t group_num,
                          const ck_tile::index_t n, const ck_tile::index_t k,
                          const ck_tile::index_t strideA, const ck_tile::index_t strideB,
                          const ck_tile::index_t strideC, const ck_tile::index_t k_batch) {
    const int64_t group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    args_ptr[group_id].group_karg.a_ptr    = a_ptr + group_offs_ptr[group_id] * k;
    args_ptr[group_id].group_karg.b_ptr    = b_ptr + group_id * n * k;
    args_ptr[group_id].group_karg.e_ptr    = c_ptr + group_offs_ptr[group_id] * n;
    args_ptr[group_id].group_karg.M        = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.N        = n;
    args_ptr[group_id].group_karg.K        = k;
    args_ptr[group_id].group_karg.stride_A = strideA;
    args_ptr[group_id].group_karg.stride_B = strideB;
    args_ptr[group_id].group_karg.stride_E = strideC;
    args_ptr[group_id].group_karg.k_batch  = k_batch;
}

// TODO: Refactor
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_grouped_gemm(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                     CDataType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
                     const bool transA, const bool transB, const ck_tile::index_t group_num,
                     const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
                     hipStream_t stream) {
    // TODO: k_batch control splitk
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = transA ? m : k;
    const ck_tile::index_t strideB = transB ? k : n;
    const ck_tile::index_t strideC = n;

    // Setting args
    {
        const int threads = std::min(1024, group_num);
        const int blocks  = (group_num + threads - 1) / threads;
        compute_grouped_gemm_args<ADataType, BDataType, CDataType><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<ck_tile::GemmTransKernelArg *>(args_ptr), a_ptr, b_ptr, c_ptr,
            group_lens_ptr, group_offs_ptr, group_num, n, k, strideA, strideB, strideC, k_batch);
    }

    using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
    using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout  = RowMajor;

    const auto stream_cfg = ck_tile::stream_config{stream};
    if (!transA && !transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        grouped_gemm_tileloop<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout,
                              CLayout>(stream_cfg, group_num, args_ptr, splitk);
    } else if (!transA && transB) { // NT
        using ALayout = RowMajor;
        using BLayout = ColMajor;
        grouped_gemm_tileloop<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout,
                              CLayout>(stream_cfg, group_num, args_ptr, splitk);
    } else {
        // TODO: more layout
    }
}

template <typename ADataType, typename BDataType, typename CDataType>
__global__ void compute_grouped_gemm_variable_k_args(
    ck_tile::GemmTransKernelArg *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
    CDataType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t strideA,
    const ck_tile::index_t strideB, const ck_tile::index_t strideC,
    const ck_tile::index_t k_batch) {
    const int64_t group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    const int64_t strideAK                 = transA ? m : 1;
    const int64_t strideBK                 = transB ? 1 : n;
    args_ptr[group_id].group_karg.a_ptr    = a_ptr + group_offs_ptr[group_id] * strideAK;
    args_ptr[group_id].group_karg.b_ptr    = b_ptr + group_offs_ptr[group_id] * strideBK;
    args_ptr[group_id].group_karg.e_ptr    = c_ptr + group_id * m * n;
    args_ptr[group_id].group_karg.M        = m;
    args_ptr[group_id].group_karg.N        = n;
    args_ptr[group_id].group_karg.K        = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.stride_A = strideA;
    args_ptr[group_id].group_karg.stride_B = strideB;
    args_ptr[group_id].group_karg.stride_E = strideC;
    args_ptr[group_id].group_karg.k_batch  = k_batch;
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_grouped_gemm_variable_k(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                                CDataType *c_ptr, const int64_t *group_lens_ptr,
                                const int64_t *group_offs_ptr, const bool transA, const bool transB,
                                const ck_tile::index_t group_num, const ck_tile::index_t m,
                                const ck_tile::index_t n, const ck_tile::index_t k,
                                hipStream_t stream) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = transA ? m : k;
    const ck_tile::index_t strideB = transB ? k : n;
    const ck_tile::index_t strideC = n;

    // Setting args
    {
        const int threads = std::min(1024, group_num);
        const int blocks  = (group_num + threads - 1) / threads;
        compute_grouped_gemm_variable_k_args<ADataType, BDataType, CDataType>
            <<<blocks, threads, 0, stream>>>(
                reinterpret_cast<ck_tile::GemmTransKernelArg *>(args_ptr), a_ptr, b_ptr, c_ptr,
                group_lens_ptr, group_offs_ptr, transA, transB, group_num, m, n, strideA, strideB,
                strideC, k_batch);
    }

    using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
    using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout  = RowMajor;

    const auto stream_cfg = ck_tile::stream_config{stream};
    if (transA && !transB) { // TN
        using ALayout = ColMajor;
        using BLayout = RowMajor;
        grouped_gemm_tileloop<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout,
                              CLayout>(stream_cfg, group_num, args_ptr, splitk);
    } else {
        // TOOD:
    }

    // TODO: process zero case
}

template void ck_grouped_gemm<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::half_t *a_ptr, const ck_tile::half_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream);

template void ck_grouped_gemm<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::bfloat16_t *a_ptr, const ck_tile::bfloat16_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream);

template void ck_grouped_gemm_variable_k<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::half_t *a_ptr, const ck_tile::half_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream);

template void
ck_grouped_gemm_variable_k<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::bfloat16_t *a_ptr, const ck_tile::bfloat16_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream);

} // namespace primus_turbo
