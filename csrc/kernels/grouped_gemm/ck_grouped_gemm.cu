// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "ck_tile/host/hip_check_error.hpp"

#include "ck_grouped_gemm_kernel.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

std::int64_t get_ck_grouped_gemm_args_sizes(const int group_num) {
    return group_num * sizeof(ck_tile::GemmTransKernelArg);
}

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

    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.as_ptr)[0] =
        static_cast<const void *>(a_ptr + group_offs_ptr[group_id] * k);
    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.bs_ptr)[0] =
        static_cast<const void *>(b_ptr + group_id * n * k);
    args_ptr[group_id].group_karg.e_ptr        = c_ptr + group_offs_ptr[group_id] * n;
    args_ptr[group_id].group_karg.M            = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.N            = n;
    args_ptr[group_id].group_karg.K            = k;
    args_ptr[group_id].group_karg.stride_As[0] = strideA;
    args_ptr[group_id].group_karg.stride_Bs[0] = strideB;
    args_ptr[group_id].group_karg.stride_E     = strideC;
    args_ptr[group_id].group_karg.k_batch      = k_batch;
}

template <typename IndexType>
__global__ void compute_group_offs_device(const IndexType       *group_lens_ptr,
                                          IndexType             *group_offs_ptr,
                                          const ck_tile::index_t group_num) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        group_offs_ptr[0] = 0;
    }

    if (idx < group_num) {
        // Compute cumulative sum for group offsets
        IndexType cumsum = 0;
        for (ck_tile::index_t i = 0; i < idx; i++) {
            cumsum += group_lens_ptr[i];
        }
        group_offs_ptr[idx + 1] = cumsum + group_lens_ptr[idx];
    }
}

template <typename IndexType>
void compute_group_offs(const IndexType *group_lens_ptr, IndexType *group_offs_ptr,
                        const ck_tile::index_t group_num, hipStream_t stream) {
    const ck_tile::index_t total_elements    = group_num;
    const int              threads_per_block = 256;
    const int              blocks =
        static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block);

    compute_group_offs_device<IndexType>
        <<<blocks, threads_per_block, 0, stream>>>(group_lens_ptr, group_offs_ptr, group_num);
}

// TODO: Refactor
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_grouped_gemm(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                     CDataType *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
                     const bool transA, const bool transB, const ck_tile::index_t group_num,
                     const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
                     hipStream_t stream, const uint32_t num_cu) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = transA ? m : k;
    const ck_tile::index_t strideB = transB ? k : n;
    const ck_tile::index_t strideC = n;

    // Setting args
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, group_num);
        const int blocks  = (group_num + threads - 1) / threads;
        compute_grouped_gemm_args<ADataType, BDataType, CDataType><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<ck_tile::GemmTransKernelArg *>(args_ptr), a_ptr, b_ptr, c_ptr,
            group_lens_ptr, group_offs_ptr, group_num, n, k, strideA, strideB, strideC, k_batch);
    }

    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner;
    using CLayout         = RowMajor;
    const auto stream_cfg = ck_tile::stream_config{stream};
    if (!transA && !transB) { // NN
        using ALayout = RowMajor;
        using BLayout = RowMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout>(group_num, m, n, k);
    } else if (!transA && transB) { // NT
        using ALayout = RowMajor;
        using BLayout = ColMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout>(group_num, m, n, k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm only support NN and NT");
    }
    runner->run(stream_cfg, group_num, args_ptr, num_cu);
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

    const int64_t strideAK = transA ? m : 1;
    const int64_t strideBK = transB ? 1 : n;
    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.as_ptr)[0] =
        static_cast<const void *>(a_ptr + group_offs_ptr[group_id] * strideAK);
    const_cast<std::array<const void *, 1> &>(args_ptr[group_id].group_karg.bs_ptr)[0] =
        static_cast<const void *>(b_ptr + group_offs_ptr[group_id] * strideBK);
    args_ptr[group_id].group_karg.e_ptr        = c_ptr + group_id * m * n;
    args_ptr[group_id].group_karg.M            = m;
    args_ptr[group_id].group_karg.N            = n;
    args_ptr[group_id].group_karg.K            = group_lens_ptr[group_id];
    args_ptr[group_id].group_karg.stride_As[0] = strideA;
    args_ptr[group_id].group_karg.stride_Bs[0] = strideB;
    args_ptr[group_id].group_karg.stride_E     = strideC;
    args_ptr[group_id].group_karg.k_batch      = k_batch;
}

/**
 * PostProcess: Set the non-computed parts in C to zero.
 */
template <typename T>
__global__ void
grouped_gemm_variable_k_postprocess(T *c_ptr, const int64_t *group_lens_ptr,
                                    const int64_t *group_offs_ptr, const ck_tile::index_t group_num,
                                    const ck_tile::index_t m, const ck_tile::index_t n) {
    const int group_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_id >= group_num)
        return;

    const int group_len = group_lens_ptr[group_id];
    const int group_off = group_offs_ptr[group_id];
    if (group_len > 0)
        return;

    c_ptr = c_ptr + group_id * m * n;
    memset(c_ptr, 0, sizeof(T) * m * n);
}

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void ck_grouped_gemm_variable_k(void *args_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
                                CDataType *c_ptr, const int64_t *group_lens_ptr,
                                const int64_t *group_offs_ptr, const bool transA, const bool transB,
                                const ck_tile::index_t group_num, const ck_tile::index_t m,
                                const ck_tile::index_t n, const ck_tile::index_t k,
                                hipStream_t stream, const uint32_t num_cu) {
    const ck_tile::index_t k_batch = 1;
    const bool             splitk  = k_batch > 1;

    const ck_tile::index_t strideA = transA ? m : k;
    const ck_tile::index_t strideB = transB ? k : n;
    const ck_tile::index_t strideC = n;

    // Setting args
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, group_num);
        const int grids   = (group_num + threads - 1) / threads;
        compute_grouped_gemm_variable_k_args<ADataType, BDataType, CDataType>
            <<<grids, threads, 0, stream>>>(
                reinterpret_cast<ck_tile::GemmTransKernelArg *>(args_ptr), a_ptr, b_ptr, c_ptr,
                group_lens_ptr, group_offs_ptr, transA, transB, group_num, m, n, strideA, strideB,
                strideC, k_batch);
    }

    using CLayout = RowMajor;
    std::unique_ptr<CKGroupedGemmRunnerInterFace> runner;
    const auto                                    stream_cfg = ck_tile::stream_config{stream};
    if (transA && !transB) { // TN
        using ALayout = ColMajor;
        using BLayout = RowMajor;
        runner = get_ck_grouped_gemm_instance<ADataType, BDataType, CDataType, AccDataType, ALayout,
                                              BLayout, CLayout>(group_num, m, n, k);
    } else {
        PRIMUS_TURBO_CHECK(false, "CKGroupedGemm-VariableK only support TN");
    }
    runner->run(stream_cfg, group_num, args_ptr, num_cu);

    // Postprocess
    {
        const int threads = std::min(MAX_THREADS_PER_BLOCK, group_num);
        const int grids   = (group_num + threads - 1) / threads;
        grouped_gemm_variable_k_postprocess<CDataType>
            <<<grids, threads, 0, stream>>>(c_ptr, group_lens_ptr, group_offs_ptr, group_num, m, n);
    }
}

template void ck_grouped_gemm<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::half_t *a_ptr, const ck_tile::half_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::bfloat16_t *a_ptr, const ck_tile::bfloat16_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::fp8_t *a_ptr, const ck_tile::fp8_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::bf8_t *a_ptr, const ck_tile::bf8_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);
template void ck_grouped_gemm<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::fp8_t *a_ptr, const ck_tile::fp8_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::bf8_t *a_ptr, const ck_tile::bf8_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm_variable_k<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::half_t *a_ptr, const ck_tile::half_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void
ck_grouped_gemm_variable_k<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::bfloat16_t *a_ptr, const ck_tile::bfloat16_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm_variable_k<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::fp8_t *a_ptr, const ck_tile::fp8_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm_variable_k<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::bfloat16_t>(
    void *args_ptr, const ck_tile::bf8_t *a_ptr, const ck_tile::bf8_t *b_ptr,
    ck_tile::bfloat16_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm_variable_k<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::fp8_t *a_ptr, const ck_tile::fp8_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void ck_grouped_gemm_variable_k<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(
    void *args_ptr, const ck_tile::bf8_t *a_ptr, const ck_tile::bf8_t *b_ptr,
    ck_tile::half_t *c_ptr, const int64_t *group_lens_ptr, const int64_t *group_offs_ptr,
    const bool transA, const bool transB, const ck_tile::index_t group_num,
    const ck_tile::index_t m, const ck_tile::index_t n, const ck_tile::index_t k,
    hipStream_t stream, const uint32_t num_cu);

template void compute_group_offs<int64_t>(const int64_t *group_lens_ptr, int64_t *group_offs_ptr,
                                          const ck_tile::index_t group_num, hipStream_t stream);
} // namespace primus_turbo
