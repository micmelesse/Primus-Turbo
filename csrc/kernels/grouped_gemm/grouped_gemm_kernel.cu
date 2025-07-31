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
__global__ void update_group_gemm_variable_k_kargs(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ADataType *a_ptr, const BDataType *b_ptr,
    CDataType *c_ptr, const int64_t *p_seg_lens, ck_tile::index_t B, ck_tile::index_t M,
    ck_tile::index_t N, ck_tile::index_t stride_A, ck_tile::index_t stride_B,
    ck_tile::index_t stride_C, ck_tile::index_t k_batch) {
    // Calculate the index for this thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if this thread has work to do
    if (index < B) {
        // Calculate pointers for this specific index
        const ADataType *cur_a = a_ptr;
        const BDataType *cur_b = b_ptr;
        CDataType       *cur_c = c_ptr;

        // Move pointers to the correct position for this index
        for (int i = 0; i < index; i++) {
            const int prev_K = p_seg_lens[i];
            cur_a += M * prev_K;
            cur_b += prev_K * N;
            cur_c += M * N;
        }

        const int K = p_seg_lens[index];

        // Update all fields in the group_karg structure
        kargs_ptr[index].group_karg.a_ptr    = cur_a;
        kargs_ptr[index].group_karg.b_ptr    = cur_b;
        kargs_ptr[index].group_karg.e_ptr    = cur_c; // e_ptr and c_ptr are union
        kargs_ptr[index].group_karg.M        = M;
        kargs_ptr[index].group_karg.N        = N;
        kargs_ptr[index].group_karg.K        = K;
        kargs_ptr[index].group_karg.stride_A = stride_A;
        kargs_ptr[index].group_karg.stride_B = stride_B;
        kargs_ptr[index].group_karg.stride_E = stride_C; // stride_E and stride_C are union
        kargs_ptr[index].group_karg.k_batch  = k_batch;
    }
}

template <typename ADataType, typename BDataType, typename CDataType, typename ALayout,
          typename BLayout, typename CLayout>
void ck_grouped_gemm_variable_k_kernel(ck_tile::GemmTransKernelArg *kargs_ptr,
                                       const ADataType *a_ptr, const BDataType *b_ptr,
                                       CDataType *c_ptr, const int64_t *p_seg_lens,
                                       ck_tile::index_t B, ck_tile::index_t M, ck_tile::index_t N,
                                       ck_tile::index_t stride_A, ck_tile::index_t stride_B,
                                       ck_tile::index_t stride_C, ck_tile::index_t k_batch,
                                       hipStream_t stream_id) {
    using AccDataType = float;

    dim3 blockDims(128);
    dim3 gridDims(4);
    update_group_gemm_variable_k_kargs<<<gridDims, blockDims, 0, stream_id>>>(
        kargs_ptr, a_ptr, b_ptr, c_ptr, p_seg_lens, B, M, N, stride_A, stride_B, stride_C, k_batch);
    const auto stream = ck_tile::stream_config{stream_id};
    // Execute the grouped GEMM operation
    grouped_gemm_tileloop<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout>(
        stream, B, kargs_ptr, false /* splitk */);
}

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template void
ck_grouped_gemm_variable_k_kernel<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, Col, Row, Row>(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ck_tile::half_t *a_ptr,
    const ck_tile::half_t *b_ptr, ck_tile::half_t *c_ptr, const int64_t *p_seg_lens,
    ck_tile::index_t B, ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t stride_A,
    ck_tile::index_t stride_B, ck_tile::index_t stride_C, ck_tile::index_t k_batch,
    hipStream_t stream_id);

template void ck_grouped_gemm_variable_k_kernel<ck_tile::bfloat16_t, ck_tile::bfloat16_t,
                                                ck_tile::bfloat16_t, Col, Row, Row>(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ck_tile::bfloat16_t *a_ptr,
    const ck_tile::bfloat16_t *b_ptr, ck_tile::bfloat16_t *c_ptr, const int64_t *p_seg_lens,
    ck_tile::index_t B, ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t stride_A,
    ck_tile::index_t stride_B, ck_tile::index_t stride_C, ck_tile::index_t k_batch,
    hipStream_t stream_id);

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

} // namespace primus_turbo
