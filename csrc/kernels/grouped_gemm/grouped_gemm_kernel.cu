#include "ck_tile/host/hip_check_error.hpp"
#include "grouped_gemm.hpp"
#include "primus_turbo/grouped_gemm.h"
#include <hip/hip_runtime.h>

namespace primus_turbo {
template <typename Layout> static constexpr inline auto is_row_major(Layout layout_) {
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename CDataType>
__global__ void update_group_gemm_kargs(ck_tile::GemmTransKernelArg *kargs_ptr,
                                        const ADataType *a_ptr, const BDataType *b_ptr,
                                        CDataType *c_ptr, const int64_t *p_seg_lens,
                                        ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K,
                                        ck_tile::index_t stride_A, ck_tile::index_t stride_B,
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
            const int prev_M = p_seg_lens[i];
            cur_a += prev_M * K;
            cur_b += K * N;
            cur_c += prev_M * N;
        }

        const int M = p_seg_lens[index];

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
void ck_grouped_gemm_kernel(ck_tile::GemmTransKernelArg *kargs_ptr, const ADataType *a_ptr,
                            const BDataType *b_ptr, CDataType *c_ptr, const int64_t *p_seg_lens,
                            ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K,
                            ck_tile::index_t stride_A, ck_tile::index_t stride_B,
                            ck_tile::index_t stride_C, ck_tile::index_t k_batch,
                            hipStream_t stream_id) {
    using AccDataType = float;

    dim3 blockDims(128);
    dim3 gridDims(4);
    update_group_gemm_kargs<<<gridDims, blockDims, 0, stream_id>>>(
        kargs_ptr, a_ptr, b_ptr, c_ptr, p_seg_lens, B, N, K, stride_A, stride_B, stride_C, k_batch);
    const auto stream = ck_tile::stream_config{stream_id};
    // Execute the grouped GEMM operation
    grouped_gemm_tileloop<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout>(
        stream, B, kargs_ptr, false /* splitk */);
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
ck_grouped_gemm_kernel<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, Row, Col, Row>(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ck_tile::half_t *a_ptr,
    const ck_tile::half_t *b_ptr, ck_tile::half_t *c_ptr, const int64_t *p_seg_lens,
    ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K, ck_tile::index_t stride_A,
    ck_tile::index_t stride_B, ck_tile::index_t stride_C, ck_tile::index_t k_batch,
    hipStream_t stream_id);

template void ck_grouped_gemm_kernel<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t,
                                     Row, Col, Row>(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ck_tile::bfloat16_t *a_ptr,
    const ck_tile::bfloat16_t *b_ptr, ck_tile::bfloat16_t *c_ptr, const int64_t *p_seg_lens,
    ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K, ck_tile::index_t stride_A,
    ck_tile::index_t stride_B, ck_tile::index_t stride_C, ck_tile::index_t k_batch,
    hipStream_t stream_id);

template void
ck_grouped_gemm_kernel<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, Row, Row, Row>(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ck_tile::half_t *a_ptr,
    const ck_tile::half_t *b_ptr, ck_tile::half_t *c_ptr, const int64_t *p_seg_lens,
    ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K, ck_tile::index_t stride_A,
    ck_tile::index_t stride_B, ck_tile::index_t stride_C, ck_tile::index_t k_batch,
    hipStream_t stream_id);

template void ck_grouped_gemm_kernel<ck_tile::bfloat16_t, ck_tile::bfloat16_t, ck_tile::bfloat16_t,
                                     Row, Row, Row>(
    ck_tile::GemmTransKernelArg *kargs_ptr, const ck_tile::bfloat16_t *a_ptr,
    const ck_tile::bfloat16_t *b_ptr, ck_tile::bfloat16_t *c_ptr, const int64_t *p_seg_lens,
    ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K, ck_tile::index_t stride_A,
    ck_tile::index_t stride_B, ck_tile::index_t stride_C, ck_tile::index_t k_batch,
    hipStream_t stream_id);

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
} // namespace primus_turbo
