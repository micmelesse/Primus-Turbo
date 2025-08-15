#include <string>

#include "ck_tile/core.hpp"

#include "ck_tile/core/numeric/half.hpp"
#include <hip/hip_runtime.h>
namespace primus_turbo {

template <typename InDataType, typename ScaleType, typename OutDataType>
__global__ void quant_2d_device(const InDataType *a_ptr, const ScaleType *scale, OutDataType *b_ptr,
                                ck_tile::index_t M, ck_tile::index_t K, bool trans) {
    const ck_tile::index_t total_elements = M * K;

    for (ck_tile::index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Calculate row and column indices
        ck_tile::index_t i, j; // i: row index, j: column index
        i = idx / K;           // Row index
        j = idx % K;           // Column index

        // Apply scaling based on transposition mode
        float scale_val;
        if (trans) {
            scale_val = scale[j]; // Per-column scaling when trans=true
        } else {
            scale_val = scale[i]; // Per-row scaling when trans=false
        }

        const InDataType a_val = a_ptr[idx];
        float            a_val_float =
            ck_tile::type_convert<float>(a_val) * ck_tile::type_convert<float>(scale_val);
        b_ptr[idx] = ck_tile::type_convert<OutDataType>(a_val_float);
    }
}

template <typename InDataType, typename ScaleType, typename OutDataType>
__global__ void quant_3d_trans_device(const InDataType *a_ptr, const ScaleType *scale,
                                      OutDataType *b_ptr, ck_tile::index_t N, ck_tile::index_t M,
                                      ck_tile::index_t K) {
    const ck_tile::index_t total_elements = N * M * K;

    for (ck_tile::index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Calculate indices for 3D tensor
        ck_tile::index_t n, m, k; // n: batch index, m: row index, k: column index
        n = idx / (M * K);        // Batch index
        m = (idx % (M * K)) / K;  // Row index
        k = idx % K;              // Column index

        // Apply scaling based on transposition mode
        float scale_val;
        scale_val              = scale[n * K + k]; // Per-column scaling within each batch
        const InDataType a_val = a_ptr[idx];
        float            a_val_float =
            ck_tile::type_convert<float>(a_val) * ck_tile::type_convert<float>(scale_val);
        b_ptr[idx] = ck_tile::type_convert<OutDataType>(a_val_float);
    }
}

template <typename InDataType, typename ScaleType, typename OutDataType>
void quant_2d(const InDataType *a_ptr, const ScaleType *scale, OutDataType *b_ptr,
              ck_tile::index_t M, ck_tile::index_t K, bool trans, hipStream_t stream) {
    const ck_tile::index_t total_elements = M * K;

    // Thread block configuration for better GPU utilization
    const int threads_per_block = 256;
    const int max_blocks        = 65535; // Maximum grid size
    const int blocks            = min(
        max_blocks, static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block));

    quant_2d_device<InDataType, ScaleType, OutDataType>
        <<<blocks, threads_per_block, 0, stream>>>(a_ptr, scale, b_ptr, M, K, trans);
}

template <typename InDataType, typename ScaleType, typename OutDataType>
void quant_3d(const InDataType *a_ptr, const ScaleType *scale, OutDataType *b_ptr,
              ck_tile::index_t N, ck_tile::index_t M, ck_tile::index_t K, bool trans,
              hipStream_t stream) {
    const ck_tile::index_t total_elements = N * M * K;

    // Thread block configuration for better GPU utilization
    const int threads_per_block = 256;
    const int max_blocks        = 65535; // Maximum grid size
    const int blocks            = min(
        max_blocks, static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block));
    if (trans) {
        quant_3d_trans_device<InDataType, ScaleType, OutDataType>
            <<<blocks, threads_per_block, 0, stream>>>(a_ptr, scale, b_ptr, N, M, K);
    } else {
        quant_2d_device<InDataType, ScaleType, OutDataType>
            <<<blocks, threads_per_block, 0, stream>>>(a_ptr, scale, b_ptr, N * M, K, trans);
    }
}

template <typename InDataType, typename ScaleType, typename OutDataType>
__global__ void dequant_grouped_gemm_device(const InDataType *a_ptr, const ScaleType *scale_a,
                                            const ScaleType *scale_b, OutDataType *b_ptr,
                                            const int64_t *p_group_lens,
                                            const int64_t *p_group_offs, ck_tile::index_t B,
                                            ck_tile::index_t N) {
    // Calculate total number of elements across all groups
    ck_tile::index_t total_elements = p_group_offs[B] * N;

    // Use all threads for parallel processing with grid-stride loop
    for (ck_tile::index_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_elements; global_idx += blockDim.x * gridDim.x) {

        // Find which group and which element within that group
        ck_tile::index_t remaining_idx  = global_idx;
        ck_tile::index_t k              = 0;
        ck_tile::index_t offset_a       = 0;
        ck_tile::index_t offset_b       = 0;
        ck_tile::index_t offset_scale_b = 0;
        ck_tile::index_t offset_scale_a = 0;

        // Locate the group this element belongs to
        while (k < B) {
            ck_tile::index_t group_size = p_group_lens[k] * N;
            if (remaining_idx < group_size) {
                break;
            }
            remaining_idx -= group_size;
            offset_a += group_size;
            offset_b += group_size;
            offset_scale_b += N;
            offset_scale_a += p_group_lens[k]; // Accumulate rows for scale_a offset
            k++;
        }

        // Calculate position within the group
        ck_tile::index_t i = remaining_idx / N; // Row within group
        ck_tile::index_t j = remaining_idx % N; // Column

        // Calculate actual pointers for this element
        const InDataType *group_a_ptr   = a_ptr + offset_a;
        OutDataType      *group_b_ptr   = b_ptr + offset_b;
        const ScaleType  *group_scale_b = scale_b + offset_scale_b;
        const ScaleType  *group_scale_a = scale_a + offset_scale_a + i; // Per-row scale_a

        // Perform dequantization
        const InDataType a_val       = group_a_ptr[i * N + j];
        ScaleType        scale_a_val = *group_scale_a;   // Dereference the pointer
        ScaleType        scale_b_val = group_scale_b[j]; // Per-column scale_b

        float a_val_float = ck_tile::type_convert<float>(a_val) *
                            ck_tile::type_convert<float>(scale_a_val) *
                            ck_tile::type_convert<float>(scale_b_val);
        group_b_ptr[i * N + j] = ck_tile::type_convert<OutDataType>(a_val_float);
    }
}

template <typename InDataType, typename ScaleType, typename OutDataType>
__global__ void
dequant_grouped_gemm_variable_k_device(const InDataType *input_ptr, const ScaleType *scale_a,
                                       const ScaleType *scale_b, OutDataType *output_ptr,
                                       ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K) {
    // Calculate total number of elements
    ck_tile::index_t total_elements = B * N * K;

    // Use all threads for parallel processing with grid-stride loop
    for (ck_tile::index_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Calculate 3D indices
        ck_tile::index_t b = idx / (N * K);       // Batch index
        ck_tile::index_t n = (idx % (N * K)) / K; // Row index
        ck_tile::index_t k = idx % K;             // Column index

        // Get the input value
        const InDataType input_val = input_ptr[idx];

        // Get scale values
        // scale_a has shape [1, N], so we index it as scale_a[n]
        ScaleType scale_a_val = scale_a[n];

        // scale_b has shape [1, K], so we index it as scale_b[k]
        ScaleType scale_b_val = scale_b[k];

        // Perform dequantization
        float result = ck_tile::type_convert<float>(input_val) *
                       ck_tile::type_convert<float>(scale_a_val) *
                       ck_tile::type_convert<float>(scale_b_val);

        // Store the result
        output_ptr[idx] = ck_tile::type_convert<OutDataType>(result);
    }
}

// a must be row-wise and b must be column-wise
template <typename InDataType, typename ScaleType, typename OutDataType>
void dequant_grouped_gemm(const InDataType *a_ptr, const ScaleType *scale_a,
                          const ScaleType *scale_b, OutDataType *b_ptr, const int64_t *p_group_lens,
                          const int64_t *p_group_offs, ck_tile::index_t B, ck_tile::index_t M,
                          ck_tile::index_t N, hipStream_t stream) {
    const ck_tile::index_t total_elements = B * M * N;

    // Thread block configuration for better GPU utilization
    const int threads_per_block = 256;
    const int max_blocks        = 65535; // Maximum grid size
    const int blocks            = min(
        max_blocks, static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block));
    dequant_grouped_gemm_device<InDataType, ScaleType, OutDataType>
        <<<blocks, threads_per_block, 0, stream>>>(a_ptr, scale_a, scale_b, b_ptr, p_group_lens,
                                                   p_group_offs, B, N);
}

template <typename InDataType, typename ScaleType, typename OutDataType>
void dequant_grouped_gemm_variable_k(const InDataType *input_ptr, const ScaleType *scale_a,
                                     const ScaleType *scale_b, OutDataType *output_ptr,
                                     ck_tile::index_t B, ck_tile::index_t N, ck_tile::index_t K,
                                     hipStream_t stream) {
    const ck_tile::index_t total_elements = B * N * K;

    // Thread block configuration for better GPU utilization
    const int threads_per_block = 256;
    const int max_blocks        = 65535; // Maximum grid size
    const int blocks            = min(
        max_blocks, static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block));

    dequant_grouped_gemm_variable_k_device<InDataType, ScaleType, OutDataType>
        <<<blocks, threads_per_block, 0, stream>>>(input_ptr, scale_a, scale_b, output_ptr, B, N,
                                                   K);
}

} // namespace primus_turbo
