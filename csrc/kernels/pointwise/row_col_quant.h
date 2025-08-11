#include <string>

#include "ck_tile/core.hpp"

#include "ck_tile/core/numeric/half.hpp"
#include <hip/hip_runtime.h>
namespace primus_turbo {

template <typename InDataType, typename ScaleType, typename OutDataType>
__global__ void quant_dequant_2d_device(const InDataType *a_ptr, const ScaleType *scale,
                                        OutDataType *b_ptr, ck_tile::index_t M, ck_tile::index_t K,
                                        bool trans) {
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
void quant_dequant_2d(const InDataType *a_ptr, const ScaleType *scale, OutDataType *b_ptr,
                      ck_tile::index_t M, ck_tile::index_t K, bool trans, hipStream_t stream) {
    const ck_tile::index_t total_elements = M * K;

    // Thread block configuration for better GPU utilization
    const int threads_per_block = 256;   // Optimal for modern GPUs
    const int max_blocks        = 65535; // Maximum grid size
    const int blocks            = min(
        max_blocks, static_cast<int>((total_elements + threads_per_block - 1) / threads_per_block));

    quant_dequant_2d_device<InDataType, ScaleType, OutDataType>
        <<<blocks, threads_per_block, 0, stream>>>(a_ptr, scale, b_ptr, M, K, trans);
}
} // namespace primus_turbo
