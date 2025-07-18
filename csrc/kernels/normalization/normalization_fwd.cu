
#include "primus_turbo/common.h"
#include "primus_turbo/normalization.h"
#include "primus_turbo/reduce.cuh"

namespace primus_turbo {

// !!! This is a demo impl. It`s performance is bad.
// TODO: Opt
template <typename T>
__global__ void rmsnorm_fwd_kernel(const T *input, const T *gamma, T *output,
                                   const int64_t inner_len, const float epsilon) {
    const int32_t bid     = blockIdx.x;
    const int32_t warp_id = threadIdx.x / THREADS_PER_WARP; // in this case, the warp_id always == 0
    const int32_t lane_id = threadIdx.x % THREADS_PER_WARP;

    const T *input_ptr  = input + bid * inner_len;
    const T *gamma_ptr  = gamma;
    T       *output_ptr = output + bid * inner_len;

    float local_squares_sum = 0.0f;
    for (int64_t i = lane_id; i < inner_len; i += THREADS_PER_WARP) {
        float val = static_cast<float>(input_ptr[i]);
        local_squares_sum += (val * val);
    }

    float mean_square = warpReduceSum<float>(local_squares_sum);
    mean_square /= static_cast<float>(inner_len);
    const float norm_factor = rsqrtf(mean_square + epsilon);

    for (int64_t i = lane_id; i < inner_len; i += THREADS_PER_WARP) {
        float val =
            static_cast<float>(input_ptr[i]) * norm_factor * static_cast<float>(gamma_ptr[i]);
        output_ptr[i] = static_cast<T>(val);
    }
}

template <typename T>
void rmsnorm_fwd_impl(const T *input, const T *gamma, T *output, const int64_t inner_len,
                      const int64_t outer_len, const float epsilon, hipStream_t stream) {
    const dim3 block_dim(THREADS_PER_WARP, 1, 1);
    const dim3 grid_dim(outer_len, 1, 1);
    rmsnorm_fwd_kernel<T>
        <<<grid_dim, block_dim, 0, stream>>>(input, gamma, output, inner_len, epsilon);
}

template void rmsnorm_fwd_impl<float>(const float *input, const float *gamma, float *output,
                                      const int64_t inner_len, const int64_t outer_len,
                                      const float epsilon, hipStream_t stream);

} // namespace primus_turbo
