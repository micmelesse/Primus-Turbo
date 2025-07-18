#pragma once

#include "primus_turbo/common.h"

#define FINAL_MASK 0xffffffffffffffffULL

namespace primus_turbo {

// TODO: Refactor
template<typename T>
PRIMUS_TURBO_DEVICE T warpReduceSum(T val) {
#pragma unroll
    for (int offset = THREADS_PER_WARP >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, offset);
    }
    return val;
}

} // namespace primus_turbo
