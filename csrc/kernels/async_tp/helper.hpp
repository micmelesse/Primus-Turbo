#pragma once
#include <atomic>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

constexpr size_t kNSPerMS = 1e6;

template <std::memory_order Sem>
__device__ __forceinline__ uint32_t cas(uint32_t *addr, uint32_t compare, uint32_t val) {
    __atomic_compare_exchange_n(addr, &compare, val, false, static_cast<int>(Sem),
                                __ATOMIC_RELAXED);
    return compare;
}

__device__ __forceinline__ size_t global_timer_ns() {
    static constexpr double MI300_FREQ_GHZ = 2.1;
    return clock64() / MI300_FREQ_GHZ;
}

template <std::memory_order Sem>
__device__ __forceinline__ bool try_put_signal(uint32_t *addr, size_t timeout_ms) {
    size_t deadline = global_timer_ns() + timeout_ms * kNSPerMS;
    while (cas<Sem>(addr, 0, 1) != 0) {
        if (timeout_ms != 0 && global_timer_ns() > deadline) {
            return false;
        }
    }
    return true;
}

template <std::memory_order Sem>
__device__ __forceinline__ bool try_wait_signal(uint32_t *addr, size_t timeout_ms) {
    size_t deadline = global_timer_ns() + timeout_ms * kNSPerMS;
    while (cas<Sem>(addr, 1, 0) != 1) {
        if (timeout_ms != 0 && global_timer_ns() > deadline) {
            return false;
        }
    }
    return true;
}

template <std::memory_order Sem> __device__ __forceinline__ void wait_signal(uint32_t *addr) {
    while (cas<Sem>(addr, 1, 0) != 1)
        ;
}

__global__ void barrier_kernel(uint32_t **signal_pads, int channel, int rank, int world_size,
                               size_t timeout_ms) {
    if (threadIdx.x < world_size) {
        auto target_rank = threadIdx.x;
        if (target_rank == rank) {
            return;
        }
        auto put_success = try_put_signal<std::memory_order_release>(
            signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
        if (!put_success) {
            printf("[FATAL] CUDASymmetricMemory::barrier: rank %d failed to send signal "
                   "to rank %d on channel %d after %lu microseconds\n",
                   rank, target_rank, channel, timeout_ms);
            abort();
        }
        auto wait_success = try_wait_signal<std::memory_order_acquire>(
            signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
        if (!wait_success) {
            printf("[FATAL] CUDASymmetricMemory::barrier: rank %d failed to receive signal "
                   "from rank %d on channel %d after %lu microseconds\n",
                   rank, target_rank, channel, timeout_ms);
            abort();
        }
    }
}
