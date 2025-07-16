#pragma once
#include "symm_mem.h"
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <vector>

namespace primus_turbo::async_tp {

// using GemmFunc = void(void*, int);
// template <typename T>
// void PipelinedAllGatherMatmul(const T *A_shard, const std::vector<T *> &Bs, int m, int k,
//                               const std::vector<int> &ns, int rank, int world_size,
//                               const std::vector<T *>         &comm_buffers,
//                               const std::vector<int *>       &barrier_buffers,
//                               const std::vector<hipStream_t> &comm_streams,
//                               const std::vector<hipStream_t> &copy_streams,
//                               const std::vector<hipStream_t> &gemm_streams);

template <typename GemmOp> struct AllGatherGemm {
    // is fp8 gemm op

    void Run(void **shards, int num_shards, void **weights, int num_weights, int m, int k, int *ns,
             int rank, int world_size, void **comm_bufs) {}
};

void PipelinedAllGatherMatmul(const void *A_shard, const std::vector<void *> &Bs, int m, int k,
                              const std::vector<int> &ns, int rank, int world_size,
                              const std::vector<void *>      &comm_buffers,
                              const std::vector<uint32_t *>  &barrier_buffers,
                              const std::vector<hipStream_t> &comm_streams,
                              const std::vector<hipStream_t> &copy_streams,
                              const std::vector<hipStream_t> &gemm_streams);

} // namespace primus_turbo::async_tp
