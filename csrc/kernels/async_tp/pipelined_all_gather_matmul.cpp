#include "all_gather_matmul.h"
#include "helper.hpp"
#include "primus_turbo/macros.h"
#include <optional>

namespace primus_turbo::async_tp {

void Barrier(uint32_t **barrier, int rank, int world_size, hipStream_t stream) {
    barrier_kernel<<<1, 64, 0, stream>>>(barrier, 0, rank, world_size, 0);
}

void PipelinedCopySendWithSignals(std::vector<std::vector<void *>> &dst_ptrs,
                                  std::vector<void *> src_ptrs, std::vector<size_t> &sizes,
                                  std::vector<uint32_t *> signal_pads, int rank, int world_size,
                                  hipMemcpyKind cp_type, std::vector<hipStream_t> &comm_streams,
                                  hipEvent_t comm_event) {
    int dst_offset = -1;
    int num_stream = comm_streams.size();
    for (int dst_rank = 0, idx = 0; dst_rank < world_size; ++dst_rank) {
        if (dst_rank == rank)
            continue;
        dst_offset = rank < dst_rank ? rank : rank - 1;

        for (size_t i = 0; i < sizes.size(); ++i) {
            PRIMUS_TURBO_CHECK_HIP(
                hipMemcpyAsync((char *) dst_ptrs[dst_rank][i] + dst_offset * sizes[i], src_ptrs[i],
                               sizes[i], cp_type, comm_streams[idx % num_stream]));
        }

        // PRIMUS_TURBO_CHECK_HIP(
        //     hipMemcpyAsync(
        //         signal_pads[dst_rank] + rank * 4,
        //         one.data_ptr(),
        //         one.nbytes,
        //         comm_kind_type,
        //         ag_stream.cuda_stream,
        //     )
        // )

        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(comm_event, comm_streams[idx % num_stream]));
        ++idx;
    }
}

auto hipDataTypeToSize(hipDataType type) {
    switch (type) {
    case hipDataType::HIP_C_8I:
        return 8;

    default:
        return 0;
    }
    return -1;
}

template <typename GemmFunc, typename Communicator>
void PipelinedAllGatherGEMM(void *A_shard, hipDataType A_shard_type, std::optional<void *> A_scale,
                            std::optional<hipDataType> A_scale_type, std::vector<void *> &weights,
                            std::vector<hipDataType> &weight_types, int m, int k,
                            std::vector<int> &ns, int num_splits, bool ag_out_needed,
                            std::vector<hipStream_t> &comm_streams, hipEvent_t comm_event,
                            Communicator *comm, GemmFunc gemm_func, ChunkAllGather ag_func, ) {

    auto   A_shard_stride     = m * k * hipDataTypeToSize(A_shard_type);
    auto   A_shard_buf_stride = A_shard_stride * (world_size - 1);
    size_t request_buf_size   = A_shard_buf_stride;
    if (A_scale.has_value()) {
        request_buf_size += m * hipDataTypeToSize(A_scale_type.value()) * (world_size - 1);
    }

    auto symm_mem = SymmetricMemoryManager::Instance().GetSymmMem(request_buf_size, comm);

    int  rank             = symm_mem->rank();
    int  world_size       = symm_mem->world_size();
    auto comm_buffer_ptrs = symm_mem->buffer_ptrs();
    auto signal_pads      = symm_mem->signal_pad_ptrs();

    std::vector<std::vector<void *>> shard_bufs(world_size);
    std::vector<std::vector<void *>> local_shard_buf_chunk(num_splits);

    for (int i = 0; i < world_size; ++i) {
        shard_bufs[i].push_back(comm_buffer_ptrs[i]);

        if (A_scale.has_value()) {
            shard_bufs[i].push_back((char *) comm_buffer_ptrs[i] + A_shard_buf_stride);
        }

        if (i == rank) {
            auto base_buf_ptr = comm_buffer_ptrs[i];
            for (int j = 0; j < num_splits; ++j) {
                local_shard_buf_chunk[j].push_back(base_buf_ptr);
                if (A_scale.has_value) {
                    local_shard_buf_chunk[j].push_back((base_buf_ptr) + A_shard_buf_stride);
                }

                base_buf_ptr += j * A_shard_buf_stride / num_splits;
            }
        }
    }

    std::vector<void *> input_ptrs;
    std::vector<int>    send_sizes;
    input_ptrs.push_back(A_shard);
    send_sizes.push_back(A_shard_stride);
    if (A_scale.has_value()) {
        input_ptrs.push_back(A_scale.value());
        send_sizes.push_back(m * hipDataTypeToSize(A_scale_type.value()));
    }

    for (int step = 0; step < num_splits; ++step) {
        PipelinedCopySendWithSignals(shard_bufs, input_ptrs, send_sizes, signal_pads, rank,
                                     world_size, hipMemcpyKind::hipMemcpyDeviceToDevice,
                                     comm_streams, comm_event);
        auto tmp_outputs = gemm_func(local_shard_buf_chunk, weights);
    }
}

} // namespace primus_turbo::async_tp
