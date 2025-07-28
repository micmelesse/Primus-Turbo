#pragma once
#include "all_gather_matml_impl.hpp"
#include "tensor.hpp"
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <optional>
#include <vector>

namespace primus_turbo::async_tp {

void InteralAllGatherMatmul(AllGatherProducer *ag_producer, GemmConsumer *gemm_consumer,
                            const _internal::HIPTensor                             &A_shard,
                            std::optional<_internal::HIPTensor>                     A_scale,
                            const std::vector<_internal::HIPTensor>                &weights,
                            const std::vector<std::optional<_internal::HIPTensor>> &biases,
                            _internal::HIPTensor                                   &A_out,
                            std::vector<_internal::HIPTensor>                      &mm_outs) {
    std::vector<_internal::HIPTensor> ag_needed_tensors;
    ag_needed_tensors.push_back(A_shard);
    if (A_scale.has_value()) {
        ag_needed_tensors.push_back(A_scale.value());
    }

    auto ag_iter = ag_producer->RunIter(ag_needed_tensors);
    for (const auto &worker : ag_iter) {
        worker->Run();
        auto ag_chunk_A = worker->Wait();
        auto res        = gemm_consumer->Run(ag_chunk_A);
        worker->PostProcess(res);
    }
}

} // namespace primus_turbo::async_tp
