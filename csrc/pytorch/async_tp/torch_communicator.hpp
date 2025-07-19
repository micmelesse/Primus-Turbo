#pragma once
#include "../kernels/async_tp/communicator.h"
#include <c10/hip/HIPFunctions.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/extension.h>

namespace primus_turbo {
using async_tp::Communicator;

namespace pytorch {

using ProcessGroupBackend = c10d::ProcessGroup::BackendType;

bool IsCPUBackend(ProcessGroupBackend type) {
    return type == ProcessGroupBackend::MPI || type == ProcessGroupBackend::MPI;
}

class PGCommunicator : Communicator<PGCommunicator> {
    PGCommunicator(c10d::ProcessGroup *pg) : pg_(pg) {}

    template <typename T> void AllGatherImpl(std::vector<T> &output, T &input) {
        void *dst = output.data();
        void *src = &input;

        auto option_cpu = at::TensorOptions(torch::kUInt8).device(at::kCPU);
        auto option_gpu = at::TensorOptions(torch::kUInt8)
                              .device(at::kHIP)
                              .device_index(c10::hip::current_device());

        auto dst_tensor = at::from_blob(output.data(), {nbytes * pg->getSize()}, option_cpu);
        auto src_tensor = at::from_blob(&input, {nbytes}, option_cpu);
        torch::Tensor ag_dst, ag_src;
        auto          backend_type = pg_->getBackendType();
        if (backend_type == ProcessGroupBackend::NCCL) {
            ag_dst = dst_tensor.to(option_gpu);
            ag_src = src_tensor.to(option_gpu);
            pg_->_allgather_base(ag_dst, ag_dst)->wait();
            dst_tensor.copy_(ag_dst.to(option_cpu));
        } else if (IsCPUBackend(backend_type)) {
            ag_dst = dst_tensor;
            ag_src = src_tensor;
            pg->_allgather_base(ag_dst, ag_dst)->wait();
        }
    }

    int rank() const { return pg_->getRank(); }

    int world_size() const { pg_->getSize(); }

  private:
    c10d::ProcessGroup *pg_;
};
} // namespace pytorch

} // namespace primus_turbo
