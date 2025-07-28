#pragma once
#include "../kernels/async_tp/communicator.h"
#include "primus_turbo/macros.h"
#include <c10/hip/HIPFunctions.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/extension.h>

namespace primus_turbo {
using async_tp::Communicator;

namespace pytorch {

using ProcessGroupBackend = c10d::ProcessGroup::BackendType;

class PGCommunicator : public Communicator {
  public:
    PGCommunicator(c10d::ProcessGroup *pg) : pg_(pg) {}

    void AllGather(void *dst, void *src, size_t len) override {
        auto option_cpu = at::TensorOptions(torch::kUInt8).device(at::kCPU);
        auto option_gpu = at::TensorOptions(torch::kUInt8)
                              .device(at::kHIP)
                              .device_index(c10::hip::current_device());

        auto dst_tensor = at::from_blob(dst, {static_cast<int>(len) * pg_->getSize()}, option_cpu);
        auto src_tensor = at::from_blob(src, {static_cast<int>(len)}, option_cpu);
        torch::Tensor ag_dst, ag_src;
        auto          backend_type = pg_->getBackendType();
        if (backend_type == ProcessGroupBackend::NCCL) {
            ag_dst = dst_tensor.to(option_gpu);
            ag_src = src_tensor.to(option_gpu);
            pg_->_allgather_base(ag_dst, ag_dst)->wait();
            dst_tensor.copy_(ag_dst.to(option_cpu));
        } else if (IsCPUBackend()) {
            ag_dst = dst_tensor;
            ag_src = src_tensor;
            pg_->_allgather_base(ag_dst, ag_dst)->wait();
        } else {
            PRIMUS_TURBO_CHECK(false, "not support backend type " + pg_->getBackendName());
        }
    }

    int rank() const override { return pg_->getRank(); }

    int world_size() const override { return pg_->getSize(); }

  private:
    bool IsCPUBackend() {
        return pg_->getBackendType() == ProcessGroupBackend::MPI ||
               pg_->getBackendType() == ProcessGroupBackend::GLOO;
    }

  private:
    c10d::ProcessGroup *pg_;
};
} // namespace pytorch

} // namespace primus_turbo
