#pragma once
#include "cached_utils.hpp"
#include "communicator.h"
#include "primus_turbo/macros.h"
#include "symm_mem.h"
#include "tensor.hpp"
#include <optional>

namespace primus_turbo::async_tp {

class Worker {
  public:
    virtual std::vector<_internal::HIPTensor> Wait()                           = 0;
    virtual void                              Run() const                      = 0;
    virtual void PostProcess(std::optional<std::vector<_internal::HIPTensor>>) = 0;
};

class AllGatherProducer {
  public:
    virtual std::vector<std::unique_ptr<Worker>>
    RunIter(const std::vector<_internal::HIPTensor> &ag_needed_tensors) = 0;
};

class PipelinedAllGatherProducer : public AllGatherProducer {

    class SendWorker : public Worker {
      public:
        SendWorker(std::vector<std::vector<void *>> &dst_rank_ptrs,
                   const std::vector<void *> &src_ptrs, const std::vector<int> &sizes, int chunk_id,
                   const std::vector<hipStream_t> &comm_streams, hipEvent_t comm_event)
            : chunk_id_(chunk_id), dst_rank_ptrs_(dst_rank_ptrs), src_ptrs_(src_ptrs),
              sizes_(sizes), comm_stream_(comm_streams), comm_event_(comm_event) {}

        std::vector<_internal::HIPTensor> Wait() override {}

        void Run() const override {
            RunKernelImpl();
            RunHIPImpl();
        }

        void RunHIPImpl() const {
            int num_ag  = src_ptrs_.size();
            int num_dst = dst_rank_ptrs_.size();
            for (int j = 0; j < num_dst; ++j) {
                for (int i = 0; i < num_ag; ++i) {
                    PRIMUS_TURBO_CHECK_HIP(hipMemcpyAsync(
                        dst_rank_ptrs_[j][i], (char *) src_ptrs_[i] + chunk_id_ * sizes_[i],
                        sizes_[i], hipMemcpyDeviceToDevice, comm_stream_[j]));
                }
            }
        }

        void RunKernelImpl() const { PRIMUS_TURBO_CHECK(false, "not impl!"); }

        void PostProcess(std::optional<std::vector<_internal::HIPTensor>>) override {}

      private:
        int                               chunk_id_;
        std::vector<std::vector<void *>> &dst_rank_ptrs_;
        const std::vector<void *>        &src_ptrs_;
        const std::vector<int>           &sizes_;
        const std::vector<hipStream_t>   &comm_stream_;
        hipEvent_t                        comm_event_;
    };

  public:
    PipelinedAllGatherProducer(Communicator *comm, int num_splits, int num_stream,
                               int stream_priority = 0, bool use_sdma = false)
        : comm_(comm), num_splits_(num_splits), use_sdma_(use_sdma), comm_streams_(num_stream) {
        for (int i = 0; i < num_stream; ++i) {
            PRIMUS_TURBO_CHECK_HIP(
                hipStreamCreateWithPriority(&comm_streams_[i], 0, stream_priority));
        }

        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&comm_event_, 0));
    }

    ~PipelinedAllGatherProducer() {
        for (int i = 0; i < comm_streams_.size(); ++i) {
            hipStreamDestroy(comm_streams_[i]);
        }
        hipEventDestroy(comm_event_);
    }

  public:
    std::vector<std::unique_ptr<Worker>>
    RunIter(const std::vector<_internal::HIPTensor> &ag_needed_tensors) override {
        std::vector<std::unique_ptr<Worker>> workers(num_splits_);
        int                                  p2p_workspace_size_req = 0;
        for (const auto &t : ag_needed_tensors) {
            p2p_workspace_size_req += t.nbytes() * (comm_->world_size() - 1);
        }
        std::vector<void *> src_ptrs;
        std::vector<int>    chunk_sizes;
        for (const auto &shard : ag_needed_tensors) {
            src_ptrs.push_back(shard.data());
            chunk_sizes.push_back(shard.nbytes() / num_splits_);
        }

        for (int i = 0; i < num_splits_; ++i) {
            workers[i] = std::make_unique<SendWorker>(comm_, p2p_workspace_size_req, src_ptrs,
                                                      chunk_sizes, i, comm_streams_, comm_event_);
        }
        return workers;
    }

  private:
    Communicator            *comm_;
    int                      num_splits_;
    bool                     use_sdma_;
    std::vector<hipStream_t> comm_streams_;
    hipEvent_t               comm_event_;
};

class GemmConsumer {
  public:
    virtual std::optional<std::vector<_internal::HIPTensor>>
    Run(const std::vector<_internal::HIPTensor> &) = 0;
};

} // namespace primus_turbo::async_tp
