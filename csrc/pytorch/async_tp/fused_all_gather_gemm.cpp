#include "../extensions.h"
#include "../kernels/async_tp/all_gather_matmul.h"
#include "primus_turbo/macros.h"
#include "torch_communicator.hpp"

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

namespace primus_turbo {

namespace async_tp::_internal {
template <> HIPTensor make_hip_tensor(torch::Tensor torch_tensor) {}
} // namespace async_tp::_internal

namespace pytorch {

std::vector<torch::Tensor>
fused_all_gather_matmul(const torch::Tensor &A_shard, const std::vector<torch::Tensor> &Bs,
                        int64_t gather_dim, const std::string &group_name,
                        std::optional<torch::Tensor>               A_out,
                        std::optional<std::vector<torch::Tensor>>  mm_outs,
                        std::optional<std::vector<at::ScalarType>> out_dtypes, bool return_A,
                        bool skip_copy_local_ag_out) {

    PRIMUS_TURBO_CHECK(A_shard.dim() >= 2, "A_shard must be a matrix");
    for (const auto &B : Bs)
        PRIMUS_TURBO_CHECK(B.dim() == 2, "B must be a matrix");

    PRIMUS_TURBO_CHECK(out_dtypes->size() == Bs.size(), "out_types must be the same as Bs");
    PRIMUS_TURBO_CHECK(gather_dim >= 0 && gather_dim < A_shard.dim(), "Invalid gather_dim");

    auto group = c10d::resolve_process_group(group_name);
    auto comm  = PGCommunicator(group.get());

    auto A_shard_t   = async_tp::_internal::make_hip_tensor(A_shard);
    auto ag_producer = async_tp::PipelinedAllGatherProducer(&comm, 4, false);
    async_tp::InteralAllGatherMatmul();

    // bool flag_return_A               = return_A.value_or(true);
    // bool flag_comm_algo              = comm_algo.value_or("pipeline");
    // bool flag_num_splits             = num_splits.value_or(4);
    // bool flag_skip_copy_local_ag_out = skip_copy_local_ag_out.value_or(false);
    // bool flag_enable_sdma            = enable_sdma.value_or(false);
}
} // namespace pytorch
} // namespace primus_turbo
