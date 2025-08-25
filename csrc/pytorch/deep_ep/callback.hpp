#include "primus_turbo/any_cached.hpp"
#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <memory>
#include <torch/extension.h>
#include <unordered_map>

namespace primus_turbo::pytorch::deep_ep {

struct CallBackMeta;

class CallBackMap {

public:
    using ElementType = typename primus_turbo::AnyCached<CallBackMeta>::ElementType;

public:
    CallBackMap() = default;

    CallBackMeta *get() {
        ElementType   item = pool_.get();
        CallBackMeta *key  = item.get();
        map_.insert({reinterpret_cast<void *>(key), std::move(item)});
        return key;
    }

    void erase(void *key) { map_.erase(key); }

private:
    std::unordered_map<void *, ElementType> map_;
    primus_turbo::AnyCached<CallBackMeta>   pool_;

} g_callback_map;

#define FOREACH_ALL_CALLBACK_MEMBER(_)                                                             \
    _(recv_counter, int)                                                                           \
    _(recv_expert_counter, int)                                                                    \
    _(recv_x_ptr, uintptr_t)                                                                       \
    _(recv_topk_idx_ptr, uintptr_t)                                                                \
    _(recv_src_idx_ptr, uintptr_t)                                                                 \
    _(recv_topk_weights_ptr, uintptr_t)                                                            \
    _(recv_x_scales_ptr, uintptr_t)

struct CallBackMeta {

#define DEFINE_CALLBACKMETA_HOST_DEVICE_MAPPED_POINTER(member, type)                               \
    volatile type *moe_##member          = nullptr;                                                \
    type          *moe_##member##_mapped = nullptr;

    FOREACH_ALL_CALLBACK_MEMBER(DEFINE_CALLBACKMETA_HOST_DEVICE_MAPPED_POINTER)

    int hidden;
    int num_local_experts;
    int num_topk;
    int num_scales;
    int x_scales_dim;

    torch::Tensor                recv_x;
    torch::Tensor                recv_src_idx;
    std::optional<torch::Tensor> recv_topk_idx;
    std::optional<torch::Tensor> recv_topk_weights;
    std::optional<torch::Tensor> recv_x_scales;

    CallBackMeta() { Alloc(); }

    ~CallBackMeta() { Destroy(); }

    CallBackMeta(int moe_hidden, int moe_num_local_experts, int moe_num_topk, int moe_num_scales,
                 int moe_x_scales_dim, torch::Tensor moe_recv_x, torch::Tensor moe_recv_src_idx,
                 std::optional<torch::Tensor> moe_recv_topk_idx,
                 std::optional<torch::Tensor> moe_recv_topk_weights,
                 std::optional<torch::Tensor> moe_recv_x_scales)
        : hidden(moe_hidden), num_local_experts(moe_num_local_experts), num_topk(moe_num_topk),
          num_scales(moe_num_scales), x_scales_dim(moe_x_scales_dim), recv_x(moe_recv_x),
          recv_src_idx(moe_recv_src_idx), recv_topk_idx(moe_recv_topk_idx),
          recv_topk_weights(moe_recv_topk_weights), recv_x_scales(moe_recv_x_scales) {
        Alloc();
    }

private:
    void Alloc() {
#define MALLO_HOST_DEVICE_MAPPED_POINTER(member, type)                                             \
    PRIMUS_TURBO_CHECK_HIP(hipHostMalloc(&moe_##member, sizeof(type), hipHostAllocMapped));        \
    PRIMUS_TURBO_CHECK_HIP(hipHostGetDevicePointer(                                                \
        reinterpret_cast<void **>(&moe_##member##_mapped), const_cast<type *>(moe_##member), 0));

        FOREACH_ALL_CALLBACK_MEMBER(MALLO_HOST_DEVICE_MAPPED_POINTER);
#undef MALLO_HOST_DEVICE_MAPPED_POINTER
    }
    void Destroy() {
#define FREE_HOST_DEVICE_MAPPED_POINTER(member, type)                                              \
    PRIMUS_TURBO_CHECK_HIP(hipFreeHost(const_cast<type *>(moe_##member)));                         \
    PRIMUS_TURBO_CHECK_HIP(hipFreeHost(const_cast<type *>(moe_##member##_mapped)));

        FOREACH_ALL_CALLBACK_MEMBER(FREE_HOST_DEVICE_MAPPED_POINTER);
#undef FREE_HOST_DEVICE_MAPPED_POINTER
    }
};

} // namespace primus_turbo::pytorch::deep_ep
