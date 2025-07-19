#include "symm_mem.h"
#include "primus_turbo/macros.h"
#include <hip/hip_runtime_api.h>

namespace primus_turbo::async_tp {

#define ROUND_UP(a, b) (((a + b - 1) / b) * b)

SymmetricMemory::SymmetricMemory(std::vector<void *> buffers, std::vector<void *> signal_pads,
                                 size_t buffer_size, int local_device_idx, int rank, int world_size)
    : buffers_(std::move(buffers)), signal_pads_(std::move(signal_pads)), buffer_size_(buffer_size),
      local_device_idx_(local_device_idx), rank_(rank), world_size_(world_size) {
    const size_t arr_size = sizeof(void *) * world_size_;
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(buffers_dev_, arr_size));
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(signal_pads_dev_, arr_size));

    PRIMUS_TURBO_CHECK_HIP(
        hipMemcpy(buffers_dev_, buffers_.data(), arr_size, hipMemcpyHostToDevice));
    PRIMUS_TURBO_CHECK_HIP(
        hipMemcpy(signal_pads_dev_, signal_pads_.data(), arr_size, hipMemcpyHostToDevice));
}

std::vector<void *> SymmetricMemory::buffer_ptrs() {
    return buffers_;
}

std::vector<void *> SymmetricMemory::signal_pad_ptrs() {
    return signal_pads_;
}

void **SymmetricMemory::buffer_ptrs_dev() {
    return buffers_dev_;
}

void **SymmetricMemory::signal_pad_ptrs_dev() {
    return signal_pads_dev_;
}

size_t SymmetricMemory::buffer_size() {
    return buffer_size_;
}

size_t SymmetricMemory::signal_pad_size() {
    return signal_pad_size;
}

int SymmetricMemory::rank() {
    return rank_;
}
int SymmetricMemory::world_size() {
    return world_size_;
}

template <typename Comm>
std::unique_ptr<SymmetricMemory> rendezvous_symm_mem(size_t alloc_size, Communicator<Comm> *comm) {
    size_t signal_pad_offset = ROUND_UP(alloc_size, 16);
    size_t block_size        = signal_pad_offset + signal_pad_size;
    int    world_size        = comm->world_size();
    int    rank              = comm->rank();

    void *ptr{nullptr};
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(&ptr, block_size));

    hipIpcMemHandle_t handle;
    PRIMUS_TURBO_CHECK_HIP(hipIpcGetMemHandle(&handle, ptr));

    std::vector<hipIpcMemHandle_t> handles(world_size);
    comm->AllGather(&handle, handles.data());
    std::vector<void *> buffers(world_size);
    std::vector<void *> signals(world_size);

    for (int i = 0; i < world_size; ++i) {
        if (i == rank) {
            buffers[i] = ptr;
            signals[i] = (char *) ptr + signal_pad_offset;
        } else {
            PRIMUS_TURBO_CHECK_HIP(hipIpcOpenMemHandle(&buffers[i], handles[i], 0));
            signals[i] = (char *) buffers[i] + signal_pad_offset;
        }
    }

    int device_id{-1};
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device_id));
    return std::make_unique<SymmetricMemory>(buffers, signals, alloc_size, device_id, rank,
                                             world_size);
}

SymmetricMemoryManager &SymmetricMemoryManager::Instance() {
    static SymmetricMemoryManager instance;
    return instance;
}

template <typename Comm>
SymmetricMemory *SymmetricMemoryManager::GetSymmMem(size_t alloc_size, Communicator<Comm> *comm) {
    int  world_size = comm->world_size();
    auto key        = std::make_tuple(alloc_size, world_size);
    auto iter       = symm_mem_.find(key);
    if (iter != symm_mem_.end()) {
        return iter->second.get();
    }
    auto symm_mem = rendezvous_symm_mem(alloc_size, comm);
    symm_mem_.insert({key, symm_mem});
    return symm_mem.get();
}

} // namespace primus_turbo::async_tp
