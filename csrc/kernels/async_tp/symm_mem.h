#pragma once
#include "communicator.h"

#include <map>
#include <memory>
#include <vector>

namespace primus_turbo::async_tp {

static constexpr size_t kSignalPadSize = 1024;

class SymmetricMemory {

  public:
    SymmetricMemory(std::vector<void *> buffers, std::vector<void *> signal_pads,
                    size_t buffer_size, int local_device_idx, int rank, int world_size);

    ~SymmetricMemory();

    template <typename T = void> std::vector<T *> buffer_ptrs();
    std::vector<void *>                           signal_pad_ptrs();
    void                                        **buffer_ptrs_dev();
    void                                        **signal_pad_ptrs_dev();
    size_t                                        buffer_size();
    size_t                                        signal_pad_size();

    int rank();
    int world_size();

  private:
    std::vector<void *> buffers_;
    std::vector<void *> signal_pads_;
    size_t              buffer_size_;
    int                 local_device_idx_;
    int                 rank_;
    int                 world_size_;
    void              **buffers_dev_;
    void              **signal_pads_dev_;
};

class SymmetricMemoryManager {
  public:
    using key = std::tuple<size_t, int>;

    static SymmetricMemoryManager &Instance();

    SymmetricMemory *GetSymmMem(size_t alloc_size, Communicator *comm);

  private:
    SymmetricMemoryManager() = default;

  private:
    std::map<key, std::unique_ptr<SymmetricMemory>> symm_mem_;
};

} // namespace primus_turbo::async_tp
