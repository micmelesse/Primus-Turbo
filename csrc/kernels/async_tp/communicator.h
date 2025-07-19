#pragma once
#include <vector>
namespace primus_turbo::async_tp {

template <typename Comm> class Communicator {
  public:
    Comm       *derived() { return reinterpret_cast<Comm *>(this); }
    const Comm *const_derived() const { return derived(); }

    template <typename T> void AllGather(std::vector<T> &output, T &input) {
        derived()->AllGatherImpl(output, input);
    }

    int rank() const { return derived()->rank(); }

    int world_size() const { return derived()->world_size(); }
};

} // namespace primus_turbo::async_tp
