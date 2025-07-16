#pragma once
#include <vector>
namespace primus_turbo::async_tp {

template <typename Spec> class Communicator {
  public:
    Spec       *derived() { return reinterpret_cast<Spec *>(this); }
    const Spec *const_derived() const { return derived(); }

    template <typename T> std::vector<T> AllGather(const T &obj) {
        std::vector<T> result(world_size());
        return derived()->AllGatherImpl(result.data(), &obj, sizeof(obj));
    }

    int rank() const { return derived()->rank(); }

    int world_size() const { return derived()->world_size(); }
};

} // namespace primus_turbo::async_tp
