#pragma once
#include <cstddef>
namespace primus_turbo::async_tp {

class Communicator {
  public:
    virtual void AllGather(void *dst, void *src, size_t len) = 0;
    virtual int  rank() const                                = 0;
    virtual int  world_size() const                          = 0;
};

} // namespace primus_turbo::async_tp
