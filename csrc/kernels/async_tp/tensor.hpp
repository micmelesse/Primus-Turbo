#pragma once
#include "hip/hip_runtime.h"
#include <numeric>
#include <vector>

namespace primus_turbo::async_tp::_internal {

class HIPTensor {
  public:
    explicit HIPTensor(void *ptr_dev, const std::vector<size_t> &shape, const hipDataType dtype,
                       float *scale_ptr_dev = nullptr, float *scale_inv_ptr_dev = nullptr)
        : data_dev_(ptr_dev), shape_(shape), dtype_(dtype), scale_ptr_dev_(scale_ptr_dev),
          scale_inv_ptr_dev_(scale_inv_ptr_dev) {}

    void                      *data() const { return data_dev_; }
    const void                *const_data() const { return this->data(); }
    const std::vector<size_t> &shape() const { return shape_; }

    auto nbytes() const { return std::accumulate(shape_.begin(), shape_.end(), 1); }

  private:
    void               *data_dev_;
    std::vector<size_t> shape_;
    hipDataType         dtype_;
    float              *scale_ptr_dev_;
    float              *scale_inv_ptr_dev_;
};

template <typename T> HIPTensor make_hip_tensor(T other_tensor);
} // namespace primus_turbo::async_tp::_internal
