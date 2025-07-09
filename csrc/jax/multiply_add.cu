#include "extensions.h"
#include <cuda_runtime.h>

#define ELEMENT_TYPE_DISPATCH(element_type, fn, ...)                                               \
    switch (element_type) {                                                                        \
    case ffi::F32:                                                                                 \
        return fn<float>(__VA_ARGS__);                                                             \
    case ffi::F64:                                                                                 \
        return fn<double>(__VA_ARGS__);                                                            \
    case ffi::C64:                                                                                 \
        return fn<std::complex<float>>(__VA_ARGS__);                                               \
    case ffi::C128:                                                                                \
        return fn<std::complex<double>>(__VA_ARGS__);                                              \
    default:                                                                                       \
        return ffi::Error::InvalidArgument("Unsupported input data type.");                        \
    }
namespace primus_turbo::jax {

template <typename T>
__global__ void MultiplyAddKernel(const T *a, const T *b, const T *c, T *d, size_t n) {
    size_t       tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid_stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += grid_stride) {
        d[i] = a[i] * b[i] + c[i];
    }
}

template <typename T>
ffi::Error MultiplyAddImpl(ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::AnyBuffer c,
                           ffi::Result<ffi::AnyBuffer> d) {
    const int block_dim = 128;
    const int grid_dim  = 1;
    MultiplyAddKernel<<<grid_dim, block_dim>>>(a.typed_data<T>(), b.typed_data<T>(),
                                               c.typed_data<T>(), d->typed_data<T>(),
                                               a.element_count());
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

ffi::Error MultiplyAddDispatch(ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::AnyBuffer c,
                               ffi::Result<ffi::AnyBuffer> d) {

    ELEMENT_TYPE_DISPATCH(a.element_type(), MultiplyAddImpl, a, b, c, d);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MultiplyAddHandler, MultiplyAddDispatch,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());
} // namespace primus_turbo::jax
