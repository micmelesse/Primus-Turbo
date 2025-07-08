#include "cuda_runtime_api.h"
#include "xla/ffi/api/ffi.h"
#include <pybind11/pybind11.h>

namespace nb  = nanobind;
namespace ffi = xla::ffi;

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

template <typename T>
__global__ void MultiplyAddKernel(const T *a, const T *b, const T *c, T *d, size_t n) {
    size_t       tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t grid_stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += grid_stride) {
        d[i] = a[i] * b[i] + c[i]
    }
}

template <typename T>
ffi::Error MultiplyAddImpl(size_t n, ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::AnyBuffer c,
                           ffi::Result<ffi::AnyBuffer> d) {
    const int block_dim = 128;
    const int grid_dim  = 1;
    MultiplyAddKernel<<<grid_dim, block_dim>>>(a.typed_data<T>(), b.typed_data<T>(),
                                               c->typed_data<T>(), d->typed_data<T>(), n);
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

ffi::Error MultiplyAddDispatch(size_t n, ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::AnyBuffer > c,
                               ffi::Result<ffi::AnyBuffer> d) {

    ELEMENT_TYPE_DISPATCH(x.element_type(), MultiplyAddImpl, n, a, b, c, d);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MultiplyAddImpl, MultiplyAddDispatch,
                              ffi::Ffi::Bind()
                                  .Attr<int64_t>("n")
                                  ..Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());

template <typename T> nb::capsule EncapsulateFfiHandler(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

XLA_FFI_DEFINE_HANDLER(kStateInstantiate, StateInstantiate, ffi::Ffi::BindInstantiate());
XLA_FFI_DEFINE_HANDLER(kStateExecute, StateExecute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::State<State>>()
                           .Ret<ffi::BufferR0<ffi::S32>>());
                           
PYBIND11_MODULE(_cuda_examples, m) {
       m.def("registrations", []() {
        nb::dict registrations;
        registrations["rms_norm"] = EncapsulateFfiHandler(MultiplyAddImpl);
        return registrations;
    });
}