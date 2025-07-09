#include "extensions.h"
#include <pybind11/pybind11.h>

namespace primus_turbo::jax {

template <typename T> pybind11::capsule EncapsulateFFI(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
    pybind11::dict registrations;
    registrations["multiply_add"] = EncapsulateFFI(MultiplyAddHandler);
    return registrations;
}

PYBIND11_MODULE(_C, m) {
    m.def("registrations", &Registrations);
}

} // namespace primus_turbo::jax
