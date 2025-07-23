#pragma once

#include "primus_turbo/common.h"
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

namespace primus_turbo::jax {

XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormFwdHandler);
XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormBwdHandler);

} // namespace primus_turbo::jax
