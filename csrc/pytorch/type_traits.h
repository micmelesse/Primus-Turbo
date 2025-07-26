#pragma once
#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck_tile/core.hpp"
#include <torch/extension.h>

#include "primus_turbo/dtype.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// ************************************************ //

// CK supported scalar data types.
// https://rocm.docs.amd.com/projects/composable_kernel/en/develop/reference/Composable_Kernel_supported_scalar_types.html

// Map torch::ScalarType -> CK type
template <torch::ScalarType scalar_type> struct TorchToCKType;
template <torch::ScalarType scalar_type> struct TorchToCKTileType;
template <> struct TorchToCKType<torch::kFloat8_e4m3fnuz> {
    using type = ck::f8_t;
};

template <> struct TorchToCKType<torch::kFloat8_e4m3fn> {
    using type = ck::f8_t;
};

template <> struct TorchToCKType<torch::kFloat8_e5m2fnuz> {
    using type = ck::bf8_t;
};

template <> struct TorchToCKType<torch::kFloat8_e5m2> {
    using type = ck::bf8_t;
};

template <> struct TorchToCKType<torch::kHalf> {
    using type = ck::half_t;
};

template <> struct TorchToCKType<torch::kBFloat16> {
    using type = ck::bhalf_t;
};

template <> struct TorchToCKType<torch::kFloat> {
    using type = float32;
};

template <> struct TorchToCKTileType<torch::kHalf> {
    using type = ck::half_t;
};

template <> struct TorchToCKTileType<torch::kBFloat16> {
    using type = ck_tile::bfloat16_t;
};
// ************************************************ //

} // namespace primus_turbo::pytorch
