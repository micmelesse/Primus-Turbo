// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "primus_turbo/dtype.h"
#include <torch/extension.h>

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

// ************************************************

// CK supported scalar data types.
// https://rocm.docs.amd.com/projects/composable_kernel/en/develop/reference/Composable_Kernel_supported_scalar_types.html

/**
 *  DataType Mapping : torch::ScalarType -> CKType
 */

template <torch::ScalarType scalar_type> struct TorchToCKType;
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

/**
 *  DataType Mapping : torch::ScalarType -> CK-Tile Type
 */
template <torch::ScalarType scalar_type> struct TorchToCKTileType;

template <> struct TorchToCKTileType<torch::kHalf> {
    using type = ck_tile::half_t;
};

template <> struct TorchToCKTileType<torch::kBFloat16> {
    using type = ck_tile::bfloat16_t;
};

template <> struct TorchToCKTileType<torch::kFloat> {
    using type = float32;
};
// ************************************************

static inline bool is_16bit_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

static inline bool is_8bit_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fnuz || dtype == at::kFloat8_e4m3fn ||
           dtype == at::kFloat8_e5m2fnuz || dtype == at::kFloat8_e5m2;
}

static inline bool is_floating_point_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}
} // namespace primus_turbo::pytorch
