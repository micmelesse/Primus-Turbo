// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "../extensions.h"
#include "../utils.h"

#include "../../kernels/pointwise/row_col_quant.h"
#include "../../kernels/pointwise/vectorized_pointwise.h"
#include "../extensions.h"
#include "../type_traits.h"
#include "../utils.h"

namespace primus_turbo::pytorch {

using namespace primus_turbo::dtype;

struct QuantizeParam {
    const float32 *scale;
};

__device__ inline float32 quantize_func(float32 value, const QuantizeParam &param) {
    return value * (*(param.scale));
}

struct DequantizeParam {
    const float32 *scale_inv;
};

__device__ inline float32 dequantize_func(float32 value, const DequantizeParam &param) {
    return value * (*(param.scale_inv));
}

inline bool is_torch_fp8(const at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e4m3fnuz ||
           dtype == at::kFloat8_e5m2 || dtype == at::kFloat8_e5m2fnuz;
}

at::Tensor fp8_quantize(const at::Tensor input, const at::Tensor scale,
                        const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf ||
                       input.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(scale.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(scale.ndimension() == 1);
    PRIMUS_TURBO_CHECK(is_torch_fp8(dest_dtype));

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    auto device = input.device();

    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(device));

    TORCH_TYPE_SWITCH_INPUT(
        input.scalar_type(), IType,
        TORCH_TYPE_SWITCH_FP8ONLY(
            output.scalar_type(), OType, constexpr int nvec = 32 / sizeof(IType); QuantizeParam p;
            p.scale = reinterpret_cast<const float32 *>(scale.data_ptr());
            VectorizedUnaryKernelLauncher<nvec, QuantizeParam, quantize_func>(
                reinterpret_cast<const IType *>(input.data_ptr()),
                reinterpret_cast<OType *>(output.data_ptr()), input.numel(), p,
                stream);); // NOLINT(*)
    );

    return output;
}

at::Tensor fp8_dequantize(const at::Tensor input, const at::Tensor scale_inv,
                          const at::ScalarType dest_dtype) {
    PRIMUS_TURBO_CHECK(is_torch_fp8(input.scalar_type()));
    PRIMUS_TURBO_CHECK(scale_inv.scalar_type() == at::kFloat);
    PRIMUS_TURBO_CHECK(scale_inv.ndimension() == 1);
    PRIMUS_TURBO_CHECK(dest_dtype == at::kBFloat16 || dest_dtype == at::kHalf ||
                       dest_dtype == at::kFloat);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    at::Tensor output = torch::empty_like(input, torch::dtype(dest_dtype).device(device));

    TORCH_TYPE_SWITCH_FP8ONLY(
        input.scalar_type(), IType,
        TORCH_TYPE_SWITCH_INPUT(
            output.scalar_type(), OType, constexpr int nvec = 32 / sizeof(OType); DequantizeParam p;
            p.scale_inv = reinterpret_cast<const float32 *>(scale_inv.data_ptr());
            VectorizedUnaryKernelLauncher<nvec, DequantizeParam, dequantize_func>(
                reinterpret_cast<const IType *>(input.data_ptr()),
                reinterpret_cast<OType *>(output.data_ptr()), input.numel(), p,
                stream);); // NOLINT(*)
    );

    return output;
}

at::Tensor fp8_quantize_row_col(at::Tensor &input, at::Tensor &scale, const bool is_row_major) {
    int64_t dims = input.ndimension();
    PRIMUS_TURBO_CHECK(dims == 2 || dims == 3, "only support 2D or 3D tensor");
    int64_t    b = 1, m = 0, k = 0;
    at::Tensor output;
    if (dims == 2) {
        m = input.size(0);
        k = input.size(1);

        output = at::empty({m, k}, at::dtype(at::kFloat8_e4m3fnuz).device(at::kCUDA));
    }

    else if (dims == 3) {
        b      = input.size(0);
        m      = input.size(1);
        k      = input.size(2);
        output = at::empty({b, m, k}, at::dtype(at::kFloat8_e4m3fnuz).device(at::kCUDA));
        if (is_row_major) {
            m = b * m;
        } else {
            k = b * k;
        }
    }
    auto stream = at::cuda::getCurrentCUDAStream();

    if (input.dtype() == at::kBFloat16) {
        using InType    = typename TorchToCKTileType<at::kBFloat16>::type;
        using ScaleType = typename TorchToCKTileType<torch::kFloat>::type;
        using OutType   = ck_tile::fp8_t;

        quant_dequant_2d<InType, ScaleType, OutType>(
            reinterpret_cast<const InType *>(input.data_ptr()),
            reinterpret_cast<const ScaleType *>(scale.data_ptr()),
            reinterpret_cast<OutType *>(output.data_ptr()), m, k, !is_row_major, stream);
    } else if (input.dtype() == at::kHalf) {
        using InType    = typename TorchToCKTileType<at::kHalf>::type;
        using ScaleType = typename TorchToCKTileType<torch::kFloat>::type;
        using OutType   = ck_tile::fp8_t;

        quant_dequant_2d<InType, ScaleType, OutType>(
            reinterpret_cast<const InType *>(input.data_ptr()),
            reinterpret_cast<const ScaleType *>(scale.data_ptr()),
            reinterpret_cast<OutType *>(output.data_ptr()), m, k, !is_row_major, stream);
    } else {
        PRIMUS_TURBO_CHECK(false, "Quantization only support float16 and bfloat16");
    }
    if (dims == 3 && is_row_major == false) {
        output = output.reshape({m, b, k}).transpose(0, 1).contiguous();
    }
    return output;
}

at::Tensor fp8_dequantize_row_col(at::Tensor &input, at::Tensor &scale,
                                  torch::ScalarType scalar_type, const bool is_row_major) {
    int64_t dims = input.ndimension();
    PRIMUS_TURBO_CHECK(dims == 2 || dims == 3, "only support 2D or 3D tensor");
    int64_t    b = 1, m = 0, k = 0;
    at::Tensor output;
    if (dims == 2) {
        m = input.size(0);
        k = input.size(1);

        output = at::empty({m, k}, at::dtype(scalar_type).device(at::kCUDA));
    }

    else if (dims == 3) {
        b = input.size(0);
        m = input.size(1);
        k = input.size(2);
        if (is_row_major) {
            m = b * m;
        } else {
            k = b * k;
        }
        output = at::empty({b, m, k}, at::dtype(scalar_type).device(at::kCUDA));
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    using InType    = ck_tile::fp8_t; // FP8
    using ScaleType = typename TorchToCKTileType<torch::kFloat>::type;

    switch (scalar_type) {
    case torch::kHalf: {
        using OutType = typename TorchToCKTileType<torch::kHalf>::type;
        quant_dequant_2d<InType, ScaleType, OutType>(
            reinterpret_cast<const InType *>(input.data_ptr()),
            reinterpret_cast<const ScaleType *>(scale.data_ptr()),
            reinterpret_cast<OutType *>(output.data_ptr()), m, k, !is_row_major, stream);
        break;
    }
    case torch::kBFloat16: {
        using OutType = typename TorchToCKTileType<torch::kBFloat16>::type;
        quant_dequant_2d<InType, ScaleType, OutType>(
            reinterpret_cast<const InType *>(input.data_ptr()),
            reinterpret_cast<const ScaleType *>(scale.data_ptr()),
            reinterpret_cast<OutType *>(output.data_ptr()), m, k, !is_row_major, stream);
        break;
    }
    default: {
        PRIMUS_TURBO_CHECK(false, "Dequantization only support float16, bfloat16 and float32");
    }
    }

    if (dims == 3 && is_row_major == false) {
        output = output.reshape({m, b, k}).transpose(0, 1).contiguous();
    }
    return output;
}

} // namespace primus_turbo::pytorch
