###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    MXQuantConfig,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8_blockwise, gemm_fp8_tensorwise
from tests.test_utils import compute_snr

torch.manual_seed(42)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("B", [[], [1], [2], [4, 2]])
@pytest.mark.parametrize("M", [257, 4096])
@pytest.mark.parametrize("NK", [(255, 129), (2048, 7168)])
def test_gemm_fp8_blockwise_func(dtype, block_size, B, M, NK):
    if B != []:
        if not (dtype == torch.bfloat16 and block_size == 128 and M == 257 and NK == (255, 129)):
            pytest.skip("Only test one combination for B != []")

    N, K = NK
    device = "cuda:0"

    print(f"\nB={B}, M={M}, N={N}, K={K}, dtype={dtype}, block_size={block_size}")

    x = torch.randn((*B, M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((N, K), dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    # Ref
    out_ref = x_ref @ w_ref.T
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    x_grad_ref = x_ref.grad
    w_grad_ref = w_ref.grad

    # Config + FWD + BWD
    config = MXQuantConfig(block_size=block_size)
    out = gemm_fp8_blockwise(x, w, trans_a=False, trans_b=True, out_dtype=dtype, config=config)
    out.backward(grad_out)
    x_grad = x.grad
    w_grad = w.grad

    # Check Shape
    print("out shape: ", out.shape, out_ref.shape)
    print("x_grad shape: ", x_grad.shape, x_grad_ref.shape)
    print("w_grad shape: ", w_grad.shape, w_grad_ref.shape)
    assert out.shape == out_ref.shape
    assert x_grad.shape == x_grad_ref.shape
    assert w_grad.shape == w_grad_ref.shape

    # Check Results
    # print("fwd")
    # print(out)
    # print(out_ref)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    # print("dgrad")
    # print(x_grad)
    # print(x_grad_ref)
    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    # print("wgrad")
    # print(w_grad)
    # print(w_grad_ref)
    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"


@pytest.mark.parametrize("m", [256, 512, 1024, 2048])
@pytest.mark.parametrize("n", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [255, 512, 1024, 2048])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("format", [Format.E4M3, Format.E5M2, Format.HYBRID])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_fp8_tensorwise(m, n, k, layout, format, dtype):
    print(f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}")

    device = "cuda:0"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Ref
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    grad_c = torch.randn_like(c_ref)
    c_ref.backward(grad_c)
    torch.cuda.synchronize()

    # Config + FWD + BWD
    config = Float8QuantConfig(format=format)
    c = gemm_fp8_tensorwise(a, b, trans_a, trans_b, dtype, config)
    c.backward(grad_c)

    # Check Shape
    # print("out shape: ", out.shape, out_ref.shape)
    # print("x_grad shape: ", x_grad.shape, x_grad_ref.shape)
    # print("w_grad shape: ", w_grad.shape, w_grad_ref.shape)
    assert c.shape == c_ref.shape
    assert a.grad.shape == a_ref.grad.shape
    assert b.grad.shape == b_ref.grad.shape

    snr_threshold = 25 if format == Format.E4M3 else 20
    # Check Results
    c_snr = compute_snr(c_ref, c)
    print(f"C-SNR: {c_snr:.2f} dB")
    assert c_snr > snr_threshold, "c_snr too low"

    a_grad_snr = compute_snr(a_ref.grad, a.grad)
    print(f"AGrad-SNR: {a_grad_snr:.2f} dB")
    assert a_grad_snr > snr_threshold, "a_grad_snr too low"

    b_grad_snr = compute_snr(b_ref.grad, b.grad)
    print(f"BGrad-SNR: {b_grad_snr:.2f} dB")
    assert b_grad_snr > snr_threshold, "b_grad_snr too low"


# TODO: This is tmp test code.
def test_gemm_fp8_rowwise():
    dtype = torch.bfloat16
    m = 1024
    n = 1024
    k = 1024
    format = Format.E4M3

    layout = "NT"
    print(f"\nM={m}, N={n}, K={k}, layout={layout}, dtype={dtype}, format={format}")

    device = "cuda:0"

    trans_a = layout[0] == "T"
    trans_b = layout[1] == "T"

    a_shape = (m, k) if not trans_a else (k, m)
    b_shape = (k, n) if not trans_b else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Ref
    a_mat = a_ref.T if trans_a else a_ref
    b_mat = b_ref.T if trans_b else b_ref
    c_ref = a_mat @ b_mat
    # grad_c = torch.randn_like(c_ref)
    # c_ref.backward(grad_c)
    torch.cuda.synchronize()

    # Config + FWD + BWD
    config = Float8QuantConfig(format=format, granularity=ScalingGranularity.ROWWISE)
    c = gemm_fp8_tensorwise(a, b, trans_a, trans_b, dtype, config)
    # c.backward(grad_c)

    print(c)
    print(c_ref)

    snr_threshold = 25 if format == Format.E4M3 else 20
    c_snr = compute_snr(c_ref, c)
    print(f"C-SNR: {c_snr:.2f} dB")
    assert c_snr > snr_threshold, "c_snr too low"
