import pytest
import torch

from primus_turbo.pytorch.core.float8 import Format, MXQuantConfig
from primus_turbo.pytorch.ops import gemm_fp8_blockwise
from tests.test_utils import compute_snr


@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype", [Format.E4M3, Format.E5M2])
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("B", [[], [1], [2], [4, 2]])
@pytest.mark.parametrize("M", [257, 4096])
@pytest.mark.parametrize("NK", [(255, 129), (2048, 7168)])
def test_gemm_fp8_blockwise_func(ori_dtype, dtype, block_size, B, M, NK):
    if B != []:
        if not (
            ori_dtype == torch.bfloat16
            and dtype == Format.E4M3
            and block_size == 128
            and M == 257
            and NK == (255, 129)
        ):
            pytest.skip("Only test one combination for B != []")

    N, K = NK
    device = "cuda:0"

    print(f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, dtype={dtype}, block_size={block_size}")

    x = torch.randn((*B, M, K), dtype=ori_dtype, device=device, requires_grad=True)
    w = torch.randn((N, K), dtype=ori_dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    # Ref
    out_ref = x_ref @ w_ref.T
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    x_grad_ref = x_ref.grad
    w_grad_ref = w_ref.grad

    # Config + FWD + BWD
    config = MXQuantConfig(dtype=dtype, block_size=block_size)
    out = gemm_fp8_blockwise(x, w, config)
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
