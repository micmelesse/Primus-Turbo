import pytest
import torch

from primus_turbo.pytorch.ops import grouped_gemm, grouped_gemm_init
from tests.pytorch.ref.gemm_ref import generate_seq_len, grouped_gemm_ref
from tests.test_utils import compute_snr


@pytest.mark.parametrize("B", [2, 4, 8, 16])
@pytest.mark.parametrize("M", [256, 512, 2048])
@pytest.mark.parametrize("N_K", [(1024, 2048), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_blockwise_fp8_grouped_gemm_func(B, M, N_K, dtype):
    N, K = N_K
    device = "cuda"
    seg_lens = generate_seq_len(B, B * M).to(device)  # int64
    print(seg_lens, seg_lens.dtype)
    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    init_ptr = grouped_gemm_init(B)  # init before forward
    out, init_ptr = grouped_gemm(a, b, seg_lens, init_ptr)

    out_ref = grouped_gemm_ref(a_ref, b_ref, seg_lens.clone(), True)

    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    a_grad_ref = a_ref.grad
    b_grad_ref = b_ref.grad

    out.backward(grad_out)
    a_grad = a.grad
    b_grad = b.grad

    agrad_snr = compute_snr(a_grad_ref, a_grad)
    print(f"aGrad-SNR: {agrad_snr:.2f} dB")
    assert agrad_snr > 20, "aGrad too low"
    bgrad_snr = compute_snr(b_grad_ref, b_grad)
    print(f"bGrad-SNR: {bgrad_snr:.2f} dB")
    assert bgrad_snr > 20, "bGrad too low"


if __name__ == "__main__":
    # torch.manual_seed(1234)
    # test_blockwise_fp8_grouped_gemm_func(4, 512, 6528, 1536, torch.bfloat16)
    test_blockwise_fp8_grouped_gemm_func(4, 1024, (4096, 4096), torch.float16)
