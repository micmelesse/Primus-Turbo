import torch

from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref
from tests.test_utils import compute_snr


def test_blockwise_fp8_grouped_gemm_func(B, M, N, K, ori_dtype):
    device = "cuda"
    seg_lens = torch.zeros([B], dtype=torch.int32, device=device)
    seg_lens[0] = 256
    seg_lens[1] = 768
    seg_lens[2] = 768
    seg_lens[3] = 256
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out, _ = grouped_gemm(a, b, seg_lens, 0)
    out_ref = grouped_gemm_ref(a_ref, b_ref, seg_lens.clone(), True)
    # print(out)
    # print(out_ref)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    a_grad_ref = a_ref.grad
    b_grad_ref = b_ref.grad

    out.backward(grad_out)
    a_grad = a.grad
    b_grad = b.grad

    # print(a_grad_ref)
    # print(a_grad)
    # print(b_grad_ref)
    # print(b_grad)
    agrad_snr = compute_snr(a_grad_ref, a_grad)
    print(f"aGrad-SNR: {agrad_snr:.2f} dB")
    bgrad_snr = compute_snr(b_grad_ref, b_grad)
    print(f"aGrad-SNR: {bgrad_snr:.2f} dB")


if __name__ == "__main__":
    torch.manual_seed(1234)
    test_blockwise_fp8_grouped_gemm_func(4, 512, 1024, 1536, torch.float16)
