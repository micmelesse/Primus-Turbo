import torch

from primus_turbo.pytorch.modules import GroupedLinear
from tests.pytorch.ref.gemm_ref import GroupedLinearRef, generate_seq_len
from tests.test_utils import compute_snr


def test_grouped_linear(B, M, N, K, dtype, enable_torch_compile):
    B_M = B * M
    device = "cuda"
    seq_len = generate_seq_len(B, B_M).to(device)  # int64

    torch._dynamo.reset()

    primus_linear = GroupedLinear(B, K, N, device, dtype=dtype)
    torch_linear = GroupedLinearRef(B, K, N, device, dtype=dtype)
    with torch.no_grad():
        torch_linear.weight.copy_(primus_linear.weight)
    primus_linear = torch.compile(primus_linear, fullgraph=True, mode="max-autotune")
    x1 = torch.randn((B_M, K), device=device, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    out1 = primus_linear(x1, seq_len)
    out2 = torch_linear(x2, seq_len)
    # print(out1)
    # print(out2)
    out_snr = compute_snr(out1, out2)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    grad_output = torch.randn_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    x_grad_snr = compute_snr(x1.grad, x2.grad)
    print(f"X_gard-SNR: {x_grad_snr:.2f} dB")
    assert out_snr > 20, "x_grad_snr too low"

    w_grad_snr = compute_snr(primus_linear.weight.grad, torch_linear.weight.grad)
    print(f"W_gard-SNR: {w_grad_snr:.2f} dB")
    assert out_snr > 20, "w_grad_snr too low"


if __name__ == "__main__":
    torch.manual_seed(1234)
    test_grouped_linear(B=4, M=512, N=1024, K=2048, dtype=torch.float16, enable_torch_compile=False)
