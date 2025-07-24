import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.normalization import rmsnorm
from tests.test_utils import get_tolerances


def test_rmsnorm_ops():
    torch.manual_seed(1234)
    dtype = torch.float32
    device = "cuda:0"
    eps = 1e-6

    seq = 1024
    hidden = 4096

    x = torch.randn(seq, hidden, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(hidden, dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # Forward
    y_ref = F.rms_norm(x_ref, [hidden], gamma_ref, eps)
    y = rmsnorm(x, gamma, eps)

    # print(y_ref, y_ref.shape)
    # print(y, y.shape)
    torch.testing.assert_close(y_ref, y, **get_tolerances(dtype))

    # Backward
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    # print(x.grad)
    # print(x_ref.grad)

    # print(gamma.grad)
    # print(gamma_ref.grad)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **get_tolerances(dtype))


if __name__ == "__main__":
    # test_rmsnorm_ops()
    B = 2
    M = 1024
    N = 2048
    K = 4096
    ori_dtype = torch.float16
    device = "cuda"
    seg_lens = torch.zeros([B], dtype=torch.int32, device=device)
    seg_lens[0] = 512
    seg_lens[1] = B * M - seg_lens[0]
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    b = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=True)
    c = torch.randn((B * M, N), dtype=ori_dtype, device=device, requires_grad=True)
    print(help(torch.ops.primus_turbo_cpp_extension))

    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, c, seg_lens, False, True)
    # print(out.shape)
