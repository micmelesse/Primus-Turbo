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
    torch.manual_seed(1234)
    from tests.test_utils import compute_snr, cosine_similarity

    def grouped_gemm_ref(a, b, seg_lens, trans_b=True):
        seg_lens = seg_lens.cpu().numpy()
        out = []
        start = 0
        for i, size in enumerate(seg_lens):
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            out.append(a[start : start + size, :] @ rhs)
            start += size
        return torch.cat(out)

    # test_rmsnorm_ops()
    B = 2
    M = 512
    N = 1024
    K = 256
    ori_dtype = torch.float16
    # ori_dtype = torch.float32
    device = "cuda"
    seg_lens = torch.zeros([B], dtype=torch.int32, device=device)
    seg_lens[0] = 256
    seg_lens[1] = B * M - seg_lens[0]
    a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=False)
    # a[0:512, :] = 2  # 将前512行设为2
    # a[512:2048, :] = 3  # 将其余行设为3
    # 修改b矩阵初始化：按照N,K为单位，B[0:M:K] = 1; B[1:M:K] = 2
    b = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=False)
    # b[0, :, :] = 1  # B[0:N:K] = 1
    # b[1, :, :] = 2  # B[1:N:K] = 2

    print(a)
    print(b)
    c = torch.zeros((B * M, N), dtype=ori_dtype, device=device, requires_grad=False)

    out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, c, seg_lens, False, True)
    out_ref = grouped_gemm_ref(a.clone(), b.clone(), seg_lens.clone(), True)
    print(out[0])
    print(out_ref[0])
    print(cosine_similarity(out, out_ref), compute_snr(out, out_ref))
