import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.ops import grouped_gemm_fp8_blockwise
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref
from tests.test_utils import compute_snr


@pytest.mark.parametrize("B", [1, 2, 3, 32])
@pytest.mark.parametrize("M", [32, 256, 2048])
@pytest.mark.parametrize("NK", [(4096, 7168)])
@pytest.mark.parametrize("ori_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dtype", [turbo.float8_e4m3, turbo.float8_e5m2])
@pytest.mark.parametrize("block_size", [128, 256])
def test_blockwise_fp8_grouped_gemm_func(B, M, NK, ori_dtype, dtype, block_size):
    N, K = NK
    device = "cuda:0"

    print(f"\nB={B}, M={M}, N={N}, K={K}, ori_dtype={ori_dtype}, dtype={dtype}, block_size={block_size}")

    #
    dist = 0.2 + 0.8 * torch.rand(B)
    dist /= dist.sum()
    seg_lens = (dist * M * B).to(torch.long).to(device)
    error = M * B - seg_lens.sum()
    seg_lens[-1] += error
    #
    x = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    # print(x.shape, x.dtype)
    # print(w.shape, w.dtype)
    # print(seg_lens)
    # print(seg_indptr)

    # Ref
    out_ref = grouped_gemm_ref(x_ref, w_ref, seg_lens, True)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    x_grad_ref = x_ref.grad
    w_grad_ref = w_ref.grad

    # Turbo
    out = grouped_gemm_fp8_blockwise(x, w, seg_lens, block_size, dtype)
    out.backward(grad_out)
    x_grad = x.grad
    w_grad = w.grad

    out_snr = compute_snr(out_ref, out)

    print("fwd")
    # print(out, out.shape)
    # print(out_ref, out.shape)
    out_snr = compute_snr(out_ref, out)
    print(f"Out-SNR: {out_snr:.2f} dB")
    assert out_snr > 20, "out_snr too low"

    print("dgrad")
    # print(x_grad, x_grad.shape)
    # print(x_grad_ref, x_grad_ref.shape)
    xgrad_snr = compute_snr(x_grad_ref, x_grad)
    print(f"XGrad-SNR: {xgrad_snr:.2f} dB")
    assert xgrad_snr > 20, "xgrad_snr too low"

    print("wgrad")
    # print(w_grad, w_grad.shape)
    # print(w_grad_ref, w_grad_ref.shape)
    wgrad_snr = compute_snr(w_grad_ref, w_grad)
    print(f"WGrad-SNR: {wgrad_snr:.2f} dB")
    assert wgrad_snr > 20, "wgrad_snr too low"


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

    def grouped_gemm_variable_k_ref(a, b, seg_lens):
        seg_lens = seg_lens.cpu().numpy()
        out = torch.zeros((B, M, N), dtype=a.dtype, device="cuda", requires_grad=False)
        start = 0
        for i, size in enumerate(seg_lens):
            a_tmp = a[start : start + size, :].t()
            b_tmp = b[start : start + size, :]
            out_tmp = a_tmp @ b_tmp
            out[i] = out_tmp
            start += size
        return out

    NT = True
    NN = True
    TN = True
    ori_dtype = torch.float16
    # ori_dtype = torch.float32
    device = "cuda"

    if NN is True:
        B = 4
        M = 512
        N = 1024
        K = 2048
        seg_lens = torch.zeros([B], dtype=torch.int64, device=device)
        seg_lens[0] = 256
        seg_lens[1] = 768
        seg_lens[2] = 768
        seg_lens[3] = 256
        a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=False)
        b = torch.randn((B, K, N), dtype=ori_dtype, device=device, requires_grad=False)

        temp_ptr = torch.ops.primus_turbo_cpp_extension.init_grouped_gemm(
            torch.tensor(B, dtype=torch.int64, device=device)
        )
        out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, seg_lens, False, False, temp_ptr)
        print(out.shape)
        out_ref = grouped_gemm_ref(a.clone(), b.clone(), seg_lens.clone(), False)
        print(out_ref.shape)
        print("NN", cosine_similarity(out, out_ref), compute_snr(out, out_ref))

        M = 512
        N = 1024
        K = 1024
        seg_lens[0] = 256
        seg_lens[1] = 768
        seg_lens[2] = 768
        seg_lens[3] = 256
        a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=False)
        b = torch.randn((B, K, N), dtype=ori_dtype, device=device, requires_grad=False)
        out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, seg_lens, False, False, temp_ptr)
        out_ref = grouped_gemm_ref(a.clone(), b.clone(), seg_lens.clone(), False)
        print("NN", cosine_similarity(out, out_ref), compute_snr(out, out_ref))

    if NT is True:
        B = 4
        M = 512
        N = 1024
        K = 2048
        seg_lens = torch.zeros([B], dtype=torch.int64, device=device)
        seg_lens[0] = 256
        seg_lens[1] = 768
        seg_lens[2] = 768
        seg_lens[3] = 256
        a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=True)
        b = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=True)
        temp_ptr = torch.ops.primus_turbo_cpp_extension.init_grouped_gemm(
            torch.tensor(B, dtype=torch.int64, device=device)
        )
        out = torch.ops.primus_turbo_cpp_extension.grouped_gemm(a, b, seg_lens, False, True, temp_ptr)
        out_ref = grouped_gemm_ref(a.clone(), b.clone(), seg_lens.clone(), True)
        # print(out[0])
        # print(out_ref[0])
        print("NT", cosine_similarity(out, out_ref), compute_snr(out, out_ref))

        block_size = 128
        out_fp8 = grouped_gemm_fp8_blockwise(
            a.clone(), b.clone(), seg_lens.clone(), block_size, dtype=ori_dtype
        )
        grad_out = torch.randn_like(out_ref)
        out_fp8.backward(grad_out)
        x_grad = a.grad
        print(x_grad.shape)
        print("NT", cosine_similarity(out_fp8, out_ref), compute_snr(out_fp8, out_ref))

    if TN is True:
        B = 4
        M = 2048
        N = 1024
        K = 512
        seg_lens = torch.zeros([B], dtype=torch.int64, device=device)
        seg_lens[0] = 256
        seg_lens[1] = 768
        seg_lens[2] = 768
        seg_lens[3] = 256
        a = torch.randn((B * K, M), dtype=ori_dtype, device=device, requires_grad=False)
        b = torch.randn((B * K, N), dtype=ori_dtype, device=device, requires_grad=False)
        temp_ptr = torch.ops.primus_turbo_cpp_extension.init_grouped_gemm(
            torch.tensor(B, dtype=torch.int64, device=device)
        )
        out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_variable_k(
            a, b, seg_lens, True, False, temp_ptr
        )
        out_ref = grouped_gemm_variable_k_ref(a.clone(), b.clone(), seg_lens.clone())
        print(out.shape)
        print(out_ref.shape)
        # print("TN", cosine_similarity(out, out_ref), compute_snr(out, out_ref))
