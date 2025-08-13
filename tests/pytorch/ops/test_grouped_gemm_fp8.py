###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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

    from primus_turbo.pytorch.core.float8 import float8_e4m3
    from tests.pytorch.ref.gemm_ref import grouped_gemm_ref

    torch.manual_seed(1234)

    from tests.test_utils import compute_snr

    def calc_scale_and_scale_inv(x: torch.Tensor, fp8_max: float, row_wise: bool = True):
        if row_wise:
            if x.dim() == 2:
                amax = x.abs().amax(dim=1, keepdim=True)
            elif x.dim() == 3:
                amax = x.abs().amax(dim=2, keepdim=True)
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        else:
            if x.dim() == 2:
                amax = x.abs().amax(dim=0, keepdim=True)
            elif x.dim() == 3:
                amax = x.abs().amax(dim=(0, 1), keepdim=True)
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

        scale = torch.full_like(amax, fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
        scale_inv = 1.0 / scale

        return scale, scale_inv

    def compute_group_offs(group_lens: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.tensor([0], device=group_lens.device, dtype=group_lens.dtype), group_lens.cumsum(0)]
        )

    NT = True
    NN = True
    TN = True
    ori_dtype = torch.bfloat16
    device = "cuda"

    if NT is True:
        B = 4
        M = 512
        N = 1024
        K = 2048
        # row-wise
        a = torch.randn((B * M, K), dtype=ori_dtype, device=device, requires_grad=False)
        a_ref = a.clone()
        a_scale, a_scale_inv = calc_scale_and_scale_inv(a, torch.finfo(float8_e4m3).max)
        print(a_scale.shape, a_scale_inv.shape)
        a_fp8 = torch.ops.primus_turbo_cpp_extension.fp8_quantize_row_col(a, a_scale, True)
        a_fp8_ref = a * a_scale

        b = torch.randn((B, N, K), dtype=ori_dtype, device=device, requires_grad=False)
        b_ref = b.clone()
        b_scale, b_scale_inv = calc_scale_and_scale_inv(b, torch.finfo(float8_e4m3).max)
        print(b_scale.shape, b_scale_inv.shape)
        b_fp8 = torch.ops.primus_turbo_cpp_extension.fp8_quantize_row_col(b, b_scale.flatten(), True)
        b_fp8_ref = b * b_scale

        seg_lens = torch.zeros([B], dtype=torch.int64, device=device)
        seg_lens[0] = 256
        seg_lens[1] = 768
        seg_lens[2] = 768
        seg_lens[3] = 256
        out_ref = grouped_gemm_ref(a_ref, b_ref, seg_lens, True)
        print(out_ref)
        group_lens_offs = compute_group_offs(seg_lens)
        print(group_lens_offs)
        group_lens_offs2 = torch.ops.primus_turbo_cpp_extension.grouped_gemm_compute_offs(seg_lens)

        print(group_lens_offs2)

        # out = torch.ops.primus_turbo_cpp_extension.grouped_gemm_fp8(
        #     a_fp8, b_fp8, seg_lens, group_lens_offs, transA=False, transB=True
        # )
        # # print(out.shape)
        # # scale1 = a_scale_inv[: seg_lens[0]] * b_scale_inv[0].T
        # # out_1 = out[: seg_lens[0]]
        # # out_o = out_1 * scale1
        # # print(out_o)
        # out_o2 = torch.ops.primus_turbo_cpp_extension.grouped_gemm_fp8_dequant(
        #     out, seg_lens, a_scale_inv, b_scale_inv
        # )
        # print(out_o2)
        # print(compute_snr(out_ref, out_o2))
