import torch

if __name__ == "__main__":
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

    out = torch.ops.primus_turbo_cpp_extension.rmsnorm_fwd(a, b, c, seg_lens, False, True)
    # print(out.shape)
