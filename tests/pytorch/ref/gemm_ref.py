import torch


def grouped_gemm_ref(a, b, seg_lens, trans_b=True):
    seg_lens = seg_lens.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(seg_lens):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


def grouped_gemm_variable_k_ref(a, b, seg_lens, trans_a=True, trans_b=False):
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."
    seg_lens = seg_lens.cpu().numpy()
    B = len(seg_lens)
    M = a.shape[1]
    N = b.shape[1]
    out = torch.zeros((B, M, N), dtype=a.dtype, device="cuda", requires_grad=False)
    start = 0
    for i, size in enumerate(seg_lens):
        a_tmp = a[start : start + size, :].t()
        b_tmp = b[start : start + size, :]
        out_tmp = a_tmp @ b_tmp
        out[i] = out_tmp
        start += size
    return out
