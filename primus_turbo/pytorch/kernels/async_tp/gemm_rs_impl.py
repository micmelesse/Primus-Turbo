import torch
import torch.distributed.distributed_c10d as c10d

from triton_dist.kernels.amd.gemm_reduce_scatter import (
    matmul_fuse_scatter,
)

from .amd_symmetric_memory import get_amd_symm_mem_workspace


def _blockwise_fused_matmul_scatter_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    *,
    output: torch.Tensor,
    out_dtype: torch.dtype
):

    M = input.shape[0]
    N = weight.shape[0]

    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()

    p2p_workspace_size_req = M * N * input.element_size()
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    scatter_bufs = [symm_mem.get_buffer(i, [M, N], out_dtype) for i in range(num_ranks)]
    scatter_bufs_ptr = torch.tensor(
        [t.data_ptr() for t in scatter_bufs],
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    symm_mem.barrier()

    matmul_fuse_scatter(
        input, weight, scatter_bufs_ptr, rank, num_ranks, transpose_weight=False
    )

    symm_mem.barrier()

    scatter_out = scatter_bufs[rank][:M]
    if output is not None:
        output.copy_(scatter_out)
    return scatter_out
