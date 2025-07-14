from functools import lru_cache

import torch
import torch.distributed.distributed_c10d as c10d
import triton
from triton_dist.kernels.amd.gemm_reduce_scatter import matmul_fuse_scatter

from primus_turbo.triton.reduce.reduce_kernel import kernel_consumer_reduce_async_tp

from .amd_symmetric_memory import get_amd_symm_mem_workspace


def ring_reduce_after_scatter(
    rank,
    num_ranks,
    reduce_op,
    scatter_out,  # [M, N]
    output,
    stream,
):
    M, N = scatter_out.shape
    M_per_rank = M // num_ranks
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=scatter_out.dtype, device=scatter_out.device)

    REDUCE_AVG = True if reduce_op == "avg" else False
    grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]),)
    with torch.cuda.stream(stream):
        kernel_consumer_reduce_async_tp[grid](
            scatter_out,
            output,
            M_per_rank,
            N,
            rank=rank,
            num_ranks=num_ranks,
            REDUCE_AVG=REDUCE_AVG,
            BLOCK_SIZE=2048,
            num_warps=2,
        )

    return output


def _tiled_fused_matmul_scatter_out_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,
    reduce_op: str,
    *,
    output: torch.Tensor,
    rs_output: torch.Tensor,
    out_dtype: torch.dtype,
    stream: torch.cuda.Stream
):

    M = input.shape[0]
    N = weight.shape[0]

    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()

    p2p_workspace_size_req = M * N * input.element_size()
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    scatter_bufs = [symm_mem.get_buffer(i, [M, N], out_dtype) for i in range(num_ranks)]
    scatter_bufs_ptr = get_scatter_buf_ptrs((t.data_ptr() for t in scatter_bufs))

    symm_mem.barrier()
    matmul_fuse_scatter(input, weight, scatter_bufs_ptr, rank, num_ranks, transpose_weight=False)
    symm_mem.barrier()

    scatter_out = scatter_bufs[rank][:M]

    rs_output = ring_reduce_after_scatter(
        rank,
        num_ranks,
        reduce_op,
        scatter_out,  # [M, N]
        rs_output,
        stream,
    )

    if output is not None:
        output.copy_(scatter_out)
    return rs_output


@lru_cache
def get_scatter_buf_ptrs(scatter_bufs_ptr_cpu):
    return torch.tensor(
        list(scatter_bufs_ptr_cpu),
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
