from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

import pyrocshmem
import torch
import torch.distributed.distributed_c10d as c10d
import triton
import triton.language as tl
from hip import hip
from triton_dist.kernels.amd.common_ops import load_acquire_system, thread_idx
from triton_dist.utils import HIP_CHECK
from triton_dist.kernels.amd.gemm_reduce_scatter import matmul_fuse_scatter

from primus_turbo.triton.reduce.reduce_kernel import kernel_consumer_reduce_async_tp
from .amd_symmetric_memory import get_amd_symm_mem_workspace


@triton.jit
def batch_wait_eq_sys(rank, barrier_ptr, n_elements: tl.constexpr, value):
    barrier_ptr = barrier_ptr.to(tl.pointer_type(tl.int32))
    tid = thread_idx(axis=0)
    if tid < n_elements and tid != rank:
        while load_acquire_system(barrier_ptr + tid) != value:
            pass

    tl.debug_barrier()


def wait_all_gather(rank, num_ranks, barrier_ptr: int, value):
    batch_wait_eq_sys[(1,)](rank, barrier_ptr, num_ranks, value)


@lru_cache
def get_barriers_tensors(group_name: str, num_splits: int) -> List[torch.Tensor]:
    group = c10d._resolve_process_group(group_name)
    m_chunk_num = group.size() * num_splits
    barriers = pyrocshmem.hipipc_create_tensor_list(group, [m_chunk_num], torch.int32)
    barriers[group.rank()].fill_(0)
    return barriers


@lru_cache
def get_one_tensor() -> torch.Tensor:
    one = torch.ones((1,), dtype=torch.int32, device="cuda")
    return one


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


_local_mm_out_bufs_cache = {}

def get_local_mm_out_bufs(M, N, num_splits, dtype, device):
    key = (M, N, num_splits, str(dtype), str(device))
    if key not in _local_mm_out_bufs_cache:
        _local_mm_out_bufs_cache[key] = [
            torch.empty((M, N // num_splits), dtype=dtype, device=device)
            for _ in range(num_splits)
        ]
    return _local_mm_out_bufs_cache[key]

@lru_cache
def get_gemm_events(num_splits):
    return [torch.cuda.Event() for _ in range(num_splits)]


@lru_cache
def get_comm_events(num_splits):
    return [torch.cuda.Event() for _ in range(num_splits)]

@lru_cache
def get_reduce_events(num_splits):
    return [torch.cuda.Event() for _ in range(num_splits)]


def _pipeline_matmul_scatter_out_impl(
    mm_out_op: torch._ops.OpOverload,
    input: torch.Tensor,
    weight: torch.Tensor,
    group_name: str,    
    reduce_op: str,
    num_splits: int,
    enable_sdma: bool,
    comm_stream_pool: List[torch.cuda.Stream],
    gemm_stream_pool: List[torch.cuda.Stream],
    reduce_stream_pool: List[torch.cuda.Stream],
    copy_stream_pool: List[torch.cuda.Stream],
    output: torch.Tensor,
    rs_output: torch.Tensor,
    out_dtype: torch.dtype
):
    M = input.shape[0]
    N = weight.shape[1]
    group = c10d._resolve_process_group(group_name)
    rank = group.rank()
    num_ranks = group.size()

    if enable_sdma:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU
    else:
        hip_memcpy_kind = hip.hipMemcpyKind.hipMemcpyDeviceToDevice

    p2p_workspace_size_req = M * N * input.element_size()
    symm_mem = get_amd_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)

    # num_rank x (num_splits, num_ranks)
    barrier_tensors = get_barriers_tensors(group_name, num_splits)
    one = get_one_tensor()
    barrier_ptrs = [t.data_ptr() for t in barrier_tensors]

    scatter_bufs = [symm_mem.get_buffer(i, [M * num_splits, N // num_splits], out_dtype) for i in range(num_ranks)]
    scatter_bufs_ptr = get_scatter_buf_ptrs((t.data_ptr() for t in scatter_bufs))

    m_per_rank = M // num_ranks
    n_per_chunk = N // num_splits
    data_elem_size = torch.tensor([], dtype=out_dtype).element_size()
    weight_chunks = weight.chunk(num_splits, dim=-1)
    chunked_rs_outputs = get_local_mm_out_bufs(m_per_rank, N, num_splits, out_dtype, input.device)
    local_tensor_buffers = get_local_mm_out_bufs(M, N, num_splits, out_dtype, input.device) 

    gemm_events = get_gemm_events(num_splits)
    comm_events = get_comm_events(num_splits)
    reduce_events = get_reduce_events(num_splits)
    current_stream = torch.cuda.current_stream()
    stream_pool = comm_stream_pool + gemm_stream_pool + reduce_stream_pool + copy_stream_pool

    barrier_tensors[rank].fill_(0)
    symm_mem.barrier()

    for stream in stream_pool:
        stream.wait_stream(current_stream)

    for chunk_idx in range(num_splits):
        gemm_stream = gemm_stream_pool[chunk_idx % len(gemm_stream_pool)]
        reduce_stream = reduce_stream_pool[chunk_idx % len(reduce_stream_pool)]
        copy_stream = copy_stream_pool[chunk_idx % len(copy_stream_pool)]
        gemm_event = gemm_events[chunk_idx]
        comm_event = comm_events[chunk_idx]
        reduce_event = reduce_events[chunk_idx]
        with gemm_stream:
            mm_out_op(
                input, weight_chunks[chunk_idx], out=local_tensor_buffers[chunk_idx]
            )
            gemm_event.record(gemm_stream)

        if output is not None:
            copy_stream.wait_event(gemm_event)
            src_ptr = local_tensor_buffers[chunk_idx].data_ptr()
            dst_ptr = output.data_ptr() + chunk_idx * n_per_chunk * data_elem_size
            copy_width = n_per_chunk * data_elem_size  # bytes to copy per row
            dst_pitch = N * data_elem_size
            HIP_CHECK(
                hip.hipMemcpy2DAsync(
                    dst_ptr, dst_pitch, # dst pointer + row stride
                    src_ptr, copy_width, # src pointer + row stride
                    copy_width, M,   # width, height 
                    hip_memcpy_kind, copy_stream.cuda_stream,
                )
            )

        rank_orders = [(i + chunk_idx) % num_ranks for i in range(num_ranks)]
        for idx, remote_rank in enumerate(rank_orders):
            comm_stream = comm_stream_pool[idx % len(comm_stream_pool)]
            comm_stream.wait_event(gemm_event)

            M_dst_start_pos = M * chunk_idx + m_per_rank * rank
            M_src_start_pos = remote_rank * m_per_rank
            dst_ptr = (
                scatter_bufs[remote_rank].data_ptr()
                + M_dst_start_pos * n_per_chunk * data_elem_size
            )

            src_ptr = (
                local_tensor_buffers[chunk_idx].data_ptr()
                + M_src_start_pos * n_per_chunk * data_elem_size
            )

            nbytes = m_per_rank * n_per_chunk * data_elem_size
            HIP_CHECK(
                hip.hipMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    nbytes,
                    hip_memcpy_kind,
                    comm_stream.cuda_stream,
                )
            )
            HIP_CHECK(
                hip.hipMemcpyAsync(
                    barrier_ptrs[remote_rank] + chunk_idx * num_ranks * 4 + rank * 4,
                    one.data_ptr(),
                    one.nbytes,
                    hip_memcpy_kind,
                    comm_stream.cuda_stream,
                )
            )
            comm_event.record(comm_stream)
        
        
        reduce_stream.wait_event(comm_event)
        with reduce_stream:
            wait_all_gather(rank, num_ranks, barrier_ptrs[rank] + chunk_idx * num_ranks * 4, 1)
            scatter_out = scatter_bufs[rank][(M * chunk_idx):(M * (chunk_idx + 1))]
            
            ring_reduce_after_scatter(
                rank,
                num_ranks,
                reduce_op,
                scatter_out,  # [M, N]
                chunked_rs_outputs[chunk_idx],
                reduce_stream,
            )
            reduce_event.record(reduce_stream)
        
        
        if rs_output is not None:
            copy_stream.wait_event(reduce_event)
            with copy_stream:
                src_ptr = chunked_rs_outputs[chunk_idx].data_ptr()
                dst_ptr = rs_output.data_ptr() + chunk_idx * n_per_chunk * data_elem_size
                copy_width = n_per_chunk * data_elem_size
                dst_pitch = N * data_elem_size
                HIP_CHECK(
                    hip.hipMemcpy2DAsync(
                        dst_ptr, dst_pitch, # dst pointer + row stride
                        src_ptr, copy_width, # src pointer + row stride
                        copy_width, m_per_rank,   # width, height 
                        hip_memcpy_kind, copy_stream.cuda_stream,
                    )
                )

    for stream in stream_pool:
        current_stream.wait_stream(stream)

    return  rs_output if rs_output is not None else torch.cat(chunked_rs_outputs, dim=-1)