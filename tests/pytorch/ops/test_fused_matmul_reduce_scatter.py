from typing import Dict, List
import random
import numpy as np

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import primus_turbo.pytorch as pt
from tests.test_utils import get_tolerances


def native_torch_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    scatter_dim: int,
    reduce_op: str,
    group: dist.group.WORLD,
):
    output = torch.matmul(A, B)

    output_flat = output.movedim(scatter_dim, 0)
    rs_out = output.new_empty(
        output_flat.shape[0] // group.size(),
        *output_flat.shape[1:],
    )
    assert (output_flat - output).abs().max() == 0.0
    if reduce_op == "avg":
        reduce_op = ReduceOp.AVG
    elif reduce_op == "sum":
        reduce_op = ReduceOp.SUM
    else:
        raise ValueError(
            f"Only avg or sum be supported, but provided reduce_op is {reduce_op}"
        )
    torch.distributed.reduce_scatter_tensor(rs_out, output_flat, reduce_op, group)


    world_size = group.size()
    rank = group.rank()
    # Step 1: all_gather all output_flat from all ranks
    gathered = [
        torch.empty_like(output_flat) for _ in range(world_size)
    ]
    torch.distributed.all_gather(gathered, output_flat, group=group)

    # Step 2: sum over gathered tensors (simulate reduce)
    reduced = sum(gathered)  # elementwise add over same shape tensors

    # Step 3: chunk along dim=0
    chunks = torch.chunk(reduced, world_size, dim=0)

    # Return the chunk corresponding to this rank
    print(f"xxxxxxxxxxxdiff 0: {(chunks[rank] - rs_out).abs().max()}")
    rs_out_flat = rs_out.movedim(0, scatter_dim)
    assert (rs_out_flat - rs_out).abs().max() == 0.0
    # torch.save(output_flat, f"output_flat_{torch.distributed.get_rank()}.pt")
    # torch.save(rs_out_flat, f"rs_out_flat_{torch.distributed.get_rank()}.pt")
    return rs_out_flat


@instantiate_parametrized_tests
class FusedMatmulReduceScatterTestBase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    # @skip_if_lt_x_gpu(2)
    # @parametrize("scatter_dim", [0, 1])
    # @parametrize("reduce_op", ["avg", "sum"])
    # @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    # def test_fused_matmul_reduce_scatter(
    #     self, scatter_dim: int, reduce_op: str, dtype: torch.dtype
    # ) -> None:
    #     self._init_process()

    #     BATCH = 8
    #     M = 64
    #     N = 256
    #     K = 512
    #     group = dist.group.WORLD
    #     rank = self.rank

    #     torch.manual_seed(42 + rank)
    #     A = torch.rand(BATCH, M, K, dtype=dtype, device="cuda")
    #     B = torch.rand(K, N, dtype=dtype, device="cuda")

    #     rs_output_0 = native_torch_matmul_reduce_scatter(
    #         A, B, scatter_dim, reduce_op, group
    #     )
    #     rs_output_1 = pt.ops.fused_matmul_reduce_scatter(
    #         A,
    #         B,
    #         layout="NN",
    #         reduce_op=reduce_op,
    #         scatter_dim=scatter_dim,
    #         group_name=group.group_name,
    #     )

    #     # assert (
    #     #     rs_output_0.stride() == rs_output_1.stride()
    #     # ), f"rs_out_0 stride {rs_output_0.stride()} is not equal to rs_out_1 stride {rs_output_1.stride()}"
    #     print(f"rs_out_0:{rs_output_0.shape}, rs_out_1:{rs_output_1.shape}")
    #     torch.testing.assert_close(rs_output_0, rs_output_1, **get_tolerances(dtype))

    @skip_if_lt_x_gpu(2)
    @parametrize("M,K,N", [(8192, 8192, 8192), (8192, 8192, 14336)])
    @parametrize("batch_size", [1, 4])
    @parametrize("dtype", [torch.float32])
    def test_llama3_70b_fused_matmul_reduce_scatter(
        self, dtype, batch_size, M, K, N
    ) -> None:
        self._init_process()
        group = dist.group.WORLD
        rank = self.rank
        scatter_dim = 0
        reduce_op = "sum"

        torch.manual_seed(42 + rank)

        A = torch.rand(batch_size * M, K // group.size(), dtype=dtype, device="cuda")
        B = torch.rand(K // group.size(), N, dtype=dtype, device="cuda")

        rs_output_0 = native_torch_matmul_reduce_scatter(
            A, B, scatter_dim, reduce_op, group
        )

        torch.save(rs_output_0, f"native_rs_out_{torch.distributed.get_rank()}.pt")
        rs_output_1 = pt.ops.fused_matmul_reduce_scatter(
            A,
            B,
            layout="NN",
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group.group_name,
        )

        assert rs_output_0.stride() == rs_output_1.stride()
        diff_mask = (rs_output_0 != rs_output_1)
        print(f"diff_mask: {diff_mask.sum()}, rs_output_0:{rs_output_0}, rs_output_1:{rs_output_1}")
        assert torch.allclose(rs_output_0, rs_output_1, atol=1e-2, rtol=1e-2)
        # torch.testing.assert_close(rs_output_0, rs_output_1, **get_tolerances(dtype))


if __name__ == "__main__":
    run_tests()
