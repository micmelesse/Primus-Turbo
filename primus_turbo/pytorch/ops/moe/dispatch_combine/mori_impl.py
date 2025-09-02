import mori
import torch
import torch.distributed as dist
from mori.ops import (
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
)

_mori_initialized: bool = False
_handle: EpDispatchCombineOp = None
_handle_num_cus: int = 64
_num_gpus_per_node = 8


# trick flag will be removed
ENABLE_SYNC = False


def mori_initialize(group: dist.ProcessGroup):
    global _mori_initialized

    if not _mori_initialized:
        ranks = list(range(group.size()))
        gloo_group = dist.new_group(ranks, backend="cpu:gloo,cuda:nccl")
        mori.shmem.shmem_torch_process_group_init(gloo_group.group_name)
        _mori_initialized = True


def set_num_cus(num_cus: int):
    global _handle_num_cus
    _handle_num_cus = num_cus


def get_mori_dispatch_combine(
    group, hidden_states: torch.Tensor, indices: torch.Tensor, num_experts: int
) -> EpDispatchCombineOp:
    global _handle, _handle_group, _num_gpus_per_node, _handle_num_cus

    world_size = group.size()
    rank = group.rank()

    assert len(hidden_states.shape) == 2, hidden_states.shape
    assert len(indices.shape) == 2
    assert num_experts % world_size == 0

    num_tokens, hidden_size = hidden_states.shape
    _, router_topk = indices.shape

    num_nodes = world_size // _num_gpus_per_node
    num_local_experts = num_experts // world_size

    cfg = EpDispatchCombineConfig(
        data_type=hidden_states.dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_size,
        scale_dim=32,
        scale_type_size=4,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        warp_num_per_block=16,
        block_num=_handle_num_cus,
        max_token_type_size=hidden_states.dtype.itemsize,
        kernel_type=(
            EpDispatchCombineKernelType.InterNode if num_nodes > 1 else EpDispatchCombineKernelType.IntraNode
        ),
    )

    if _handle is None or _handle.config != cfg:
        _handle = EpDispatchCombineOp(cfg)
        _handle_group = group

    return _handle


def make_deepep_topken_indices(
    token_indices: torch.Tensor,
    rank: int,
    num_ranks: int,
    num_experts: int,
    recv_num_token: torch.Tensor,
    use_cuda_num_token_per_expert: bool = True,
):
    num_recv_tokens, num_topk = token_indices.shape
    num_recv_tokens_rows = torch.arange(num_recv_tokens, device=token_indices.device)

    # 1. mapping recv topk_ids to token-expert map
    routing_map = torch.zeros((num_recv_tokens, num_experts), dtype=torch.long, device=token_indices.device)

    routing_map[num_recv_tokens_rows.unsqueeze(1), token_indices] = torch.ones(
        1, device=routing_map.device, dtype=routing_map.dtype
    )

    num_local_expert = num_experts // num_ranks
    recv_expert_begin = rank * num_local_expert
    recv_expert_end = (rank + 1) * num_local_expert
    local_expert_id = torch.arange(
        recv_expert_begin, recv_expert_end, dtype=torch.long, device=token_indices.device
    )

    if not ENABLE_SYNC:
        routing_map_slice = num_recv_tokens_rows.repeat_interleave(num_experts).view(
            num_recv_tokens, num_experts
        )
        routing_map_slice_ge_recv_num_token = routing_map_slice >= recv_num_token
        routing_map.masked_fill_(routing_map_slice_ge_recv_num_token, 0)
    routing_map = torch.index_select(routing_map, dim=1, index=local_expert_id)

    # fill -1 when the expert id of token_inces not in local_expert range
    mask = (token_indices < recv_expert_begin) | (token_indices >= recv_expert_end)
    if not ENABLE_SYNC:
        mask_slice = num_recv_tokens_rows.repeat_interleave(num_topk).view(num_recv_tokens, num_topk)
        mask_slice_ge_recv_num_token = mask_slice >= recv_num_token
        mask = mask | mask_slice_ge_recv_num_token
    recv_token_indices = token_indices - recv_expert_begin
    recv_token_indices.masked_fill_(mask, -1)

    num_token_per_expert = routing_map.sum(dim=0)

    if not use_cuda_num_token_per_expert:
        num_token_per_expert = num_token_per_expert.cpu()

    return recv_token_indices, num_token_per_expert


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        use_cuda_num_token_per_expert: bool = True,
        num_use_cus: int = 64,
    ):
        """Forward pass of fused dispatch."""
        global _handle_num_cus
        _handle_num_cus = num_use_cus

        mori_initialize(group)

        op = get_mori_dispatch_combine(group, x, token_indices, num_experts)

        int32_token_indices = token_indices.to(torch.int32)
        recv_x, recv_token_probs, _, recv_token_indices, recv_num_token = op.dispatch(
            x, token_probs, None, int32_token_indices
        )
        handle = (op, recv_token_indices, int32_token_indices)

        if ENABLE_SYNC:
            recv_x = recv_x[:recv_num_token]
            recv_token_indices = recv_token_indices[:recv_num_token]
            recv_token_probs = recv_token_probs[:recv_num_token]

        recv_token_indices, num_tokens_per_expert = make_deepep_topken_indices(
            recv_token_indices,
            group.rank(),
            group.size(),
            num_experts,
            recv_num_token,
            use_cuda_num_token_per_expert=use_cuda_num_token_per_expert,
        )

        ctx.handle = handle

        return (
            recv_x.to(copy=True),
            recv_token_indices,
            recv_token_probs.to(copy=True),
            num_tokens_per_expert,
            handle,
        )

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle):
        """Backward pass of fused dispatch."""
        op, recv_token_indices, _ = ctx.handle
        grad_x, grad_token_probs = op.combine(
            grad_output.contiguous(), grad_token_probs.float(), recv_token_indices
        )
        return grad_x, None, grad_token_probs, None, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, previous_event=None, num_use_cus: int = 64):
        """Forward pass of fused combine."""
        global _handle_num_cus
        _handle_num_cus = num_use_cus

        mori_initialize(group)

        op, disaptch_indices, _ = handle
        combine_x, _ = op.combine(x, None, disaptch_indices)

        ctx.handle = handle
        return combine_x.to(copy=True), None

    @staticmethod
    def backward(ctx, grad_output, revious_event=None):
        """Backward pass of fused combine."""
        op, _, token_indices = ctx.handle
        # need mori support weight is None
        empty_weight = torch.empty_like(
            grad_output,
            dtype=torch.float32,
            device=grad_output.device,
        )
        grad_x, _, _, _, num_recv_tokens = op.dispatch(
            grad_output.contiguous(), empty_weight, None, token_indices
        )

        if ENABLE_SYNC:
            grad_x = grad_x[:num_recv_tokens]

        return grad_x, None, None, None, None
