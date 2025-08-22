from typing import Optional

from primus_turbo.pytorch.ops.moe import deepep_dispatch_combine

try:
    from primus_turbo.pytorch.ops.moe import mori_dispatch_combine

    HAVE_MORI_BACKEND = True
except ImportError:
    HAVE_MORI_BACKEND = False


def fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    use_cuda_num_token_per_expert: bool = True,
    num_use_cus: int = 64,
    backend_type: str = "deepep",
):
    """Perform fused dispatch operation if deep_ep is available.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        token_indices: Token routing indices [num_tokens, topk]
        token_probs: Token routing probabilities [num_tokens, topk]
        num_experts: Number of experts
        group: Process group
        previous_event: Previous CUDA event
        backend_type: use deepep or mori

    Returns:
        Result of FusedDispatch
    """
    if backend_type == "deepep":
        FusedDispatch = deepep_dispatch_combine.FusedDispatch
    elif backend_type == "mori":
        if not HAVE_MORI_BACKEND:
            raise NotImplementedError("Not Found mori, please install mori.")
        FusedDispatch = mori_dispatch_combine.FusedDispatch
    else:
        raise ValueError("fused_dispatch only support deepep or mori")

    return FusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event,
        use_cuda_num_token_per_expert,
        num_use_cus,
    )


def fused_combine(
    x,
    group,
    handle,
    previous_event=None,
    num_use_cus: Optional[int] = 64,
    backend_type: str = "deepep",
):
    """Perform fused combine operation if deep_ep is available.

    Args:
        x: Input tensor
        group: Process group
        handle: Communication handle
        previous_event: Previous CUDA event
        num_use_cus: number of cus of deepep and mori

    Returns:
        Result of FusedCombine
    """
    if backend_type == "deepep":
        FusedCombine = deepep_dispatch_combine.FusedCombine
    elif backend_type == "mori":
        if not HAVE_MORI_BACKEND:
            raise NotImplementedError("Not Found mori, please install mori.")
        FusedCombine = mori_dispatch_combine.FusedCombine
    else:
        raise ValueError("fused_combine only support deepep or mori")

    return FusedCombine.apply(x, group, handle, previous_event, num_use_cus)
