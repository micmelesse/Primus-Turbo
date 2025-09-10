###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from typing import Optional, Tuple
from functools import lru_cache
from primus_turbo.pytorch.kernels.attention.attention_csrc_impl import (
    attention_aiter_csrc_backward_impl,
    attention_aiter_csrc_forward_impl,
)

@lru_cache
def get_attention_cp_p2p_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
    attn_helper = AttentionCPP2PHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)
    return attn_helper

class AttentionCPP2PHelper:
    comm_stream = Optional[torch.cuda.Stream]

    def __init__(self, cp_p2p_group, b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
        self.cp_group = cp_p2p_group
        self.seq_dim = seq_dim
        # kv_recv buffer
        self.buffer_ = torch.empty((b * (s // n) * h_kv * (d_qk + d_v),), device="cuda")
        self.dkv_buffer = torch.empty_like(self.buffer_)

        # global output and softmax_lse
        self.output_ = Optional[torch.tensor]
        self.softmax_lse_ = Optional[torch.tensor]

        if AttentionCPP2PHelper.comm_stream is None:
            AttentionCPP2PHelper.comm_stream = torch.cuda.Stream()


    def rotate_kv_buffer(chunk_kv: torch.Tensor):
        """ Ring attntion rotate local KV tensor around CP group

        Args:
            send_tensor (torch.Tensor): local kv in current round
        """
        ...

    def rotate_dkv_buffer(chunk_dkv: torch.Tensor):
        """ Rotate dkv among backward

        Args:
            chunk_dkv (torch.Tensor): local dkv in current round
        """

    def merge_local_softmax_lse(self, chunk_softmax_lse: torch.Tensor, chunk_out: torch.Tensor, partial: bool):
        """ Merge local chunk softmax_lse to the global

        Args:
            chunk_softmax_lse (torch.Tensor): local chunk attenetion softmax_lse
            chunk_out (torch.Tensor): local chunk attention out
        """
        ...

    def get_buffer(self):
        """ Return next buffer
        """
        return self.buffer_

    def get_global_result(self):
        return self.output_, self.softmax_lse_.contiguous()


class AttentionCKFunctionCPP2P(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        cp_group,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        assert bias is None
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        n = cp_group.size()
        b, s, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s * n

        # bshd only
        seq_dim = 1
        attn_helper = get_attention_cp_p2p_helper(cp_group, b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

        # cp_streams = [torch.cuda.current_stream(), AttentionCPP2PHelper.cp_stream]

        local_chunk_kv = torch.cat(k, v, dim=-1).contiguous()
        rank = cp_group.rank()
        comm_stream = AttentionCPP2PHelper.comm_stream
        comm_done_event = torch.cuda.Event()

        for i in range(n):
            if i < n:
                with torch.cuda.stream(comm_stream):
                    attn_helper.rotate_kv_buffer(local_chunk_kv)
                    comm_done_event.record(comm_stream)

            if i == 0:
                local_q, local_k, local_v, partial = q, k, v, False
            else:
                torch.cuda.current_stream().wait_event(comm_done_event)
                local_chunk_kv = attn_helper.get_buffer()
                local_k, local_v = local_chunk_kv[: k.numel()].rehape(k.shape), local_chunk_kv[v.numel():].rehape(v.shape)
                if not causal:
                    local_q, local_k, local_v, partial = q, k, v, False
                elif i <= rank:
                    local_q, local_k, local_v, partial = (
                        q,
                        local_k.chunk(2, dim=seq_dim)[0],
                        local_v.chunk(2, dim=seq_dim)[0],
                        False,
                    )
                else:
                    local_q, local_k, local_v, partial = q.chunk(2, dim=seq_dim)[1], local_k, local_v, True

            # do attention
            if d_qk % 8 != 0:
                local_q = torch.nn.functional.pad(local_q, [0, 8 - d_qk % 8])
                local_k = torch.nn.functional.pad(local_k, [0, 8 - d_qk % 8])
            if d_v % 8 != 0:
                local_v = torch.nn.functional.pad(local_v, [0, 8 - d_v % 8])

            out_padded, softmax_lse, S_dmask, rng_state = attention_aiter_csrc_forward_impl(
                local_q,
                local_k,
                local_v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=True,
                return_softmax=return_softmax and dropout_p > 0,
            )

            output_local_chunk = out_padded[..., :d_v]

            attn_helper.merge_local_softmax_lse(softmax_lse, output_local_chunk, partial)

        out, softmax_lse = attn_helper.get_global_result()

        if is_grad:
            ctx.save_for_backward(
                q, k, v, out, softmax_lse, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.d_qk = d_qk
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
            ctx.attn_helper = attn_helper
            ctx.cp_group = cp_group
            ctx.seq_dim = seq_dim

        return out
    
    
    @staticmethod
    def backward(ctx, dout, *args):
        (
            q, k, v, out, softmax_lse, rng_state
        ) = ctx.saved_tensors
        attn_helper: AttentionCPP2PHelper = ctx.attn_helper

        dq, dk, dv = [torch.zeros_like(x) for x in (q, k, v)]

        n = ctx.cp_group.size()
        rank = ctx.cp_group.rank()
        seq_dim = ctx.seq_dim
        comm_stream = AttentionCPP2PHelper.comm_stream

        local_chunk_kv = torch.cat(k, v, dim=-1).contiguous()
        local_chunk_dkv = None
        comm_done_event = torch.cuda.Event()
        dbias = None
        d_qk = ctx.d_qk
        d_v = local_out.size(3)

        for i in range(n):
            # exchange kv
            if i < n:
                with torch.cuda.stream(comm_stream):
                    attn_helper.rotate_kv_buffer(local_chunk_kv)
                    comm_done_event.record(comm_stream)

            # init local q, k, v, out, dout, lse
            if i == 0:
                local_q, local_k, local_v, partial = q, k, v, False
            else:
                torch.cuda.current_stream().wait_event(comm_done_event)
                local_chunk_kv = attn_helper.get_buffer()
                local_k, local_v = local_chunk_kv[: k.numel()].rehape(k.shape), local_chunk_kv[v.numel():].rehape(v.shape)
                if not ctx.causal:
                    local_q, local_k, local_v, partial = q, k, v, False
                elif i <= rank:
                    local_q, local_k, local_v, local_out, local_dout, local_lse = (
                        q,
                        local_k.chunk(2, dim=seq_dim)[0],
                        local_v.chunk(2, dim=seq_dim)[0],
                        out,
                        dout,
                        softmax_lse,
                    )
                else:
                    local_q, local_k, local_v, local_out, local_dout, local_lse = (
                        q.chunk(2, dim=seq_dim)[1],
                        k,
                        v,
                        out.chunk(2, dim=seq_dim)[1],
                        dout.chunk(2, dim=seq_dim)[1],
                        softmax_lse.chunk(2, dim=seq_dim)[1].contiguous(),
                    )
            
            # do attention (TODO: add casual behavier)
            local_out_padded = local_dout

            if d_v % 8 != 0:
                dout_padded = torch.nn.functional.pad(local_dout, [0, 8 - d_v % 8])
            if d_qk != d_v:
                local_v = torch.nn.functional.pad(local_v, [0, d_qk - d_v])
                local_out_padded = torch.nn.functional.pad(local_out, [0, d_qk - d_v])
                local_dout = torch.nn.functional.pad(local_dout, [0, d_qk - d_v])

            attention_aiter_csrc_backward_impl(
                dout_padded,
                local_q,
                local_k,
                local_v,
                local_out_padded,
                local_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                int(ctx.window_size[0]),
                int(ctx.window_size[1]),
                ctx.bias,
                ctx.alibi_slopes,
                ctx.deterministic,
                rng_state,
                ctx.is_v3_atomic_fp32,
                ctx.how_v3_bf16_cvt,
            )
            dq = dq[..., :d_qk]  # We could have padded the head dimension
            dk = dk[..., :d_qk]
            dv = dv[..., :d_v]

            # exchange dkv and partial update
            

                



