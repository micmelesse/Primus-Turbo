import torch
import torch.nn as nn

import primus_turbo.pytorch as turbo

from .basic_llama import (
    BasicAttention,
    BasicMLP,
    BasicTransformerBlock,
    LlamaBasicModel,
)
from .rope import apply_rotary_emb


class TurboAttention(BasicAttention):
    def __init__(self, config):
        super().__init__(config)
        self.sdpa = turbo.modules.TurboAttention(causal=True)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Attn
        attn_output = self.sdpa(xq, xk, xv)
        attn_output = attn_output.view(bs, seqlen, -1)

        return self.wo(attn_output)


class TurboTransformerBlock(BasicTransformerBlock):
    def __init__(self, layer_id: int, config):
        super().__init__(layer_id, config)
        self.attention = TurboAttention(config)
        self.mlp = BasicMLP(config)


class LlamaTurboModel(LlamaBasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Only test 4 layers for now
        self.layers = nn.ModuleList([TurboTransformerBlock(layer_id, config) for layer_id in range(4)])
        self.init_weights()
