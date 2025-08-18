###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import json
import math
from dataclasses import asdict, dataclass
from einops import rearrange, repeat
from pathlib import Path

import torch
# import aiter
from flash_attn import flash_attn_func

import primus_turbo.pytorch as pt

from metrics import (
    cosine_similarity,
    max_abs_error,
    relative_error,
    compute_snr,
)
from utils import (
    DEVICE,
    dump_tensor,
    get_device_name,
    get_device_type,
    get_tensor_name,
    load_tensor,
    merge_excels,
    save_to_excel,
)

FUNC_NAME = "attention"


@dataclass
class AttnModelCfg:
    model_name: str
    seqlen_q: int
    seqlen_kv: int
    num_head_q: int
    num_head_kv: int
    head_dim_qk: int
    head_dim_v: int
    batch_size: int

    def to_string(self):
        return (
            f"{self.model_name}_q{self.seqlen_q}_kv{self.seqlen_kv}_"
            f"hq{self.num_head_q}_hkv{self.num_head_kv}_"
            f"hdqk{self.head_dim_qk}_hdv{self.head_dim_v}_bs{self.batch_size}"
        )


g_model_cfg = [
    AttnModelCfg(
        model_name="llama", seqlen_q=8, seqlen_kv=8, num_head_q=16, num_head_kv=4, head_dim_qk=128, head_dim_v=128, batch_size=1
    ),
]


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    import pdb;pdb.set_trace()
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    # v_re = v.permute(0, 2, 1, 3).contiguous()  # [b, h, s, d]
    # attn_out = torch.matmul(attention_drop, v_re)  # [b, h, t, d]
    # output = attn_out.permute(0, 2, 1, 3).contiguous()  # [b, t, h, d]

    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention(model_config, dtype, seed, backend='FA'):
    torch.manual_seed(seed)
    q_layout = (model_config.batch_size, model_config.seqlen_q, model_config.num_head_q, model_config.head_dim_qk)
    k_layout = (model_config.batch_size, model_config.seqlen_kv, model_config.num_head_kv, model_config.head_dim_qk)
    v_layout = (model_config.batch_size, model_config.seqlen_kv, model_config.num_head_kv, model_config.head_dim_v)
	
    query_cpu = torch.randn(q_layout, dtype=dtype, requires_grad=True)
    key_cpu = torch.randn(k_layout, dtype=dtype, requires_grad=True)
    value_cpu = torch.randn(v_layout, dtype=dtype, requires_grad=True)

    query = query_cpu.detach().to(DEVICE).requires_grad_()
    key = key_cpu.detach().to(DEVICE).requires_grad_()
    value = value_cpu.detach().to(DEVICE).requires_grad_()
	
    out_ref, attn_ref = attention_ref(
		query_cpu,
		key_cpu,
		value_cpu,
		causal=True,
		upcast=False,
	)

    if backend == 'FA':
        out, lse, S_dmask = flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=True,
        )
    else:
        out = pt.ops.attention(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=query.shape[-1] ** (-0.5),
            causal=True,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            backend_type="ck",
        )
        # out, _, S_dmask = aiter.flash_attn_func(
		# 	query,
        #     key,
        #     value,
		# 	dropout_p=0.0,
		# 	causal=True,
		# 	window_size=(-1, -1),
		# 	bias=None,
		# 	alibi_slopes=None,
		# 	deterministic=False,
		# 	return_lse=True,
		# 	return_attn_probs=True,
		# )

    grad_out_cpu = torch.randn_like(out_ref)
    grad_out = grad_out_cpu.detach().to(DEVICE)

    (
        dq,
        dk,
        dv,
    ) = torch.autograd.grad(out, (query, key, value), grad_out)
    (
        dq_ref,
        dk_ref,
        dv_ref,
    ) = torch.autograd.grad(out_ref, (query_cpu, key_cpu, value_cpu), grad_out_cpu)

    return out_ref, dq_ref, dk_ref, dv_ref, out, dq, dk, dv


def benchmark(seed, report_dir_path, load_config_path=None, dump_dir_path=None, backend='FA'):
    import pdb;pdb.set_trace()
    device_type = get_device_type()
    device_name = get_device_name()
    ref_device = "CPU"

    report_dir = Path(report_dir_path) / Path(FUNC_NAME) / Path(backend)
    report_dir.mkdir(parents=True, exist_ok=True)

    if load_config_path is not None:
        with open(load_config_path, "r", encoding="utf-8") as f:
            load_config: dict = json.load(f)
            load_dir = Path(load_config.get("load_dir"))

    if dump_dir_path is not None:
        dump_dir = Path(dump_dir_path) / Path(FUNC_NAME) / Path(backend)
        dump_dir.mkdir(parents=True, exist_ok=True)

    results_with_cpu = []
    results_with_gpu = []

    def item(metric):
        return f"{metric:.3e}" if isinstance(metric, float) else str(metric)

    for config in g_model_cfg:
        for dtype in [torch.float16, torch.bfloat16]:
            out_ref, dq_ref, dk_ref, dv_ref, out, dq, dk, dv = attention(config, dtype, seed, backend)

            result = {
                "func": f"{FUNC_NAME.upper()} ({device_name} vs {ref_device})",
				"backend": backend,
                "dtype": str(dtype).split(".")[-1],
                "config": config.to_string(),
                "RelError(out)": item(relative_error(out_ref, out.cpu())),
                "RelError(dq)": item(relative_error(dq_ref, dq.cpu())),
                "RelError(dk)": item(relative_error(dk_ref, dk.cpu())),
                "RelError(dv)": item(relative_error(dv_ref, dv.cpu())),
                "MaxAbsErr(out)": item(max_abs_error(out_ref, out.cpu())),
                "MaxAbsErr(dq)": item(max_abs_error(dq_ref, dq.cpu())),
                "MaxAbsErr(dk)": item(max_abs_error(dk_ref, dk.cpu())),
                "MaxAbsErr(dv)": item(max_abs_error(dv_ref, dv.cpu())),
                "CosSim(out)": f"{cosine_similarity(out_ref, out.cpu()):.6f}",
                "CosSim(dq)": f"{cosine_similarity(dq_ref, dq.cpu()):.6f}",
                "CosSim(dk)": f"{cosine_similarity(dk_ref, dk.cpu()):.6f}",
                "CosSim(dv)": f"{cosine_similarity(dv_ref, dv.cpu()):.6f}",
				"SNR(out)": f"{compute_snr(out_ref, out.cpu()):.2f}",
				"SNR(dq)": f"{compute_snr(dq_ref, dq.cpu()):.2f}",
				"SNR(dk)": f"{compute_snr(dk_ref, dk.cpu()):.2f}",
				"SNR(dv)": f"{compute_snr(dv_ref, dv.cpu()):.2f}", 
            }
            results_with_cpu.append(result)

            if dump_dir_path is not None:
                dump_file = get_tensor_name(device_type, FUNC_NAME, dtype, config=config)
                output_dict = {
                    'output': out.cpu(),
                    'dq': dq.cpu(),
                    'dk': dk.cpu(),
                    'dv': dv.cpu(),
                }
                dump_tensor(output_dict, dump_dir, dump_file)

            if load_config_path is not None:
                load_file = get_tensor_name(load_config.get("device_type"), FUNC_NAME, dtype, config=config)
                tensors_load = load_tensor(load_dir, load_file)
                out_load = tensors_load['output']
                dq_load = tensors_load['dq']
                dk_load = tensors_load['dk']
                dv_load = tensors_load['dv']
                          
                gpu_result = {
                    "func": f"{FUNC_NAME.upper()} ({device_name} vs {load_config.get('device_name')})",
                    "backend": f"{load_config.get('backend', 'FA')} vs {backend}",
                    "dtype": str(dtype).split(".")[-1],
                    "config": str(config),
					"RelError(out)": item(relative_error(out_load, out.cpu())),
                    "RelError(dq)": item(relative_error(dq_load, dq.cpu())),
                    "RelError(dk)": item(relative_error(dk_load, dk.cpu())),
                    "RelError(dv)": item(relative_error(dv_load, dv.cpu())),
                    "MaxAbsErr(out)": item(max_abs_error(out_load, out.cpu())),
                    "MaxAbsErr(dq)": item(max_abs_error(dq_load, dq.cpu())),
                    "MaxAbsErr(dk)": item(max_abs_error(dk_load, dk.cpu())),
                    "MaxAbsErr(dv)": item(max_abs_error(dv_load, dv.cpu())),
                    "CosSim(out)": f"{cosine_similarity(out_load, out.cpu()):.6f}",
                    "CosSim(dq)": f"{cosine_similarity(dq_load, dq.cpu()):.6f}",
                    "CosSim(dk)": f"{cosine_similarity(dk_load, dk.cpu()):.6f}",
                    "CosSim(dv)": f"{cosine_similarity(dv_load, dv.cpu()):.6f}",
                    "SNR(out)": f"{compute_snr(out_load, out.cpu()):.2f}",
                    "SNR(dq)": f"{compute_snr(dq_load, dq.cpu()):.2f}",
                    "SNR(dk)": f"{compute_snr(dk_load, dk.cpu()):.2f}",
                    "SNR(dv)": f"{compute_snr(dv_load, dv.cpu()):.2f}",  
                }
                results_with_gpu.append(gpu_result)
        results_with_cpu.append({k: "" for k in results_with_cpu[-1].keys()})
        if len(results_with_gpu) > 0:
            results_with_gpu.append({k: "" for k in results_with_gpu[-1].keys()})

    report_with_cpu = report_dir / f"benchmark_{device_type}_{FUNC_NAME}.xlsx"
    report_with_gpu = report_dir / f"benchmark_GPU_{FUNC_NAME}.xlsx"

    save_to_excel(results_with_cpu, report_with_cpu)
    save_to_excel(results_with_gpu, report_with_gpu)

    benchmark_reports = [report_with_cpu]
    if load_config_path and load_config.get("report_path") and Path(load_config.get("report_path")).exists():
        benchmark_reports.append(Path(load_config.get("report_path")))
    if Path(report_with_gpu).exists():
        benchmark_reports.append(report_with_gpu)
    if len(benchmark_reports) >= 2:
        report_path = report_dir / f"benchmark_{FUNC_NAME}.xlsx"
        merge_excels(benchmark_reports, report_path)


if __name__ == "__main__":
    # python eval_attention_accuracy.py --report-dir-path output --backend AITER --dump-dir-path dump
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config-path", default=None, type=str)
    parser.add_argument("--dump-dir-path", default=None, type=str)
    parser.add_argument("--report-dir-path", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--backend", default="FA", type=str, choices=["FA", "AITER"])

    args = parser.parse_args()
    benchmark(args.seed, args.report_dir_path, args.load_config_path, args.dump_dir_path, args.backend)