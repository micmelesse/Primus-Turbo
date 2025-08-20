###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import gc
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange, repeat
from flash_attn import flash_attn_func
from metrics import compute_snr, cosine_similarity, max_abs_error, relative_error
from tqdm import tqdm
from utils import (
    DEVICE,
    dump_tensor,
    get_device_name,
    get_device_type,
    get_tensor_name,
    load_tensor,
    save_to_sheets,
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
        model_name="llama2-7B",
        seqlen_q=4096,
        seqlen_kv=4096,
        num_head_q=32,
        num_head_kv=32,
        head_dim_qk=128,
        head_dim_v=128,
        batch_size=1,
    ),
    AttnModelCfg(
        model_name="llama2-70B",
        seqlen_q=4096,
        seqlen_kv=4096,
        num_head_q=64,
        num_head_kv=8,
        head_dim_qk=128,
        head_dim_v=128,
        batch_size=1,
    ),
    AttnModelCfg(
        model_name="llama3-8B",
        seqlen_q=8192,
        seqlen_kv=8192,
        num_head_q=32,
        num_head_kv=8,
        head_dim_qk=128,
        head_dim_v=128,
        batch_size=1,
    ),
    AttnModelCfg(
        model_name="llama3-70B",
        seqlen_q=8192,
        seqlen_kv=8192,
        num_head_q=64,
        num_head_kv=8,
        head_dim_qk=128,
        head_dim_v=128,
        batch_size=1,
    ),
    # AttnModelCfg(
    #     model_name="llama3.1-405B",
    #     seqlen_q=16384,
    #     seqlen_kv=16384,
    #     num_head_q=128,
    #     num_head_kv=8,
    #     head_dim_qk=128,
    #     head_dim_v=128,
    #     batch_size=1,
    # ),
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
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
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
    Compute reference output and softmax_lse using FlashAttn's test function
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

    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention

    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)

    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout="bshd"):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""
    org_dtype = q.dtype
    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown layout {layout}")

    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        is_causal=causal,
        scale=sm_scale,
        enable_gqa=n_rep > 1,
    )
    if layout == "bshd":
        o_ref = o_ref.transpose(1, 2)
    return o_ref.to(org_dtype)


def attention(model_config, dtype, backend="FA", baseline="fa_test"):
    q_layout = (
        model_config.batch_size,
        model_config.seqlen_q,
        model_config.num_head_q,
        model_config.head_dim_qk,
    )
    k_layout = (
        model_config.batch_size,
        model_config.seqlen_kv,
        model_config.num_head_kv,
        model_config.head_dim_qk,
    )
    v_layout = (
        model_config.batch_size,
        model_config.seqlen_kv,
        model_config.num_head_kv,
        model_config.head_dim_v,
    )
    out_layout = (
        model_config.batch_size,
        model_config.seqlen_q,
        model_config.num_head_q,
        model_config.head_dim_v,
    )

    query_cpu = torch.randn(q_layout, dtype=dtype, requires_grad=True)
    key_cpu = torch.randn(k_layout, dtype=dtype, requires_grad=True)
    value_cpu = torch.randn(v_layout, dtype=dtype, requires_grad=True)
    grad_out_cpu = torch.randn(out_layout, dtype=dtype)

    query = query_cpu.detach().to(DEVICE).requires_grad_()
    key = key_cpu.detach().to(DEVICE).requires_grad_()
    value = value_cpu.detach().to(DEVICE).requires_grad_()
    grad_out = grad_out_cpu.detach().to(DEVICE)

    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()
    grad_out_ref = grad_out.clone().detach()

    if baseline == "torch_nn":
        out_ref = attention_vanilla_forward_pytorch_ref_impl(
            query_ref,
            key_ref,
            value_ref,
            sm_scale=(query_ref.shape[-1] ** (-0.5)),
            causal=True,
            layout="bshd",
        )
    else:
        out_ref, _ = attention_ref(
            query_ref,
            key_ref,
            value_ref,
            causal=True,
            upcast=True,
        )

    if backend == "FA":
        out, _, _ = flash_attn_func(
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
        import aiter

        out, _, _ = aiter.flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=True,
            return_attn_probs=True,
        )

    (
        dq,
        dk,
        dv,
    ) = torch.autograd.grad(out, (query, key, value), grad_out)
    (
        dq_ref,
        dk_ref,
        dv_ref,
    ) = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), grad_out_ref)

    result = (
        query_cpu,
        key_cpu,
        value_cpu,
        grad_out_cpu,
        out_ref.cpu(),
        dq_ref.cpu(),
        dk_ref.cpu(),
        dv_ref.cpu(),
        out.cpu(),
        dq.cpu(),
        dk.cpu(),
        dv.cpu(),
    )

    del query, key, value, grad_out
    del query_ref, key_ref, value_ref, grad_out_ref
    del out, dq, dk, dv
    del out_ref, dq_ref, dk_ref, dv_ref

    torch.cuda.empty_cache()
    gc.collect()

    return result


def benchmark(seed, report_dir_path, load_config_path=None, dump_dir_path=None, backend="FA"):
    num_iters = 100
    torch.manual_seed(seed)
    device_type = get_device_type()
    device_name = get_device_name()

    report_dir = Path(report_dir_path) / Path(FUNC_NAME) / Path(backend)
    report_dir.mkdir(parents=True, exist_ok=True)

    if load_config_path is not None:
        with open(load_config_path, "r", encoding="utf-8") as f:
            load_config: dict = json.load(f)
            load_dir = Path(load_config.get("load_dir"))

    if dump_dir_path is not None:
        dump_dir = Path(dump_dir_path) / Path(FUNC_NAME)
        dump_dir.mkdir(parents=True, exist_ok=True)

    all_results = defaultdict(list)
    baseline_results = defaultdict(list)

    def format_metric(val):
        if isinstance(val, float):
            if abs(val) < 0.1:
                return f"{val:.3e}"
            if abs(val) >= 0.1 and abs(val) <= 1.5:
                return f"{val:.4f}"
            return f"{val:.2f}"
        return str(val)

    def compute_metric_stats(metrics_dict):
        return {
            name: (max(values), min(values), sum(values) / len(values))
            for name, values in metrics_dict.items()
        }

    def collect_error(metrics, func, ref_dict, out_dict):
        for key in ref_dict.keys():
            metrics[key].append(func(ref_dict[key], out_dict[key]))

    def make_metric_table(metric_name, stats_dict):
        result = {}
        for name, vals in stats_dict.items():
            result[f"{metric_name}({name}-max)"] = format_metric(vals[0])
            result[f"{metric_name}({name}-min)"] = format_metric(vals[1])
            result[f"{metric_name}({name}-mean)"] = format_metric(vals[2])
        return result

    def append_null_item(results):
        for _, metric_stats in results.items():
            metric_stats.append({k: "" for k in metric_stats[-1].keys()})

    for config in g_model_cfg:
        for dtype in [torch.float16, torch.bfloat16]:
            # Initialize metric collectors
            relative_errors = defaultdict(list)
            max_abs_errors = defaultdict(list)
            cosine_sims = defaultdict(list)
            snrs = defaultdict(list)
            dump_tensors = defaultdict(list)

            for _ in tqdm(range(num_iters), desc=f"Running {config.model_name}-{dtype}"):
                (
                    query_cpu,
                    key_cpu,
                    value_cpu,
                    grad_out_cpu,
                    out_ref,
                    dq_ref,
                    dk_ref,
                    dv_ref,
                    out,
                    dq,
                    dk,
                    dv,
                ) = attention(config, dtype, backend)
                ref_dict = {"output": out_ref, "dq": dq_ref, "dk": dk_ref, "dv": dv_ref}
                out_dict = {"output": out, "dq": dq, "dk": dk, "dv": dv}

                collect_error(relative_errors, relative_error, ref_dict, out_dict)
                collect_error(max_abs_errors, max_abs_error, ref_dict, out_dict)
                collect_error(cosine_sims, cosine_similarity, ref_dict, out_dict)
                collect_error(snrs, compute_snr, ref_dict, out_dict)

                for name, tensor in zip(
                    ["output", "dq", "dk", "dv", "query", "key", "value", "grad_out"],
                    [
                        out_ref,
                        dq_ref,
                        dk_ref,
                        dv_ref,
                        query_cpu,
                        key_cpu,
                        value_cpu,
                        grad_out_cpu,
                    ],
                ):
                    dump_tensors[name].append(tensor)

            # Average tensors for dump
            avg_dump_tensors = {k: torch.stack(v).mean(dim=0) for k, v in dump_tensors.items()}

            # Compute statistics
            rel_stats = compute_metric_stats(relative_errors)
            abs_stats = compute_metric_stats(max_abs_errors)
            cos_stats = compute_metric_stats(cosine_sims)
            snr_stats = compute_metric_stats(snrs)

            base_info = {
                "func": f"{FUNC_NAME.upper()}({device_name})",
                "backend": f"unfused vs {backend}",
                "dtype": str(dtype).split(".")[-1],
                "config": str(config),
            }

            # Collect result dicts per metric
            result_snr = {**base_info, **make_metric_table("SNR", snr_stats)}
            result_cos = {**base_info, **make_metric_table("CosSim", cos_stats)}
            result_rel = {**base_info, **make_metric_table("RelError", rel_stats)}
            result_abs = {**base_info, **make_metric_table("MaxAbsErr", abs_stats)}
            all_results["SNR"].append(result_snr)
            all_results["CosSim"].append(result_cos)
            all_results["RelError"].append(result_rel)
            all_results["MaxAbsErr"].append(result_abs)

            if dump_dir_path is not None:
                dump_file = get_tensor_name(device_type, FUNC_NAME, dtype, config=config)
                dump_tensor(avg_dump_tensors, dump_dir, dump_file)

            if load_config_path is not None:
                load_file = get_tensor_name(load_config.get("device_type"), FUNC_NAME, dtype, config=config)
                tensors_load = load_tensor(load_dir, load_file)

                load_info = {
                    "func": f"{FUNC_NAME.upper()} ",
                    "backend": f"unfused({device_name} vs {load_config.get('device_name')})",
                    "dtype": str(dtype).split(".")[-1],
                    "config": str(config),
                }
                tensor_stats = defaultdict(defaultdict)
                for name in tensors_load.keys():
                    ref = tensors_load[name]
                    val = avg_dump_tensors[name]
                    tensor_stats["SNR"][name] = format_metric(compute_snr(ref, val))
                    tensor_stats["CosSim"][name] = format_metric(cosine_similarity(ref, val))
                    tensor_stats["RelError"][name] = format_metric(relative_error(ref, val))
                    tensor_stats["MaxAbsErr"][name] = format_metric(max_abs_error(ref, val))
                baseline_results["SNR"].append({**load_info, **tensor_stats["SNR"]})
                baseline_results["CosSim"].append(
                    {
                        **load_info,
                        **tensor_stats["CosSim"],
                    }
                )
                baseline_results["RelError"].append(
                    {
                        **load_info,
                        **tensor_stats["RelError"],
                    }
                )
                baseline_results["MaxAbsErr"].append(
                    {
                        **load_info,
                        **tensor_stats["MaxAbsErr"],
                    }
                )
            torch.cuda.empty_cache()
            gc.collect()

        append_null_item(all_results)
        if load_config_path is not None:
            append_null_item(baseline_results)

    # Save results into separate Excel sheets
    local_report = report_dir / f"benchmark_{device_type}_{backend}_vs_unfused_{FUNC_NAME}.xlsx"

    save_to_sheets(local_report, all_results)
    if load_config_path is not None:
        baseline_report = (
            report_dir
            / f"benchmark_{device_type}_vs_{load_config.get('device_type')}_baseline_{FUNC_NAME}.xlsx"
        )
        save_to_sheets(baseline_report, baseline_results)


if __name__ == "__main__":
    """
    Example:
    [NV PLATFORM]python eval_attention_accuracy.py --report-dir-path output --dump-dir-path dump
    [NV PLATFORM]tar -cvzf data.tar.gz output dump
    [AMD PLATFORM]tar -xvzf data.tar.gz
    [AMD PLATFORM]python eval_attention_accuracy.py --report-dir-path output --backend "AITER"
    [AMD PLATFORM]python eval_attention_accuracy.py --report-dir-path output --load-config-path load_config.json
    load_config.json:
    {
        "device_type": "NVIDIA",
        "device_name": "H200",
        "load_dir": "dump/attention",
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config-path", default=None, type=str)
    parser.add_argument("--dump-dir-path", default=None, type=str)
    parser.add_argument("--report-dir-path", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--backend", default="FA", type=str, choices=["FA", "AITER"])

    args = parser.parse_args()
    benchmark(
        args.seed,
        args.report_dir_path,
        args.load_config_path,
        args.dump_dir_path,
        args.backend,
    )
