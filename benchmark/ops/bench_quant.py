###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.core.float8 import float8_e4m3

M_SIZE_LIST = [4096]
EP_SIZE_LIST = [16]


def generate_deepseekv3_test_cases():
    test_cases = []
    n_routed_experts = 256
    moe_intermediate_size = 2048
    hidden_size = 7168
    for ep in EP_SIZE_LIST:
        B = n_routed_experts // ep
        for M in M_SIZE_LIST:
            for N, K in [
                (2 * moe_intermediate_size, hidden_size),
                (hidden_size, moe_intermediate_size),
            ]:
                for dtype in [torch.bfloat16]:
                    test_cases.append(
                        {
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def generate_deepseekv2_test_cases():
    test_cases = []
    n_routed_experts = 160
    moe_intermediate_size = 1536
    hidden_size = 5120
    for ep in EP_SIZE_LIST:
        B = n_routed_experts // ep
        for M in M_SIZE_LIST:
            for N, K in [
                (2 * moe_intermediate_size, hidden_size),
                (hidden_size, moe_intermediate_size),
            ]:
                for dtype in [torch.bfloat16]:
                    test_cases.append(
                        {
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        }
                    )
    return test_cases


def calc_scale_and_scale_inv(x: torch.Tensor, fp8_max: float, row_wise: bool = True):
    if row_wise:
        if x.dim() == 2:
            amax = x.abs().amax(dim=1, keepdim=True)
        elif x.dim() == 3:
            amax = x.abs().amax(dim=2, keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
    else:
        if x.dim() == 2:
            amax = x.abs().amax(dim=0, keepdim=True)
        elif x.dim() == 3:
            amax = x.abs().amax(dim=(1), keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

    scale = torch.full_like(amax, fill_value=fp8_max, dtype=torch.float32, device=x.device) / amax
    scale_inv = 1.0 / scale

    return scale, scale_inv


def bench_quant(B, M, K, dtype, row_major):
    device = "cuda"
    # Prepare inputs
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=False)
    x_scale, x_scale_inv = calc_scale_and_scale_inv(x, torch.finfo(float8_e4m3).max, row_major)
    print(x_scale.shape)

    # Forward pass for implementation
    fwd_func = lambda: torch.ops.primus_turbo_cpp_extension.fp8_quantize_row_col(
        x, x_scale, float8_e4m3, row_major
    )
    out = fwd_func()
    print(x.shape, out.shape)
    # Warmup
    warmup = 20
    for _ in range(warmup):
        fwd_func()

    torch.cuda.synchronize()

    # Benchmark
    fwd_timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fwd_func},
    )

    fwd_measurement = fwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    if row_major:
        bytes = B * M * K * 4 + B * M * 2
    else:
        bytes = B * M * K * 4 + K * 2
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms")
    print(f"bytes per sec: {(bytes *1e-9 / fwd_mean_time_ms * 1e-3):.3f} GB/s")
    return fwd_mean_time_ms, 0, 0, 0


if __name__ == "__main__":
    dpv2_test_cases = generate_deepseekv2_test_cases()
    dpv3_test_cases = generate_deepseekv3_test_cases()
    test_configs = dpv3_test_cases

    import pandas as pd

    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "TestID",
            "B",
            "M",
            "N",
            "K",
            "dtype",
            "Forward Time (ms)",
            "Forward TFLOPS",
            "Backward Time (ms)",
            "Backward TFLOPS",
        ]
    )
    test_id = 0
    for config in test_configs:
        B = config["B"]
        M = config["M"]
        K = config["K"]
        dtype = config["dtype"]
        print(f"\n{'='*50}")
        print(f"Testing config: {config}")
        print(f"{'='*50}")
        test_id += 1
        try:
            # Run benchmark
            fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops = bench_quant(
                B=B, M=M, K=K, dtype=dtype, row_major=True
            )

        except Exception as e:
            print(f"Failed to run {config}: {str(e)}")
