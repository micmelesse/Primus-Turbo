import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import generate_uniform_seq_len, grouped_gemm_ref
from tests.test_utils import compute_snr

def generate_deepseekv3_test_cases():
    test_cases = []
    EP = [8,16,32]
    n_routed_experts = 256
    moe_intermediate_size = 2048
    hidden_size = 7168
    for ep in EP:
        B = n_routed_experts // ep
        for M in [512,2048,4096]:
            for N,K in [(2 * moe_intermediate_size, hidden_size),(hidden_size,moe_intermediate_size)]:
                    for dtype in [torch.bfloat16]:
                        test_cases.append({
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        })
    return test_cases

def generate_deepseekv2_test_cases():
    test_cases = []
    EP = [8,16,32]
    n_routed_experts = 160
    moe_intermediate_size = 1536
    hidden_size = 5120
    for ep in EP:
        B = n_routed_experts // ep

        for M in [512,2048,4096]:
            for N,K in [(2 * moe_intermediate_size, hidden_size),(hidden_size,moe_intermediate_size)]:
                    for dtype in [torch.bfloat16]:
                        test_cases.append({
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        })
    return test_cases

def generate_poolside_test_cases():
    test_cases = []
    EP = [8,16,32]
    n_routed_experts = 128
    moe_intermediate_size = 2048
    hidden_size = 8192
    for ep in EP:
        B = n_routed_experts // ep

        for M in [512,2048,4096]:
            for N,K in [(2 * moe_intermediate_size, hidden_size),(hidden_size,moe_intermediate_size)]:
                    for dtype in [torch.bfloat16]:
                        test_cases.append({
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
                        })
    return test_cases
def bench_grouped_gemm(B, M, N, K, dtype, test_backward):
    device = "cuda"

    # Prepare inputs
    x = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    w = torch.randn((B, N, K), dtype=dtype, device=device, requires_grad=True)
    seg_lens = generate_uniform_seq_len(B, B * M).to(device)  # int64
    print("seg_lens: ", seg_lens)
    x_ref = x.clone().detach().requires_grad_()
    w_ref = w.clone().detach().requires_grad_()

    # Reference forward pass

    out_ref = grouped_gemm_ref(x_ref, w_ref, seg_lens, True)

    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out, retain_graph=True)
    # Forward pass for implementation
    fn_forward = lambda: grouped_gemm(x, w, seg_lens)
    out = fn_forward()

    out.backward(grad_out, retain_graph=True)

    # Compute SNRs
    out_snr = compute_snr(out_ref, out)
    assert out_snr > 20, "out_snr too low"

    x_grad_snr = compute_snr(x_ref.grad, x.grad)
    w_grad_snr = compute_snr(w_ref.grad, w.grad)
    assert x_grad_snr > 15, "x_grad_snr too low"
    assert w_grad_snr > 15, "w_grad_snr too low"

    # Calculate FLOPs
    total_flops = 2 * B * M * N * K

    if test_backward:
        # Re-run forward pass to get fresh output
        o = fn_forward()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        total_flops *= 2  # Approximate factor for backward pass
    else:
        fn = fn_forward

    # Warmup
    warmup = 20
    for _ in range(warmup):
        _ = fn_forward()
    torch.cuda.synchronize()

    # Benchmark
    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
    )
    measurement = timer.timeit(100)
    mean_time_ms = measurement.mean * 1e3
    flops_per_sec = total_flops / (mean_time_ms * 1e-3) / 1e12  # TFLOPs/s

    print(f"Mean time: {mean_time_ms:.3f} ms | TFLOPS: {flops_per_sec:.2f}")
    return mean_time_ms, flops_per_sec, total_flops


if __name__ == "__main__":
    dpv2_test_cases = generate_deepseekv2_test_cases()
    dpv3_test_cases = generate_deepseekv3_test_cases()
    ps_test_cases = generate_poolside_test_cases()
    test_configs = dpv3_test_cases + dpv3_test_cases + ps_test_cases
    # print(test_configs)
    import pandas as pd
    from tabulate import tabulate
    
    # DataFrame to store results
    results = pd.DataFrame(
        columns=[
            "Test id",
            "B",
            "M",
            "N",
            "K",
            "dtype",
            "Test Backward",
            "Time (ms)",
            "TFLOPS",
        ]
    )
    test_id = 0
    for config in test_configs:
        B = config["B"]
        M = config["M"]
        N = config["N"]
        K = config["K"]
        dtype = config["dtype"]
        print(f"\n{'='*50}")
        print(f"Testing config: {config}")
        print(f"{'='*50}")
        for test_backward in [False,True]:
            test_id += 1
            try:
                # Run benchmark
                time_ms, tflops, _ = bench_grouped_gemm(
                    B=B,
                    M=M,
                    N=N,
                    K=K,
                    dtype=dtype,
                    test_backward=test_backward,
                )

                # Add to results table
                new_row = {
                    "Test id": test_id,
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "dtype": dtype,
                    "Test Backward": test_backward,
                    "Time (ms)": f"{time_ms:.2f}",
                    "TFLOPS": f"{tflops:.2f}",
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                print(f"Failed to run {config}: {str(e)}")
                new_row = {
                    "Test id": test_id,
                    "B": B,
                    "M": M,
                    "N": N,
                    "K": K,
                    "dtype": dtype,
                    "Test Backward": test_backward,
                    "Time (ms)": "Failed",
                    "TFLOPS": "N/A",
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    # Print results
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV
    results.to_csv("grouped_gemm_benchmark_results.csv", index=False)
    print("Results saved to grouped_gemm_benchmark_results.csv")
