# 💡 Primus-Turbo Example

This page shows usage of **Primus-Turbo**.


## 1. Operators

### 1.1 Gemm
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16
M = 128
N = 256
K = 512

# a [M, K]
a = torch.randn((M, K), dtype=dtype, device=device)
# b [K, N]
b = torch.randn((K, N), dtype=dtype, device=device)
# c [M, N]
c = turbo.ops.gemm(a, b, trans_a=False, trans_b=False, out_dtype=dtype)

print(c)
print(c.shape)
```

### 1.2 Attention
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16

B = 4
S = 4096
H = 32
D = 128

q = torch.randn((B, S, H, D), dtype=dtype, device=device)
k = torch.randn((B, S, H, D), dtype=dtype, device=device)
v = torch.randn((B, S, H, D), dtype=dtype, device=device)
softmax_scale = q.shape[-1] ** (-0.5)

o = turbo.ops.attention(q, k, v, softmax_scale=softmax_scale, causal=True)

print(o)
print(o.shape)
```

### 1.3 Grouped Gemm
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16

G = 4
M = 128  # 128=32+16+48+32
N = 256
K = 512

group_lens = torch.tensor([32, 16, 48, 32], dtype=torch.long, device=device)
a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(G, K, N, device=device, dtype=dtype)
c = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=False)

print(c)
print(c.shape) # [128, 256]
```

## 2. Modules

### 2.1 Linear
```python
import torch
import primus_turbo.pytorch as turbo

device = "cuda:0"
dtype = torch.bfloat16

in_features = 512
out_features = 256
bias = True

input = torch.randn(128, in_features, device=device, dtype=dtype)
model = turbo.modules.Linear(
    in_features, out_features, bias=bias, device=device, dtype=dtype
)

# If you want to use torch.compile.
model = torch.compile(model, fullgraph=True, mode="max-autotune")

output = model(input)

print(model)
print(output)
print(output.shape)
```
