import time
import torch
import argparse




# PyTorch semi-structured sparsity API
from torch.sparse import to_sparse_semi_structured

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Train a model")

parser.add_argument("--M", type=int, default=8192,
                    help="M")
parser.add_argument("--K", type=int, default=8192,
                    help="K")
parser.add_argument("--N", type=int, default=8192)


args = parser.parse_args()



assert torch.cuda.is_available()
device = "cuda"

dtype = torch.float16

# Matrix sizes
M = args.M
K = args.K
N = args.N

# Input
A_dense = torch.randn(M, K, device=device, dtype=dtype)

# Dense weight
B = torch.randn(N, K, device=device, dtype=dtype)


############################################################
# Convert weight to 2:4 sparsity
############################################################

# Keep largest 2 values in every group of 4
A = A_dense.clone()

A_view = A.view(-1, 4)

absvals = A_view.abs()
top2 = torch.topk(absvals, k=2, dim=1).indices

mask = torch.zeros_like(A_view, dtype=torch.bool)
mask.scatter_(1, top2, True)

A_view *= mask

# Convert to hardware sparse format
A_sparse = to_sparse_semi_structured(A)
A_dense = A_sparse.to_dense()

############################################################
# Warmup
############################################################

for _ in range(20):
    y = torch.nn.functional.linear(A_dense, B)
    y = torch.nn.functional.linear(A_sparse, B)

torch.cuda.synchronize()


############################################################
# Benchmark helper
############################################################

def benchmark(fn, iters=100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


############################################################
# Dense benchmark
############################################################

dense_latency = benchmark(lambda: torch.nn.functional.linear(A_dense, B))

############################################################
# Sparse benchmark
############################################################

sparse_latency = benchmark(lambda: torch.nn.functional.linear(A_sparse ,B))

############################################################
# Verify correctness
############################################################

dense_result = torch.nn.functional.linear(A_dense, B)
sparse_result = torch.nn.functional.linear(A_sparse, B)

max_error = (dense_result - sparse_result).abs().max().item()
print(f"Max error: {max_error}")

print(f"Dense latency : {dense_latency:.3f} ms")
print(f"Sparse latency: {sparse_latency:.3f} ms")
print(f"Speedup       : {dense_latency/sparse_latency:.2f}x")
