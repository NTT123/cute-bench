"""Benchmark matrix multiplication across different sizes and plot results."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from cute_bench import benchmark, benchmark_cuda_event


def generate_workspace(n):
    """Create workspace generator for n×n×n matmul."""
    def _gen():
        a = torch.zeros(n, n, dtype=torch.float32, device='cuda')
        b = torch.zeros(n, n, dtype=torch.float32, device='cuda')
        c = torch.zeros(n, n, dtype=torch.float32, device='cuda')
        return a, b, c
    return _gen


device_name = torch.cuda.get_device_name(0)
print(f"Device: {device_name}\n")

# Matrix sizes to benchmark
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Storage for results
profiler_avgs = []
profiler_errors = []
cuda_event_avgs = []
cuda_event_errors = []

for n in sizes:
    print(f"Benchmarking {n}×{n}×{n}...")

    num_warmup = 5000
    num_active = 5000

    workspace_gen = generate_workspace(n)
    num_workspaces = min(num_warmup + num_active, 200 * (1024 // n) ** 2)

    # Benchmark with torch profiler
    results = benchmark(
        fn=lambda a, b, c: torch.matmul(a, b, out=c),
        workspace_generator=workspace_gen,
        num_warmup_runs=num_warmup,
        num_active_runs=num_active,
        num_workspaces=num_workspaces,
    )

    # Find sgemm kernel (in case there are multiple kernels)
    sgemm_kernel = None
    for kernel_name, measurement in results.items():
        if 'sgemm' in kernel_name.lower():
            sgemm_kernel = measurement
            break

    # Fallback to first kernel if no sgemm found
    if sgemm_kernel is None:
        sgemm_kernel = next(iter(results.values()))

    profiler_avgs.append(sgemm_kernel.avg)  # in μs
    profiler_errors.append(sgemm_kernel.error)  # in μs

    # Benchmark with CUDA events
    result = benchmark_cuda_event(
        fn=lambda a, b, c: torch.matmul(a, b, out=c),
        workspace_generator=workspace_gen,
        num_warmup_runs=num_warmup,
        num_active_runs=num_active,
        num_workspaces=num_workspaces,
        num_blocked_cycles=1_000_000,
    )

    cuda_event_avgs.append(result.avg)  # in μs
    cuda_event_errors.append(result.error)  # in μs

    print(f"  torch.profiler: {sgemm_kernel}")
    print(f"  CUDA event:     {result}\n")

# Convert to numpy arrays
profiler_avgs = np.array(profiler_avgs)
profiler_errors = np.array(profiler_errors)
cuda_event_avgs = np.array(cuda_event_avgs)
cuda_event_errors = np.array(cuda_event_errors)

# Create plot
plt.figure(figsize=(10, 6))

# Plot with error bars
plt.errorbar(sizes, profiler_avgs, yerr=profiler_errors,
                marker='o', label='torch.profiler', capsize=5, linewidth=2)
plt.errorbar(sizes, cuda_event_avgs, yerr=cuda_event_errors,
                marker='s', label='CUDA event', capsize=5, linewidth=2)

plt.xscale('log')
plt.xlabel('Matrix Size (n×n×n)')
plt.ylabel('Time (μs)')
plt.title(f'Matrix Multiplication Benchmark - {device_name}')
plt.legend()
plt.grid(True, alpha=0.3)

# Custom x-axis ticks
xtick_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
plt.xticks(xtick_values, [str(x) for x in xtick_values])

# Save plot
plt.savefig('matmul_benchmark.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to matmul_benchmark.png")

# Optionally display
# plt.show()
