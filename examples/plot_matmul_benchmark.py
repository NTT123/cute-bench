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
sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Storage for results
profiler_avgs = []
profiler_errors = []
cuda_event_avgs = []
cuda_event_errors = []

for n in sizes:
    print(f"Benchmarking {n}×{n}×{n}...")

    # Adjust benchmark parameters based on matrix size
    if n <= 64:
        num_warmup = 5000
        num_active = 5000
    elif n <= 256:
        num_warmup = 5000
        num_active = 5000
    else:
        num_warmup = 5000
        num_active = 5000

    workspace_gen = generate_workspace(n)

    # Benchmark with torch profiler
    results = benchmark(
        fn=lambda a, b, c: torch.matmul(a, b, out=c),
        workspace_generator=workspace_gen,
        num_warmup_runs=num_warmup,
        num_active_runs=num_active,
        num_workspaces=2000,
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

    profiler_avgs.append(sgemm_kernel.avg / 1e3)  # Convert μs to ms
    profiler_errors.append(sgemm_kernel.error / 1e3)  # Convert μs to ms

    # Benchmark with CUDA events
    result = benchmark_cuda_event(
        fn=lambda a, b, c: torch.matmul(a, b, out=c),
        workspace_generator=workspace_gen,
        num_warmup_runs=num_warmup,
        num_active_runs=num_active,
        num_workspaces=1000,
    )

    cuda_event_avgs.append(result.avg / 1e3)  # Convert μs to ms
    cuda_event_errors.append(result.error / 1e3)  # Convert μs to ms

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
plt.yscale('log')
plt.xlabel('Matrix Size (n×n×n)')
plt.ylabel('Time (ms)')
plt.title(f'Matrix Multiplication Benchmark - {device_name}')
plt.legend()
plt.grid(True, alpha=0.3)

# Custom y-axis ticks at 2us, 5us, 10us, 30us, 150us
ytick_values = [0.002, 0.005, 0.01, 0.03, 0.15]  # in ms
ytick_labels = ['2 μs', '5 μs', '10 μs', '30 μs', '150 μs']
plt.yticks(ytick_values, ytick_labels)

# Save plot
plt.savefig('matmul_benchmark.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to matmul_benchmark.png")

# Optionally display
# plt.show()
