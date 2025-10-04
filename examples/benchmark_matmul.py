"""Example: Benchmark matrix multiplication"""
import torch
from cute_bench import benchmark, benchmark_cuda_event


def generate_workspace(m=4096, n=4096, k=4096, dtype=torch.float32, device='cuda'):
    """Generate workspace for matrix multiplication: C = A @ B"""
    a = torch.zeros(m, k, dtype=dtype, device=device)
    b = torch.zeros(k, n, dtype=dtype, device=device)
    c = torch.zeros(m, n, dtype=dtype, device=device)
    return a, b, c


if __name__ == "__main__":
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}\n")

    # Benchmark with torch profiler
    results = benchmark(
        fn=lambda a, b, c: torch.matmul(a, b, out=c),
        workspace_generator=generate_workspace,
        num_warmup_runs=1000,
        num_active_runs=100,
        num_workspaces=50
    )

    print("Profiler results:")
    for kernel_name, measurement in results.items():
        print(f"  {kernel_name}: {measurement.avg/1e3:.5f} ms ± {measurement.error:.2f} μs")

    # Benchmark with cuda events
    avg, error = benchmark_cuda_event(
        fn=lambda a, b, c: torch.matmul(a, b, out=c),
        workspace_generator=generate_workspace,
        num_warmup_runs=1000,
        num_active_runs=100,
        num_workspaces=50
    )

    print(f"\nCUDA Event result:")
    print(f"  {avg/1e3:.5f} ms ± {error:.2f} μs")