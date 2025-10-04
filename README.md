# cute-bench

Simple GPU kernel benchmarking with clock locking for stable measurements.

## Installation

```bash
pip install git+https://github.com/NTT123/cute-bench.git
```

## Usage

```python
import torch
from cute_bench import benchmark, benchmark_cuda_event

def generate_workspace():
    a = torch.randn(4096, 4096, device='cuda')
    b = torch.randn(4096, 4096, device='cuda')
    c = torch.zeros(4096, 4096, device='cuda')
    return a, b, c

# Benchmark with torch profiler (returns dict of kernel measurements)
results = benchmark(
    fn=lambda a, b, c: torch.matmul(a, b, out=c),
    workspace_generator=generate_workspace,
    num_warmup_runs=1000,
    num_active_runs=100,
)

for kernel_name, measurement in results.items():
    print(f"{kernel_name}: {measurement}")

# Benchmark with CUDA events (returns KernelMeasurement)
result = benchmark_cuda_event(
    fn=lambda a, b, c: torch.matmul(a, b, out=c),
    workspace_generator=generate_workspace,
    num_warmup_runs=1000,
    num_active_runs=100,
)

print(f"Duration: {result.avg/1e3:.5f} ms ± {result.error:.2f} μs")
```

## API

### `benchmark(fn, workspace_generator, **kwargs)`

Benchmark using torch profiler. Returns dict of kernel measurements.

- `fn`: Function to benchmark
- `workspace_generator`: Function that generates workspace data
- `num_warmup_runs`: Warmup iterations (default: 1000)
- `num_active_runs`: Measured iterations (default: 50)
- `num_workspaces`: Pre-generated workspaces (default: 50)
- `lock_clocks`: Lock GPU clocks (default: True)
- `device_index`: GPU device index (default: 0)

Returns: `dict[str, KernelMeasurement]`

### `benchmark_cuda_event(fn, workspace_generator, **kwargs)`

Benchmark using torch.cuda.Event. Returns single timing measurement.

Same parameters as `benchmark()`.

Returns: `KernelMeasurement`

### `KernelMeasurement`

Stores kernel timing measurements.

- `timings`: Array of timing measurements (μs)
- `avg`: Average timing (μs)
- `error`: Mean absolute error (μs)

### `GPUClockLocker`

Context manager for locking GPU clocks to TDP frequency.

```python
from cute_bench import GPUClockLocker

with GPUClockLocker(device_index=0):
    # Your benchmark code
    pass
```
