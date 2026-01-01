# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cute-bench is a lightweight GPU kernel benchmarking library for stable, repeatable CUDA kernel timing measurements. It provides accurate performance metrics through GPU clock locking and multiple measurement strategies.

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Run examples
python examples/benchmark_matmul.py
python examples/plot_matmul_benchmark.py
```

## Architecture

The library is contained in `cute_bench/benchmark.py` with three main components:

### Core Classes

**GPUClockLocker** - Context manager that locks GPU clocks to TDP base frequency using pynvml. Prevents clock throttling during benchmarks for stable measurements.

**KernelMeasurement** - Result container storing timing measurements in microseconds (μs). Calculates average and mean absolute error (MAE). Provides intelligent formatting that converts to appropriate units (ns, μs, ms, s) based on magnitude.

### Benchmarking Functions

**benchmark(fn, ...)** - Uses torch.profiler to measure individual CUDA kernels. Returns `dict[str, KernelMeasurement]` with per-kernel timings. Pre-generates workspace tensors to avoid allocation overhead during measurement.

**benchmark_cuda_event(fn, ...)** - Uses torch.cuda.Event for end-to-end timing. Returns single `KernelMeasurement`. Includes `num_blocked_cycles` parameter using `torch.cuda._sleep()` to block GPU and prevent CPU-launch-delay gaps.

### Measurement Strategy

Both functions follow a warmup-then-measure pattern:
1. **Warmup phase** (default 1000 runs) - Stabilizes GPU state
2. **Active phase** (default 50 runs) - Records measurements used for statistics
3. **Workspace cycling** - Pre-allocates multiple workspaces and cycles through them to avoid memory pressure during timing

## Key Technical Details

- Measurements stored internally in microseconds (μs)
- Error reported as Mean Absolute Error (MAE), not standard deviation
- `workspace_generator` parameter must be a callable that returns inputs for each run
- Uses torch.zeros (not torch.randn) for consistent benchmark inputs
