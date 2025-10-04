import pynvml
import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity


class GPUClockLocker:
    """Lock GPU clocks to TDP base frequency for stable performance."""

    def __init__(self, device_index=0, enabled=True):
        self.device_index = device_index
        self.handle = None
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        pynvml.nvmlDeviceSetGpuLockedClocks(
            self.handle,
            pynvml.NVML_CLOCK_LIMIT_ID_TDP,
            pynvml.NVML_CLOCK_LIMIT_ID_TDP
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
        pynvml.nvmlShutdown()
        return False


class KernelMeasurement:
    """Stores kernel timing measurements with statistics."""

    def __init__(self, timings):
        self.timings = np.array(timings)
        self.avg = np.mean(self.timings)
        self.error = np.mean(np.abs(self.timings - self.avg))

    def __repr__(self):
        return f"KernelMeasurement(avg={self.avg/1e3:.5f}ms, error={self.error:.2f}μs)"


def benchmark(fn, workspace_generator, num_warmup_runs=1000, num_active_runs=50,
              num_workspaces=50, lock_clocks=True, device_index=0):
    """
    Benchmark using torch profiler.

    Args:
        fn: Function to benchmark
        workspace_generator: Function that generates workspace data
        num_warmup_runs: Warmup iterations
        num_active_runs: Measured iterations
        num_workspaces: Pre-generated workspaces to cycle through
        lock_clocks: Whether to lock GPU clocks
        device_index: GPU device index

    Returns:
        dict[str, KernelMeasurement]: Kernel name -> measurement
    """
    with GPUClockLocker(device_index, enabled=lock_clocks):
        N = num_warmup_runs + num_active_runs
        workspaces = [workspace_generator() for _ in range(num_workspaces)]
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for i in range(N):
                workspace = workspaces[i % num_workspaces]
                fn(*workspace)
                prof.step()
            torch.cuda.synchronize()

        # Collect kernel durations
        kernel_durations = {}
        for e in prof.events():
            if e.device_type.name == "CUDA":
                kernel_name = e.name
                if kernel_name not in kernel_durations:
                    kernel_durations[kernel_name] = []
                kernel_durations[kernel_name].append(e.device_time)

        # Create measurements (only active runs)
        results = {}
        for kernel_name, durations in kernel_durations.items():
            active_durations = durations[-num_active_runs:]
            results[kernel_name] = KernelMeasurement(active_durations)

        return results


def benchmark_cuda_event(fn, workspace_generator, num_warmup_runs=1000,
                         num_active_runs=50, num_workspaces=50,
                         lock_clocks=True, device_index=0):
    """
    Benchmark using torch.cuda.Event.

    Args:
        fn: Function to benchmark
        workspace_generator: Function that generates workspace data
        num_warmup_runs: Warmup iterations
        num_active_runs: Measured iterations
        num_workspaces: Pre-generated workspaces to cycle through
        lock_clocks: Whether to lock GPU clocks
        device_index: GPU device index

    Returns:
        KernelMeasurement: Timing measurements
    """
    with GPUClockLocker(device_index, enabled=lock_clocks):
        N = num_warmup_runs + num_active_runs
        workspaces = [workspace_generator() for _ in range(num_workspaces)]
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(N)]

        for i in range(N):
            workspace = workspaces[i % num_workspaces]
            start_events[i].record()
            fn(*workspace)
            end_events[i].record()

        torch.cuda.synchronize()

        timings = []
        for i in range(N):
            elapsed_time = start_events[i].elapsed_time(end_events[i]) * 1000  # ms to μs
            timings.append(elapsed_time)

        # Only keep active runs
        active_timings = timings[-num_active_runs:]
        return KernelMeasurement(active_timings)
