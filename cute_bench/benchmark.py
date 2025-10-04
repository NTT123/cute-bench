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

    def _format_time(self, value_us):
        """Format time value with appropriate unit.

        Args:
            value_us: Time value in microseconds

        Returns:
            tuple: (scaled_value, unit_string)
        """
        if value_us >= 1e6:  # >= 1 second
            return value_us / 1e6, "s"
        elif value_us >= 1e3:  # >= 1 millisecond
            return value_us / 1e3, "ms"
        elif value_us >= 1:  # >= 1 microsecond
            return value_us, "μs"
        else:  # < 1 microsecond (nanoseconds)
            return value_us * 1e3, "ns"

    def __str__(self):
        """Format as 'X.XX unit ± Y.YY unit' with smart precision."""
        avg_val, avg_unit = self._format_time(self.avg)
        err_val, err_unit = self._format_time(self.error)

        # Calculate how many decimal places needed for avg based on error magnitude
        # Convert error to avg's unit for comparison
        unit_scales = {"s": 1e6, "ms": 1e3, "μs": 1, "ns": 1e-3}
        err_in_avg_unit = self.error / unit_scales[avg_unit]

        # Determine precision: we want the last digit of avg to be at the same
        # magnitude as the error or smaller
        if err_in_avg_unit > 0:
            # log10(err) tells us the order of magnitude
            # We want avg to have decimals down to at least that level
            err_magnitude = np.floor(np.log10(err_in_avg_unit))
            # Number of decimals = -err_magnitude (but at least 0)
            decimals_needed = max(0, int(-err_magnitude) + 1)
        else:
            decimals_needed = 1

        # Cap at reasonable maximum
        decimals_needed = min(decimals_needed, 6)

        # Format error with 2-3 significant figures
        if err_val >= 100:
            err_decimals = 0
        elif err_val >= 10:
            err_decimals = 1
        else:
            err_decimals = 2

        avg_str = f"{avg_val:.{decimals_needed}f}"
        err_str = f"{err_val:.{err_decimals}f}"

        return f"{avg_str} {avg_unit} ± {err_str} {err_unit}"

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
                         lock_clocks=True, device_index=0, num_blocked_cycles=1_000_000):
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
        num_blocked_cycles: Number of cycles to block GPU before recording start event
                           (prevents CPU-side launch delay gaps)

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
            torch.cuda._sleep(num_blocked_cycles)
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
