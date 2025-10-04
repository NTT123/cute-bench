import pytest
import numpy as np
from cute_bench import KernelMeasurement


def test_large_values_seconds():
    """Test formatting when avg is in seconds range."""
    # avg=2.5s, error=1ms
    timings = np.array([2_500_000.0] * 10)  # 2.5s in μs
    timings[0] += 1_000.0  # Add 1ms variation
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    assert "s" in result
    assert "±" in result
    # Should show avg in seconds with enough precision
    assert "2.5" in result


def test_medium_values_milliseconds():
    """Test formatting when avg is in milliseconds range."""
    # avg≈9.297ms, error≈10.11μs
    timings = np.array([9_296.52] * 10)  # in μs
    timings[0] += 10.11
    timings[1] -= 10.11
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    assert "ms" in result
    assert "±" in result
    # Should have enough decimal places to show error magnitude
    assert "9.29" in result


def test_small_values_microseconds():
    """Test formatting when avg is in microseconds range."""
    # avg≈123.456μs, error≈0.12μs
    timings = np.array([123.456] * 10)  # in μs
    timings[0] += 0.12
    timings[1] -= 0.12
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    assert "μs" in result
    assert "±" in result
    assert "123" in result


def test_very_small_values_nanoseconds():
    """Test formatting when avg is in nanoseconds range."""
    # avg≈0.0456μs = 45.6ns, error≈0.0012μs = 1.2ns
    timings = np.array([0.0456] * 10)  # in μs
    timings[0] += 0.0012
    timings[1] -= 0.0012
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    assert "ns" in result
    assert "±" in result
    assert "45" in result or "46" in result


def test_very_small_error():
    """Test formatting when error is very small compared to avg."""
    # avg=1ms, error≈0.01μs - need high precision
    timings = np.array([1_000.0] * 10)  # 1ms in μs
    timings[0] += 0.01
    timings[1] -= 0.01
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    # Should show avg in ms with enough decimals to preserve error info
    assert "ms" in result
    assert "±" in result


def test_large_error_relative_to_avg():
    """Test formatting when error is large relative to avg."""
    # avg=100μs, error≈50μs
    timings = np.array([100.0, 150.0, 50.0, 100.0, 100.0])  # in μs
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    assert "μs" in result
    assert "±" in result
    assert "100" in result or "10" in result  # avg around 100


def test_exact_millisecond():
    """Test formatting of exact millisecond value."""
    # avg=5ms, error=0
    timings = np.array([5_000.0] * 10)  # exactly 5ms in μs
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    assert "ms" in result
    assert "5" in result


def test_format_consistency():
    """Test that format is consistent: 'X.XX unit ± Y.YY unit'."""
    timings = np.array([1_000.0] * 10)  # 1ms
    measurement = KernelMeasurement(timings)

    result = str(measurement)
    # Should have the ± symbol
    assert "±" in result
    # Should have two parts separated by ±
    parts = result.split("±")
    assert len(parts) == 2
