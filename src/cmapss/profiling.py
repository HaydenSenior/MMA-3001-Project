"""Timing, memory and floating-point-operation (FLOP) analysis for the RUL pipeline.

The FLOP counts here describe :class:`cmapss.models.RidgeRegressor` **as implemented**
(standardise, stack the penalty rows, solve by least squares), not a textbook normal-equation
ridge. They are leading-order estimates: terms of order ``n`` or ``p`` alone are kept only
where they are cheap to state, and library-level details (blocking, vectorisation) are ignored.
The least-squares term assumes the usual ``2 (n + p) p^2 - (2/3) p^3`` cost of reducing a tall
matrix by orthogonal factorisation; NumPy's ``lstsq`` calls LAPACK ``gelsd``, which does that
reduction and then an SVD of the small ``p x p`` factor, so the constant differs somewhat.

**FLOPs predict work, not time.** The rate at which a machine executes them depends strongly on
the shape of the problem: a tall, narrow least-squares solve runs at a small fraction of the
throughput the same machine reaches on a large square matrix multiply. Comparing a measured
FLOP rate against that ceiling is the point of this module, not a reason to distrust the counts.

Boosted trees are deliberately not given a FLOP count. Their cost is dominated by histogram
building, comparisons and memory traffic rather than floating-point arithmetic, so a FLOP
figure would be misleading; they are compared by measured time and model size instead.

Arithmetic intensity is FLOPs per byte of memory traffic. A low value means the computation
is limited by memory bandwidth rather than by the processor's arithmetic throughput, so making
the arithmetic cheaper would not make it faster.
"""

from __future__ import annotations

import pickle
import statistics
import time
import tracemalloc
from typing import Any, Callable, Dict, Tuple

BYTES_PER_FLOAT64 = 8


def ridge_fit_flops(n: int, p: int) -> float:
    """Leading-order FLOPs to fit :class:`RidgeRegressor` on ``n`` rows and ``p`` features.

    Breakdown:
      * column means, standard deviations and standardising ``X``: about ``6 n p``
      * least squares on the stacked ``(n + p) x p`` system by Householder QR:
        ``2 (n + p) p^2 - (2/3) p^3``
      * triangular solve and rescaling the coefficients: about ``p^2``

    Args:
        n: Number of training rows, ``>= 1``.
        p: Number of features, ``>= 1``.

    Returns:
        Estimated floating-point operations.

    Raises:
        ValueError: If ``n`` or ``p`` is less than 1.
    """
    if n < 1 or p < 1:
        raise ValueError("n and p must be >= 1")
    n, p = float(n), float(p)
    return 6.0 * n * p + 2.0 * (n + p) * p**2 - (2.0 / 3.0) * p**3 + p**2


def ridge_predict_flops(n: int, p: int) -> float:
    """FLOPs to predict ``n`` rows: one multiply and one add per feature per row, ``2 n p``.

    Raises:
        ValueError: If ``n`` or ``p`` is less than 1.
    """
    if n < 1 or p < 1:
        raise ValueError("n and p must be >= 1")
    return 2.0 * float(n) * float(p)


def ridge_predict_bytes(n: int, p: int) -> float:
    """Memory traffic to predict ``n`` rows: read the design matrix and coefficients, write the output.

    ``8 * (n p + p + n)`` bytes for float64, assuming each value is touched once and the
    coefficients stay in cache.

    Raises:
        ValueError: If ``n`` or ``p`` is less than 1.
    """
    if n < 1 or p < 1:
        raise ValueError("n and p must be >= 1")
    return float(BYTES_PER_FLOAT64) * (float(n) * float(p) + float(p) + float(n))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte of memory traffic.

    Raises:
        ValueError: If ``bytes_moved`` is not positive or ``flops`` is negative.
    """
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    if flops < 0:
        raise ValueError("flops must be non-negative")
    return float(flops) / float(bytes_moved)


def time_callable(fn: Callable[[], Any], repeats: int = 7, warmup: int = 1) -> Dict[str, float]:
    """Time ``fn`` several times and summarise.

    The **minimum** is the headline figure: it is the run least disturbed by other activity
    on the machine. The spread shows how trustworthy that is.

    Args:
        fn: Zero-argument callable to time.
        repeats: Number of timed runs, ``>= 1``.
        warmup: Untimed runs first, to let caches and any lazy imports settle.

    Returns:
        Dict with ``min``, ``median``, ``mean``, ``std`` (0.0 for a single repeat) and
        ``repeats``, in seconds.

    Raises:
        ValueError: If ``repeats < 1`` or ``warmup < 0``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"min": min(times), "median": statistics.median(times), "mean": statistics.fmean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0, "repeats": float(repeats)}


def peak_memory(fn: Callable[[], Any]) -> Tuple[Any, int]:
    """Run ``fn`` and report the peak additional memory it allocated, in bytes.

    Uses :mod:`tracemalloc`, which tracks Python and NumPy allocations but not memory
    allocated inside compiled extensions that bypass the Python allocator. Treat the figure
    as a good lower bound rather than the process's true peak.

    Returns:
        ``(result_of_fn, peak_bytes)``.
    """
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    result = fn()
    peak = tracemalloc.get_traced_memory()[1]
    if not already:
        tracemalloc.stop()
    return result, max(0, peak - before)


def model_size_bytes(model: Any) -> int:
    """Serialised size of a fitted model, as a stand-in for what must be stored or shipped.

    Raises:
        ValueError: If the model cannot be pickled.
    """
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception as exc:
        raise ValueError(f"model cannot be pickled: {exc}") from exc
