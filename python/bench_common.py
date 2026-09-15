import time

import numpy as np

WARMUP = 50
ITERS = 500


def percentiles(latencies_ms):
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def time_calls(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    latencies_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms
