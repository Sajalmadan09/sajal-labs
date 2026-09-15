import json
import re
import subprocess
import time

import numpy as np

WARMUP = 50
ITERS = 500

RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


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


def run_timed(cmd, cwd=None):
    """Run cmd wrapped in `/usr/bin/time -l` (macOS), parse the JSON result
    line the program printed plus its peak RSS from the timing report."""
    proc = subprocess.run(["/usr/bin/time", "-l", *cmd], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed:\n{proc.stdout}\n{proc.stderr}")
    json_line = next(line for line in proc.stdout.splitlines() if line.strip().startswith("{"))
    result = json.loads(json_line)
    m = RSS_RE.search(proc.stderr)
    result["peak_rss_mb"] = int(m.group(1)) / 1e6 if m else None
    return result


def dir_size_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
