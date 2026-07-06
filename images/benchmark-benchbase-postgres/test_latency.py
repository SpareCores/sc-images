import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import benchbase_latency_ms  # noqa: E402

BENCHBASE_SAMPLE = {
    "Throughput (requests/second)": 1343.9,
    "Latency Distribution": {
        "Median Latency (microseconds)": 25201,
        "95th Percentile Latency (microseconds)": 210714,
        "99th Percentile Latency (microseconds)": 362173,
        "Average Latency (microseconds)": 55535,
        "Minimum Latency (microseconds)": 200,
        "Maximum Latency (microseconds)": 2114342,
    },
}


def test_benchbase_latency_ms():
    latency = benchbase_latency_ms(BENCHBASE_SAMPLE)
    assert latency == {
        "p50": 25.201,
        "p95": 210.714,
        "p99": 362.173,
        "avg": 55.535,
        "min": 0.2,
        "max": 2114.342,
    }
