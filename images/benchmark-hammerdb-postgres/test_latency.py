import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import hammerdb_timing_latency_ms  # noqa: E402

HAMMERDB_SAMPLE = """\
TEST RESULT : System achieved 341602 NOPM from 784938 PostgreSQL TPM
SC_TIMING_JSON_START
{
  "NEWORD": {
    "p99_ms": "13.309",
    "p95_ms": "9.818",
    "p50_ms": "5.354",
    "avg_ms": "5.765",
    "min_ms": "0.586",
    "max_ms": "108.869",
    "ratio_pct": "52.308"
  },
  "PAYMENT": {
    "p99_ms": "8.856",
    "p95_ms": "6.463",
    "p50_ms": "3.422",
    "avg_ms": "3.696",
    "min_ms": "0.419",
    "max_ms": "101.212",
    "ratio_pct": "33.668"
  }
}
SC_TIMING_JSON_END
"""


def test_hammerdb_timing_latency_ms():
    latency = hammerdb_timing_latency_ms(HAMMERDB_SAMPLE)
    assert latency is not None
    assert latency["p99"] > latency["p95"] > latency["p50"]
    assert latency["min"] == 0.419
    assert latency["max"] == 108.869
