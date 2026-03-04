#!/usr/bin/env python3
"""
Run curated stress-ng stressors with --metrics, parse output, emit JSON.

Three stressor categories:
  - Compute: CPU integer/float/SIMD, memory bandwidth, crypto, algorithms
  - Hypervisor: syscall/trap/VM-exit/paging/timer overhead
  - Multi-core: subset run with both 1 and N physical CPUs

On failure (exit code, timeout, parse error) the entry is None.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any

# --- Stressor sets ---

COMPUTE_CPU_INTEGER = [
    "cpu", "branch", "atomic", "funccall", "funcret",
    "hash", "rotate", "str", "wcs", "nop",
    "bitops", "intmath", "prime",
]

COMPUTE_CPU_FLOAT_SIMD = [
    "fma", "fp", "trig", "matrix", "matrix-3d", "eigen",
    "vecmath", "vecfp", "vecshuf", "vecwide", "monte-carlo", "mpfr",
    "expmath", "logmath", "powmath", "hyperbolic", "fractal", "veccmp",
]

COMPUTE_CRYPTO = ["crypt", "ipsec-mb"]

COMPUTE_COMPRESSION = ["zlib", "jpeg"]

COMPUTE_MEMORY = [
    "cache", "cacheline", "l1cache", "memcpy", "memrate",
    "memthrash", "stream", "vm",
    "cachehammer", "ptr-chase", "spinmem",
]

COMPUTE_ALGORITHMS = [
    "bsearch", "hsearch", "qsort", "heapsort", "mergesort",
    "bitonicsort", "skiplist", "tree", "malloc", "mcontend",
    "insertionsort", "bubblesort", "fibsearch",
]

# Stressors whose throughput is dominated by hypervisor overhead:
# VM exits, nested page-table walks, timer emulation, trap injection.
# Comparing these across providers reveals virtualization quality.
HYPERVISOR = [
    # Context switching / process creation  (VM state save/restore)
    "context", "switch", "clone", "fork", "vfork",
    # Page faults / TLB  (EPT/NPT nested walk overhead)
    "fault", "tlb-shootdown", "mprotect", "munmap",
    # Minimal-kernel-I/O  (raw syscall round-trip cost)
    "null", "zero", "getrandom",
    # IPC through kernel  (two syscalls per round-trip)
    "pipe", "eventfd",
    # Timer virtualization
    "hrtimers", "clock",
    # Scheduling
    "yield", "resched",
    # Privileged / arch-specific  (direct VM-exit / vDSO paths, x86 only → None on ARM)
    "vdso", "usersyscall", "priv-instr", "tsc", "x86syscall",
    "cpu-sched", "timermix", "mtx",
]

CATEGORIES = {
    "cpu_integer": COMPUTE_CPU_INTEGER,
    "cpu_float_simd": COMPUTE_CPU_FLOAT_SIMD,
    "crypto": COMPUTE_CRYPTO,
    "compression": COMPUTE_COMPRESSION,
    "memory": COMPUTE_MEMORY,
    "algorithms": COMPUTE_ALGORITHMS,
    "hypervisor": HYPERVISOR,
}

ALL_STRESSORS = sorted({s for group in CATEGORIES.values() for s in group})

# Also run these with all physical CPUs to capture multi-core scaling
MULTI_CORE_COMPUTE = frozenset({
    "cpu", "matrix", "cache", "stream", "memcpy",
    "vecmath", "zlib", "hash", "crypt", "malloc",
})

# Hypervisor stressors worth running concurrently: contention on nested
# page tables, vCPU scheduling, cross-core TLB invalidation, IPC paths
MULTI_CORE_HYPERVISOR = frozenset({
    "context", "switch", "clone", "fork", "vfork",
    "fault", "tlb-shootdown", "mprotect", "munmap",
    "pipe", "eventfd",
    "yield", "resched",
})

MULTI_CORE = MULTI_CORE_COMPUTE | MULTI_CORE_HYPERVISOR

DEFAULT_TIMEOUT_SEC = 5
LONGER_TIMEOUT = {"stream": 15}

METRC_LINE = re.compile(
    r"stress-ng:\s+metrc:\s+\[\d+\]\s+(\S+)\s+"
    r"[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+"
    r"([\d.]+)"
    r"(?:\s+[\d.]+)*"
)


def get_physical_cpus() -> int:
    avail = len(os.sched_getaffinity(0))
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        return min(phys, avail) if phys else avail
    except ImportError:
        return avail


def get_available_stressors() -> set[str]:
    r = subprocess.run(
        ["stress-ng", "--stressors"],
        capture_output=True, text=True, timeout=10,
    )
    return set(r.stdout.strip().split()) if r.returncode == 0 else set()


def run_stressor(name: str, workers: int, timeout_sec: int) -> dict[str, Any] | None:
    cmd = [
        "stress-ng", "--metrics", "--metrics-brief",
        f"--{name}", str(workers),
        "-t", str(timeout_sec),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec + 30,
        )
        out = proc.stdout + "\n" + proc.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    return parse_metrics(name, out)


def parse_metrics(name: str, text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if "metrc:" not in line:
            continue
        m = METRC_LINE.match(line.strip())
        if m and m.group(1) == name:
            try:
                return {"bogo_ops_per_sec": float(m.group(2))}
            except ValueError:
                return None
    return None


def main() -> int:
    available = get_available_stressors()
    if not available:
        json.dump({"error": "could not get stressor list"}, sys.stderr)
        return 1

    physical_cpus = get_physical_cpus()
    quick = os.environ.get("STRESSNG_QUICK") == "1"
    base_timeout = 2 if quick else DEFAULT_TIMEOUT_SEC

    stressors = [s for s in ALL_STRESSORS if s in available]

    limit_env = os.environ.get("STRESSNG_LIMIT")
    if limit_env:
        try:
            stressors = stressors[: int(limit_env)]
        except ValueError:
            pass

    results: dict[str, Any] = {
        "stressors": {},
        "categories": {k: [s for s in v if s in available] for k, v in CATEGORIES.items()},
        "meta": {
            "timeout_sec_per_stressor": base_timeout,
            "physical_cpus": physical_cpus,
            "stressor_count": len(stressors),
        },
    }

    for name in stressors:
        t = LONGER_TIMEOUT.get(name, base_timeout)
        entry: dict[str, Any] = {}
        entry["1"] = run_stressor(name, 1, t)
        if physical_cpus > 1:
            entry[str(physical_cpus)] = run_stressor(name, physical_cpus, t)
        results["stressors"][name] = entry

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
