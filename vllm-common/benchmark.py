#!/usr/bin/env python3
"""vLLM serving benchmark: start vllm serve, run GuideLLM, emit JSONL metrics."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import signal
import tempfile
from collections import Counter
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import cache
from logging import DEBUG, StreamHandler, basicConfig, getLogger
from os import environ
from pathlib import Path
from shutil import disk_usage
from subprocess import DEVNULL, Popen, TimeoutExpired, run
from sys import exit as sys_exit
from sys import stderr, stdout
from time import monotonic, sleep
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from psutil import virtual_memory

basicConfig(
    level=DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler(stderr)],
)
logger = getLogger("benchmark-vllm")

VLLM_PORT = 8000
HEALTH_URL = f"http://127.0.0.1:{VLLM_PORT}/health"
TARGET_URL = f"http://127.0.0.1:{VLLM_PORT}"
OVERALL_TIMEOUT_SEC = 2 * 60 * 60
SERVER_START_TIMEOUT_GPU_SEC = 5 * 60
SERVER_START_TIMEOUT_CPU_SEC = 10 * 60
SERVER_START_TIMEOUT_PROBE_GPU_SEC = 8 * 60
SERVER_START_TIMEOUT_PROBE_CPU_SEC = 15 * 60
MIN_OUTPUT_TOKENS_PER_SEC = 1.0
TUNING_VERSION = 8
BUDGET_RESERVE_STARTUP_SEC = 600
BUDGET_MIN_PER_RUN_SEC = 45
BUDGET_MAX_PER_RUN_SEC = 240
BUDGET_MODEL_START_CPU_SEC = 120
BUDGET_MODEL_START_GPU_SEC = 60
# Cap CPU data-parallel replicas so vLLM startup stays inside the per-model reserve.
MAX_CPU_DP = 32
# Throughput-stage ramp must stay a small fraction of --max-seconds (GuideLLM counts
# rampup inside the stage budget). Growing rampup with vCPU was the ~96-core cliff.
RAMPUP_MAX_SEC = 8.0
RAMPUP_MIN_SEC = 2.0
RAMPUP_FRACTION = 0.15
MIN_THROUGHPUT_MEASURE_SEC = 25
# Small dense models stop scaling OpenMP threads well past this; add DP replicas instead.
CPU_THREADS_PER_RANK_TINY = 16
CPU_THREADS_PER_RANK_SMALL = 32
CPU_THREADS_PER_RANK_MEDIUM = 64
CPU_THREADS_PER_RANK_LARGE = 128
# Share of a NUMA node's *currently* free RAM all colocated vLLM ranks may reserve.
# Must stay well under 1.0: workers allocate KV sequentially, so the last rank needs
# a full util×MemTotal chunk still free (c8i.96xlarge DP=24 / 4 ranks/node OOMed at 0.85).
CPU_NODE_MEMORY_BUDGET = 0.70
# Extra derate when multiple ranks share a node (warmup RSS + reclaim lag).
CPU_NODE_MULTI_RANK_UTIL_FACTOR = 0.90

cli_parser = ArgumentParser(description="Benchmark vLLM LLM serving with GuideLLM")
cli_parser.add_argument("--version", action="store_true", help="Print versions and exit")
cli_parser.add_argument(
    "--models",
    nargs="+",
    default=None,
    help="HuggingFace model IDs (overrides default ladder).",
)
cli_parser.add_argument(
    "--models-dir",
    type=str,
    default="/models",
    help="HuggingFace hub cache directory.",
)
cli_parser.add_argument(
    "--benchmark-timeout-scale",
    type=int,
    default=1,
    help="Scale GuideLLM per-run time limits.",
)
cli_parser.add_argument(
    "--probe-only",
    action="store_true",
    help="Start smallest model, wait for /health, exit (no GuideLLM).",
)
cli_parser.add_argument(
    "--plan-only",
    action="store_true",
    help="Resolve model metadata and print host-fit JSON without starting vLLM.",
)
# Importable from unit tests (pytest/unittest pass extra argv).
cli_args = cli_parser.parse_args(None if __name__ == "__main__" else [])


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    prompt_tokens: int
    output_tokens: int
    max_model_len: int
    gpu_only: bool = False


WORKLOADS: list[WorkloadSpec] = [
    WorkloadSpec("chat", 256, 128, 2048),
    WorkloadSpec("rag", 1024, 256, 4096),
    WorkloadSpec("long", 4096, 512, 8192, gpu_only=True),
]


@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    model_id: str
    params_b: float
    memory_gb: float | None = None
    num_attention_heads: int | None = None
    serve_extra_args: tuple[str, ...] = ()
    gpu_only: bool = False
    cpu_only: bool = False


@dataclass(frozen=True)
class ModelMetadata:
    """Authoritative model footprint inputs resolved from the Hugging Face repo."""

    weight_bytes: int
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    hidden_size: int | None
    torch_dtype: str | None
    source: str
    head_dim: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "weight_bytes": self.weight_bytes,
            "weight_gib": self.weight_bytes / (1024**3),
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_size": self.hidden_size,
            "torch_dtype": self.torch_dtype,
            "head_dim": self.head_dim,
            "source": self.source,
        }


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        "smol-135m",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        0.135,
        num_attention_heads=9,
    ),
    ModelSpec(
        "qwen-0.5b",
        "Qwen/Qwen2.5-0.5B-Instruct",
        0.5,
        num_attention_heads=14,
    ),
    ModelSpec(
        "gemma-2b",
        "google/gemma-2-2b-it",
        2.0,
        num_attention_heads=8,
    ),
    ModelSpec(
        "llama-8b",
        "meta-llama/Llama-3.1-8B-Instruct",
        8.0,
        num_attention_heads=32,
    ),
    ModelSpec(
        "phi-4",
        "microsoft/phi-4",
        14.0,
        num_attention_heads=40,
    ),
    ModelSpec(
        "llama-70b",
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        70.0,
        memory_gb=48.0,
        serve_extra_args=(
            "--quantization",
            "bitsandbytes",
            "--load-format",
            "bitsandbytes",
        ),
    ),
]

# (short_name, guidellm metric key, emitted unit, scale to emitted unit)
LATENCY_METRICS = (
    ("ttft", "time_to_first_token_ms", "ms", 1.0),
    ("tpot", "time_per_output_token_ms", "ms", 1.0),
    ("itl", "inter_token_latency_ms", "ms", 1.0),
    ("e2el", "request_latency", "ms", 1000.0),  # GuideLLM: seconds → ms
)

THROUGHPUT_METRICS = (
    ("output_throughput", "output_tokens_per_second", "tokens/sec"),
    ("total_throughput", "tokens_per_second", "tokens/sec"),
    ("request_throughput", "requests_per_second", "requests/sec"),
)

PERCENTILES = ("p50", "p95", "p99")


@dataclass(frozen=True)
class HostProfile:
    vcpus: int
    ram_total_gb: float
    ram_avail_gb: float


@dataclass(frozen=True)
class BudgetPlan:
    per_run_sec: int
    total_runs: int
    overall_timeout_sec: int
    reserve_sec: int


@dataclass(frozen=True)
class BenchmarkTuning:
    tuning_version: int
    autoconfig: bool
    sweep_size: int
    max_concurrency: int
    max_requests: int | None
    max_workers: int
    rampup_duration: float
    warmup: str
    max_seconds_per_strategy: int
    per_run_budget_sec: int
    max_num_seqs: int
    max_num_batched_tokens: int
    dtype: str
    kv_memory_util: float
    gpu_memory_util: float
    kv_cache_gib: int | None = None
    max_model_len: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "tuning_version": self.tuning_version,
            "autoconfig": self.autoconfig,
            "sweep_size": self.sweep_size,
            "max_concurrency": self.max_concurrency,
            "max_requests": self.max_requests,
            "max_workers": self.max_workers,
            "rampup_duration": self.rampup_duration,
            "warmup": self.warmup,
            "max_seconds_per_strategy": self.max_seconds_per_strategy,
            "per_run_budget_sec": self.per_run_budget_sec,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "dtype": self.dtype,
            "kv_memory_util": self.kv_memory_util,
            "gpu_memory_util": self.gpu_memory_util,
        }
        if self.kv_cache_gib is not None:
            out["kv_cache_gib"] = self.kv_cache_gib
        if self.max_model_len is not None:
            out["max_model_len"] = self.max_model_len
        return out


_HOST: HostProfile | None = None
_BUDGET: BudgetPlan | None = None
_TUNING: BenchmarkTuning | None = None
_SERVER_STDERR_PATH: Path | None = None
_EMITTED_ROWS = 0


@cache
def read_pin(filename: str, env_key: str, default: str = "unknown") -> str:
    path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fp:
            return fp.read().strip()
    return environ.get(env_key, default)


def read_vllm_version() -> str:
    return read_pin("VLLM_VERSION", "VLLM_VERSION")


def read_guidellm_version() -> str:
    return read_pin("GUIDELLM_VERSION", "GUIDELLM_VERSION")


def get_vllm_runtime_version() -> str:
    """Best-effort runtime version; prefer read_vllm_version() for stable reporting."""
    env = {**os.environ, "VLLM_CONFIGURE_LOGGING": "0"}
    result = run(
        ["vllm", "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    for text in (result.stdout, result.stderr):
        if not text:
            continue
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if not line or line.startswith("INFO ") or "Triton is installed" in line:
                continue
            # argparse --version prints e.g. "0.22.1"
            if line[0].isdigit() or (line.startswith("v") and line[1:2].isdigit()):
                return line.lstrip("v")
    return read_vllm_version()


def guidellm_runtime_version() -> str:
    result = run(
        ["guidellm", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=guidellm_env(),
    )
    if result.returncode == 0:
        return read_guidellm_version()
    return read_guidellm_version()


def guidellm_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("USER", "benchmark")
    env.setdefault("HOME", "/tmp")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_inductor")
    tuning = _TUNING
    if tuning and tuning.autoconfig:
        if "GUIDELLM__MAX_WORKER_PROCESSES" not in env:
            env["GUIDELLM__MAX_WORKER_PROCESSES"] = str(tuning.max_workers)
        if "GUIDELLM__MAX_CONCURRENCY" not in env:
            env["GUIDELLM__MAX_CONCURRENCY"] = str(tuning.max_concurrency)
    return env


def autoconfig_enabled() -> bool:
    return environ.get("BENCHMARK_VLLM_AUTOCONFIG", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def per_workload_server_enabled() -> bool:
    """Restart vLLM per workload with workload-specific max_model_len."""
    default = "1" if autoconfig_enabled() else "0"
    return environ.get("BENCHMARK_VLLM_PER_WORKLOAD_SERVER", default).lower() not in (
        "0",
        "false",
        "no",
    )


def parse_cpu_id_list(raw: str) -> list[int]:
    """Parse ``0-3,8`` / ``0-31`` strings into CPU ids (vLLM bind format)."""
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def compact_cpu_ids(ids: list[int]) -> str:
    """Compress CPU ids to vLLM bind fragments (``0-3,8``)."""
    if not ids:
        return ""
    ordered = sorted(set(ids))
    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu_id in ordered[1:]:
        if cpu_id == prev + 1:
            prev = cpu_id
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu_id
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def host_vcpus() -> int:
    override = environ.get("BENCHMARK_VLLM_VCPUS", "").strip()
    if override:
        return max(1, int(override))
    topo = environ.get("BENCHMARK_VLLM_CPU_TOPOLOGY", "").strip()
    if topo:
        n = sum(len(parse_cpu_id_list(part)) for part in topo.split("|") if part.strip())
        if n:
            return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


@cache
def host_numa_count() -> int:
    """Number of NUMA nodes visible to this process (CPU topology)."""
    override = environ.get("BENCHMARK_VLLM_NUMA_NODES", "").strip()
    if override:
        return max(1, int(override))
    topo = environ.get("BENCHMARK_VLLM_CPU_TOPOLOGY", "").strip()
    if topo:
        return max(1, len([p for p in topo.split("|") if p.strip()]))
    base = Path("/sys/devices/system/node")
    if base.is_dir():
        nodes = [p for p in base.glob("node[0-9]*") if p.is_dir()]
        if nodes:
            return max(1, len(nodes))
    return 1


@cache
def host_cpus_by_numa() -> tuple[tuple[int, ...], ...]:
    """Allowed CPU ids grouped by NUMA node (logical CPUs, including SMT)."""
    topo = environ.get("BENCHMARK_VLLM_CPU_TOPOLOGY", "").strip()
    if topo:
        nodes = []
        for part in topo.split("|"):
            ids = parse_cpu_id_list(part)
            if ids:
                nodes.append(tuple(ids))
        if nodes:
            return tuple(nodes)
    affinity: set[int]
    try:
        affinity = set(os.sched_getaffinity(0))
    except AttributeError:
        affinity = set(range(max(1, os.cpu_count() or 1)))
    base = Path("/sys/devices/system/node")
    nodes: list[tuple[int, ...]] = []
    if base.is_dir():
        for node_dir in sorted(base.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
            cpulist = node_dir / "cpulist"
            if not cpulist.is_file():
                continue
            try:
                raw = cpulist.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            ids = [cpu_id for cpu_id in parse_cpu_id_list(raw) if cpu_id in affinity]
            if ids:
                nodes.append(tuple(ids))
    if nodes:
        return tuple(nodes)
    return (tuple(sorted(affinity)),)


def reset_autoconfig_state() -> None:
    """Clear cached host/budget/tuning (for tests and env-var overrides)."""
    global _HOST, _BUDGET, _TUNING
    _HOST = None
    _BUDGET = None
    _TUNING = None
    host_numa_count.cache_clear()
    host_cpus_by_numa.cache_clear()
    get_cpu_flags.cache_clear()
    gpu_info.cache_clear()


def host_profile() -> HostProfile:
    global _HOST
    if _HOST is None:
        mem = virtual_memory()
        _HOST = HostProfile(
            vcpus=host_vcpus(),
            ram_total_gb=mem.total / 1e9,
            ram_avail_gb=mem.available / 1e9,
        )
    return _HOST


def sublinear_scale(vcpus: int, base: float, exp: float, floor: int) -> int:
    return max(floor, int(base * max(vcpus, 1) ** exp))


def _split_even(items: list[int], parts: int) -> list[list[int]]:
    """Contiguous nearly-equal chunks; empty only when ``len(items) < parts``."""
    n = max(1, parts)
    if not items:
        return [[] for _ in range(n)]
    base, extra = divmod(len(items), n)
    out: list[list[int]] = []
    idx = 0
    for i in range(n):
        take = base + (1 if i < extra else 0)
        out.append(items[idx : idx + take])
        idx += take
    return out


def _ensure_rank_cpus(groups: list[list[int]]) -> list[list[int]]:
    """Give empty ranks a CPU stolen from a rank with at least two."""
    for i, group in enumerate(groups):
        if group:
            continue
        donor = max(range(len(groups)), key=lambda j: len(groups[j]))
        if len(groups[donor]) >= 2:
            groups[i].append(groups[donor].pop())
    return groups


def partition_cpus_for_ranks(
    cpus_by_numa: list[list[int]],
    tp: int,
    dp: int,
    *,
    reserve: int = 0,
) -> list[list[int]]:
    """Split logical CPUs into ``tp * dp`` NUMA-local groups (DP-major, then TP).

    vLLM parses ``VLLM_CPU_OMP_THREADS_BIND`` as ``|``-separated lists and slices
    ``local_world_size`` (TP) entries per local DP rank. Keep every logical CPU
    (including SMT siblings) — unlike ``auto``, which drops SMT on x86.
    """
    tp = max(1, tp)
    dp = max(1, dp)
    world = tp * dp
    nodes = [list(node) for node in cpus_by_numa if node]
    if not nodes:
        return [[] for _ in range(world)]
    all_cpus = [cpu_id for node in nodes for cpu_id in node]
    if reserve and len(all_cpus) - reserve >= world:
        reserved = set(all_cpus[-reserve:])
        nodes = [[cpu_id for cpu_id in node if cpu_id not in reserved] for node in nodes]
        nodes = [node for node in nodes if node]
    groups: list[list[int]] = [[] for _ in range(world)]
    if not nodes:
        return groups
    if tp == 1:
        nonempty = [node for node in nodes if node]
        if dp <= len(nonempty):
            node_chunks = _split_even(list(range(len(nonempty))), dp)
            for dp_rank, node_idxs in enumerate(node_chunks):
                groups[dp_rank] = [cpu_id for i in node_idxs for cpu_id in nonempty[i]]
        else:
            rank_chunks = _split_even(list(range(dp)), len(nonempty))
            for node_idx, rank_ids in enumerate(rank_chunks):
                slices = _split_even(nonempty[node_idx], len(rank_ids))
                for rank_id, slice_cpus in zip(rank_ids, slices, strict=True):
                    groups[rank_id] = slice_cpus
    else:
        numa = len(nodes)
        for tp_rank in range(tp):
            home = [nodes[i] for i in range(numa) if i % tp == tp_rank]
            pool = [cpu_id for node in home for cpu_id in node]
            for dp_rank, slice_cpus in enumerate(_split_even(pool, dp)):
                groups[dp_rank * tp + tp_rank] = slice_cpus
    return _ensure_rank_cpus(groups)


def cpu_tensor_parallel_size(spec: ModelSpec) -> int:
    """TP across NUMA nodes when attention heads divide evenly by NUMA count.

    Official CPU guidance: set tensor-parallel-size to the NUMA node count so
    each rank stays NUMA-local. If heads are not divisible by NUMA count, keep
    TP=1 and replicate with data-parallel instead (see ``cpu_data_parallel_size``).
    """
    override = _env_int("BENCHMARK_VLLM_CPU_TP", "VLLM_CPU_TENSOR_PARALLEL_SIZE")
    if override is not None:
        return max(1, override)
    numa = host_numa_count()
    if numa <= 1:
        return 1
    heads = model_metadata(spec).num_attention_heads or spec.num_attention_heads
    if heads and heads % numa == 0:
        return numa
    return 1


def cpu_target_threads_per_rank(spec: ModelSpec) -> int:
    """OpenMP threads per CPU rank; small models need fatter DP, not fatter OMP."""
    if spec.params_b <= 0.5:
        return CPU_THREADS_PER_RANK_TINY
    if spec.params_b <= 2.0:
        return CPU_THREADS_PER_RANK_SMALL
    if spec.params_b <= 8.0:
        return CPU_THREADS_PER_RANK_MEDIUM
    return CPU_THREADS_PER_RANK_LARGE


def cpu_data_parallel_size(spec: ModelSpec) -> int:
    """DP replicas: at least one per unused NUMA node, more for small models.

    A single CPU rank does not scale to hundreds of OpenMP threads (especially
    135M/0.5B). Extra DP ranks split the machine into NUMA-local replicas so
    throughput can grow from 1 to 896 vCPUs. Each replica loads a full copy of
    the weights, so DP is also capped by available RAM.
    """
    override = _env_int("BENCHMARK_VLLM_CPU_DP", "VLLM_CPU_DATA_PARALLEL_SIZE")
    if override is not None:
        return max(1, override)
    v = host_profile().vcpus
    numa = host_numa_count()
    tp = cpu_tensor_parallel_size(spec)
    # Cover every socket when TP cannot span them.
    base_dp = numa if tp < numa else 1
    target_threads = max(1, cpu_target_threads_per_rank(spec))
    desired = max(base_dp, v // (tp * target_threads) or 1)
    dp = max(1, min(desired, max(1, v // tp), MAX_CPU_DP))
    # Exact repository weight bytes + config-derived minimum KV, checked on the
    # NUMA node where every worker will actually allocate.
    while dp > 1 and not cpu_layout_memory_plan(
        spec,
        tp=tp,
        dp=dp,
        max_model_len=min(w.max_model_len for w in workloads_for_mode("cpu")),
    )["fits"]:
        dp -= 1
    return max(1, dp)


def cpu_omp_threads_bind(spec: ModelSpec, tp: int | None = None, dp: int | None = None) -> str:
    """Explicit CPU lists for every TP×DP rank (all logical CPUs, NUMA-local).

    ``VLLM_CPU_OMP_THREADS_BIND=auto`` on x86 keeps one thread per physical core
    (drops SMT) and, with world_size=1, binds rank 0 to NUMA node 0 only. Cloud
    vCPU counts include SMT, so auto silently uses ~half the advertised cores
    and leaves extra sockets idle unless TP/DP already span NUMA. Explicit lists
    keep SMT and partition every allowed CPU across ranks. ``auto`` / empty
    env means "let the harness decide"; any other value is passed through.
    """
    override = environ.get("VLLM_CPU_OMP_THREADS_BIND", "").strip()
    if override and override.lower() not in ("auto",):
        return override
    tp_size = tp if tp is not None else cpu_tensor_parallel_size(spec)
    dp_size = dp if dp is not None else cpu_data_parallel_size(spec)
    nodes = [list(node) for node in host_cpus_by_numa()]
    reserve = 1 if host_profile().vcpus >= 2 else 0
    groups = partition_cpus_for_ranks(nodes, tp_size, dp_size, reserve=reserve)
    if any(not group for group in groups):
        groups = partition_cpus_for_ranks(nodes, tp_size, dp_size, reserve=0)
    parts = [compact_cpu_ids(group) for group in groups]
    if not parts or any(not part for part in parts):
        return "auto"
    return "|".join(parts)


def cpu_ranks_per_memory_node(
    spec: ModelSpec,
    tp: int | None = None,
    dp: int | None = None,
) -> int:
    """Largest number of vLLM CPU workers that share one NUMA memory node.

    ``CPUWorker`` binds each rank to the NUMA node of its *first* allowed CPU and
    reserves ``--gpu-memory-utilization × that node's total RAM`` per rank, so
    co-located ranks must split the fraction or every rank after the first dies
    with "Available memory ... is less than desired CPU memory utilization".
    """
    tp_size = max(1, tp if tp is not None else cpu_tensor_parallel_size(spec))
    dp_size = max(1, dp if dp is not None else cpu_data_parallel_size(spec))
    world = tp_size * dp_size
    bind = cpu_omp_threads_bind(spec, tp_size, dp_size).strip().lower()
    if bind in ("", "auto"):
        # vLLM's own `auto` spreads ranks over the NUMA nodes it can see.
        return max(1, math.ceil(world / max(1, host_numa_count())))
    if bind == "nobind":
        # Every rank inherits the full affinity mask, hence the same first CPU.
        return world
    node_of_cpu = {
        cpu_id: index
        for index, node in enumerate(host_cpus_by_numa())
        for cpu_id in node
    }
    per_node: dict[int, int] = {}
    for part in bind.split("|"):
        ids = parse_cpu_id_list(part)
        if not ids:
            continue
        node = node_of_cpu.get(ids[0], 0)
        per_node[node] = per_node.get(node, 0) + 1
    return max(per_node.values(), default=1)


def cpu_rank_counts_by_memory_node(
    spec: ModelSpec,
    tp: int | None = None,
    dp: int | None = None,
) -> dict[int, int]:
    """Worker count by NUMA memory node for the effective binding."""
    tp_size = max(1, tp if tp is not None else cpu_tensor_parallel_size(spec))
    dp_size = max(1, dp if dp is not None else cpu_data_parallel_size(spec))
    world = tp_size * dp_size
    nodes = host_cpus_by_numa()
    node_of_cpu = {
        cpu_id: index
        for index, node in enumerate(nodes)
        for cpu_id in node
    }
    bind = cpu_omp_threads_bind(spec, tp_size, dp_size).strip().lower()
    if bind == "nobind":
        first_cpu = min((cpu for node in nodes for cpu in node), default=0)
        return {node_of_cpu.get(first_cpu, 0): world}
    if bind in ("", "auto"):
        return dict(Counter(rank % max(1, len(nodes)) for rank in range(world)))
    counts: Counter[int] = Counter()
    for part in bind.split("|"):
        ids = parse_cpu_id_list(part)
        if ids:
            counts[node_of_cpu.get(ids[0], 0)] += 1
    return dict(counts) or {0: world}


def host_memory_by_numa_gb() -> tuple[tuple[float, float], ...]:
    """(total, available) GiB for each visible NUMA node.

    Available is read live (not cached): per-workload server restarts must see
    reclaim after the previous vLLM exit, matching vLLM's own node meminfo math.
    """
    if environ.get("BENCHMARK_VLLM_CPU_TOPOLOGY", "").strip():
        total, available = current_ram_gb()
        count = max(1, host_numa_count())
        return tuple((total / count, available / count) for _ in range(count))
    out: list[tuple[float, float]] = []
    for index, _cpus in enumerate(host_cpus_by_numa()):
        path = Path(f"/sys/devices/system/node/node{index}/meminfo")
        try:
            values: dict[str, float] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                match = re.search(r"Node\s+\d+\s+(\S+):\s+(\d+)\s+kB", line)
                if match:
                    values[match.group(1)] = int(match.group(2)) / (1024**2)
            total = values.get("MemTotal", 0.0)
            # Same reclaimable sum vLLM uses in get_memory_node_info().
            available = (
                values.get("MemFree", 0.0)
                + values.get("SReclaimable", 0.0)
                + values.get("Active(file)", 0.0)
                + values.get("Inactive(file)", 0.0)
            )
            if total > 0:
                out.append((total, max(0.0, min(total, available))))
        except OSError:
            pass
    if out:
        return tuple(out)
    total, available = current_ram_gb()
    count = max(1, host_numa_count())
    return tuple((total / count, available / count) for _ in range(count))


def cpu_layout_memory_plan(
    spec: ModelSpec,
    *,
    tp: int,
    dp: int,
    max_model_len: int,
    min_sequences: int = 1,
) -> dict[str, Any]:
    """Check weight+KV footprint against every NUMA node used by the layout."""
    rank_counts = cpu_rank_counts_by_memory_node(spec, tp, dp)
    memories = host_memory_by_numa_gb()
    per_worker = model_worker_memory_gb(
        spec,
        tp=tp,
        max_model_len=max_model_len,
        min_sequences=min_sequences,
        block_size=128,
    )
    nodes: list[dict[str, Any]] = []
    fits = True
    limiting_fraction = 0.0
    for node, ranks in sorted(rank_counts.items()):
        total, available = memories[min(node, len(memories) - 1)]
        required = ranks * per_worker
        budget = available * CPU_NODE_MEMORY_BUDGET
        node_fits = required <= budget
        fits = fits and node_fits
        limiting_fraction = max(
            limiting_fraction,
            per_worker / total if total > 0 else 1.0,
        )
        nodes.append(
            {
                "node": node,
                "ranks": ranks,
                "total_gib": total,
                "available_gib": available,
                "required_gib": required,
                "budget_gib": budget,
                "fits": node_fits,
            }
        )
    return {
        "fits": fits,
        "tp": tp,
        "dp": dp,
        "world_size": tp * dp,
        "binding": cpu_omp_threads_bind(spec, tp, dp),
        "per_worker_gib": per_worker,
        "memory_fraction": limiting_fraction,
        "nodes": nodes,
        "model": model_metadata(spec).as_dict(),
    }


def runnable_models(mode: str) -> list[ModelSpec]:
    return [
        spec
        for spec in models_to_run(mode)
        if model_supported_on_mode(spec, mode) and model_fits(spec, mode)
    ]


def compute_budget(mode: str) -> BudgetPlan:
    models = runnable_models(mode)
    workloads = workloads_for_mode(mode)
    total_runs = max(1, len(models) * len(workloads))
    overall = OVERALL_TIMEOUT_SEC * max(1, cli_args.benchmark_timeout_scale)
    start_cost = BUDGET_MODEL_START_CPU_SEC if mode == "cpu" else BUDGET_MODEL_START_GPU_SEC
    starts_per_model = len(workloads) if per_workload_server_enabled() else 1
    reserve = BUDGET_RESERVE_STARTUP_SEC + len(models) * starts_per_model * start_cost
    available = max(0, overall - reserve)
    per_run = available // total_runs
    per_run = max(BUDGET_MIN_PER_RUN_SEC, min(BUDGET_MAX_PER_RUN_SEC, per_run))
    return BudgetPlan(
        per_run_sec=per_run,
        total_runs=total_runs,
        overall_timeout_sec=overall,
        reserve_sec=reserve,
    )


def _env_int(*keys: str) -> int | None:
    for key in keys:
        raw = environ.get(key, "").strip()
        if raw:
            return int(raw)
    return None


def workload_time_factor(max_model_len: int) -> float:
    """Scale per-strategy wall time for longer contexts (rag @ 4096 vs chat @ 2048)."""
    return max(1.0, max_model_len / 2048.0)


def _fit_sweep_to_budget(
    desired_sweep: int,
    per_run_sec: int,
    *,
    min_seconds_per_strategy: int | None = None,
) -> tuple[int, int]:
    """Return (sweep_size, max_seconds_per_strategy) within per-run wall budget."""
    min_sec = min_seconds_per_strategy or (BUDGET_MIN_PER_RUN_SEC // 2)
    min_sec = min(min_sec, BUDGET_MAX_PER_RUN_SEC)
    sweep = max(2, desired_sweep)
    while sweep >= 2:
        max_seconds = max(min_sec, per_run_sec // sweep)
        max_seconds = min(max_seconds, BUDGET_MAX_PER_RUN_SEC)
        if sweep * max_seconds <= per_run_sec:
            return sweep, max_seconds
        sweep -= 1
    max_seconds = max(min_sec, min(per_run_sec, BUDGET_MAX_PER_RUN_SEC))
    return 2, max_seconds


def current_ram_gb() -> tuple[float, float]:
    mem = virtual_memory()
    return mem.total / 1e9, mem.available / 1e9


def cpu_kv_cache_gib(spec: ModelSpec, hp: HostProfile) -> int | None:
    """Autoconfig uses --gpu-memory-utilization only; set VLLM_CPU_KVCACHE_SPACE manually to override."""
    return None


def compute_tuning(
    mode: str,
    spec: ModelSpec,
    budget: BudgetPlan,
    max_model_len: int | None = None,
) -> BenchmarkTuning:
    hp = host_profile()
    v = hp.vcpus
    if max_model_len is None:
        max_model_len = max(w.max_model_len for w in workloads_for_mode(mode))

    tp = cpu_tensor_parallel_size(spec) if mode == "cpu" else 1
    dp = cpu_data_parallel_size(spec) if mode == "cpu" else 1
    replicas = max(1, tp * dp)

    if mode == "cpu":
        # Floor concurrency with v so 1–2 vCPU hosts are not oversubscribed, and
        # with replica count so extra DP ranks are not starved of in-flight load.
        conc_floor = max(1, min(32, v * 4), replicas * 4)
        max_conc = sublinear_scale(v, 6.0, 0.65, conc_floor)
        # GuideLLM workers are async; a handful can drive high concurrency. Cap
        # so the client does not steal the server's CPUs on the same box.
        max_workers = max(1, min(32, v, max(1, max_conc // 8)))
        warmup = "10"
    else:
        max_conc = sublinear_scale(v, 8.0, 0.60, 64)
        max_workers = min(sublinear_scale(v, 3.0, 0.50, 8), max(4, v))
        warmup = "0.05"

    max_requests: int | None = None

    ctx_factor = workload_time_factor(max_model_len)
    min_sec_per_strategy = min(
        BUDGET_MAX_PER_RUN_SEC,
        max(MIN_THROUGHPUT_MEASURE_SEC, int((BUDGET_MIN_PER_RUN_SEC // 2) * ctx_factor)),
    )
    # Keep sweep shallow on CPU so each stage stays long enough to measure peak
    # throughput. Growing sweep with log(vCPU) used to shrink --max-seconds just
    # as --rampup grew, which is what flattened the curve after ~96 cores.
    if mode == "cpu":
        desired_sweep = max(2, min(4, 2 + int(math.log2(max(v, 2)) // 3)))
    else:
        desired_sweep = max(
            2,
            min(
                3 + int(math.log2(max(v, 2)) // 2),
                6 + int(math.log2(max(v, 2)) // 3),
            ),
        )
    sweep, max_seconds = _fit_sweep_to_budget(
        desired_sweep,
        budget.per_run_sec,
        min_seconds_per_strategy=min_sec_per_strategy,
    )

    if sweep_env := (
        _env_int("GUIDELLM_CPU_SWEEP_SIZE", "GUIDELLM_SWEEP_SIZE")
        if mode == "cpu"
        else _env_int("GUIDELLM_GPU_SWEEP_SIZE", "GUIDELLM_SWEEP_SIZE")
    ):
        sweep = max(2, sweep_env)
        _, max_seconds = _fit_sweep_to_budget(
            sweep,
            budget.per_run_sec,
            min_seconds_per_strategy=min_sec_per_strategy,
        )
    if max_conc_env := _env_int("GUIDELLM__MAX_CONCURRENCY"):
        max_conc = max_conc_env
    if max_req_env := _env_int("GUIDELLM_MAX_REQUESTS", "GUIDELLM_MAX_REQUESTS_CPU"):
        max_requests = max_req_env
    if workers_env := _env_int("GUIDELLM__MAX_WORKER_PROCESSES"):
        max_workers = workers_env

    if rampup_env := environ.get("BENCHMARK_VLLM_RAMPUP", "").strip():
        rampup = max(0.0, float(rampup_env))
    elif mode == "cpu":
        # Rampup is inside GuideLLM's per-stage --max-seconds (first N concurrent
        # requests are staggered). Keep it a small fraction so the throughput stage
        # still measures peak load on large machines.
        measure_budget = max(0.0, float(max_seconds) - MIN_THROUGHPUT_MEASURE_SEC)
        rampup = min(
            RAMPUP_MAX_SEC,
            max(RAMPUP_MIN_SEC, RAMPUP_FRACTION * float(max_seconds)),
            measure_budget if measure_budget > 0 else RAMPUP_FRACTION * float(max_seconds),
        )
        rampup = max(0.0, min(rampup, max(0.0, float(max_seconds) - 1.0)))
    else:
        rampup = min(10.0, max(2.0, 0.15 * float(max_seconds)))

    seq_scale = min(1.0, 2048 / max(max_model_len, 512))
    if mode == "cpu":
        seqs_floor = max(1, min(16, v * 2))
        # Cap batching per DP/TP worker: --max-num-seqs is per DP rank.
        cores_per_worker = max(1, v // replicas)
        seq_cap = max(seqs_floor, cores_per_worker)
        max_num_seqs = max(
            seqs_floor,
            int(min(max_conc, sublinear_scale(v, 4.0, 0.55, seqs_floor), seq_cap) * seq_scale),
        )
    else:
        max_num_seqs = max(
            4,
            int(min(max_conc, sublinear_scale(v, 4.0, 0.55, 16)) * seq_scale),
        )
    max_batched = min(max_model_len, max(2048, sublinear_scale(v, 64.0, 0.50, 2048)))
    if seqs_env := _env_int("BENCHMARK_VLLM_MAX_NUM_SEQS"):
        max_num_seqs = max(1, seqs_env)
    if batched_env := _env_int("BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS"):
        max_batched = max(1, batched_env)

    dtype = cpu_serve_dtype(spec)
    ranks_per_node = cpu_ranks_per_memory_node(spec, tp, dp) if mode == "cpu" else 1
    kv_util = cpu_kv_memory_util(
        spec,
        hp,
        max_model_len=max_model_len,
        ranks_per_node=ranks_per_node,
    )
    gpu_util = gpu_memory_utilization(spec)
    kv_gib = cpu_kv_cache_gib(spec, hp)
    if not _env_int("BENCHMARK_VLLM_MAX_NUM_SEQS"):
        memory_seq_cap = max_sequences_for_layout(
            spec,
            mode=mode,
            tp=tp,
            dp=dp,
            max_model_len=max_model_len,
            cpu_memory_util=kv_util,
            gpu_memory_util_value=gpu_util,
        )
        max_num_seqs = max(1, min(max_num_seqs, memory_seq_cap))

    return BenchmarkTuning(
        tuning_version=TUNING_VERSION,
        autoconfig=True,
        sweep_size=sweep,
        max_concurrency=max_conc,
        max_requests=max_requests,
        max_workers=max_workers,
        rampup_duration=rampup,
        warmup=warmup,
        max_seconds_per_strategy=max_seconds,
        per_run_budget_sec=budget.per_run_sec,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_batched,
        dtype=dtype,
        kv_memory_util=kv_util,
        gpu_memory_util=gpu_util,
        kv_cache_gib=kv_gib,
        max_model_len=max_model_len,
    )


def legacy_tuning(
    mode: str,
    spec: ModelSpec,
    max_model_len: int | None = None,
) -> BenchmarkTuning:
    if max_model_len is None:
        max_model_len = max(w.max_model_len for w in workloads_for_mode(mode))
    sweep = int(
        guidellm_sweep_size(mode)
        if not _guidellm_profile_override(mode)
        else 2
    )
    return BenchmarkTuning(
        tuning_version=0,
        autoconfig=False,
        sweep_size=max(2, sweep),
        max_concurrency=512,
        max_requests=int(
            environ.get("GUIDELLM_MAX_REQUESTS", "120")
            if mode == "gpu"
            else environ.get("GUIDELLM_MAX_REQUESTS_CPU", "25")
        ),
        max_workers=10,
        rampup_duration=0.0,
        warmup="0.05",
        max_seconds_per_strategy=(
            (40 + int(spec.params_b * 8) if mode == "gpu" else 45 + int(spec.params_b * 12))
            * max(1, cli_args.benchmark_timeout_scale)
        ),
        per_run_budget_sec=0,
        max_num_seqs=128,
        max_num_batched_tokens=max_model_len,
        dtype=cpu_serve_dtype(spec),
        kv_memory_util=cpu_gpu_memory_utilization(),
        gpu_memory_util=0.9,
        max_model_len=max_model_len,
    )


def init_benchmark_tuning(
    mode: str,
    spec: ModelSpec,
    max_model_len: int | None = None,
) -> BenchmarkTuning:
    global _BUDGET, _TUNING
    if not autoconfig_enabled():
        _TUNING = legacy_tuning(mode, spec, max_model_len)
        return _TUNING
    if _BUDGET is None:
        _BUDGET = compute_budget(mode)
    _TUNING = compute_tuning(mode, spec, _BUDGET, max_model_len=max_model_len)
    logger.info(
        "autoconfig vcpus=%s numa=%s cpu_tp=%s cpu_dp=%s budget_per_run=%ss sweep=%s max_conc=%s max_req=%s "
        "workers=%s rampup=%ss max_sec/strategy=%s max_model_len=%s max_num_seqs=%s per_workload_server=%s",
        host_profile().vcpus,
        host_numa_count() if mode == "cpu" else 1,
        cpu_tensor_parallel_size(spec) if mode == "cpu" else 1,
        cpu_data_parallel_size(spec) if mode == "cpu" else 1,
        _TUNING.per_run_budget_sec,
        _TUNING.sweep_size,
        _TUNING.max_concurrency,
        _TUNING.max_requests if _TUNING.max_requests is not None else "time-only",
        _TUNING.max_workers,
        _TUNING.rampup_duration,
        _TUNING.max_seconds_per_strategy,
        _TUNING.max_model_len,
        _TUNING.max_num_seqs,
        per_workload_server_enabled(),
    )
    return _TUNING


def current_tuning(
    mode: str,
    spec: ModelSpec,
    max_model_len: int | None = None,
) -> BenchmarkTuning:
    if _TUNING is None or (
        autoconfig_enabled()
        and max_model_len is not None
        and _TUNING.max_model_len != max_model_len
    ):
        return init_benchmark_tuning(mode, spec, max_model_len=max_model_len)
    return _TUNING


def detect_mode() -> str:
    explicit = environ.get("BENCHMARK_VLLM_MODE", "").strip().lower()
    if explicit in ("gpu", "cpu"):
        return explicit
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


@cache
def get_cpu_flags() -> frozenset[str]:
    """CPU feature flags from /proc/cpuinfo (x86 ``flags`` or ARM ``Features``)."""
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("flags", "features"):
                    flags.update(line.split(":", 1)[1].strip().lower().split())
    except OSError:
        pass
    return frozenset(flags)


def cpu_has_avx512() -> bool:
    return "avx512f" in get_cpu_flags()


def cpu_has_bf16() -> bool:
    """True when the CPU can run bfloat16 matmuls (ARM ``bf16`` or x86 ``avx512_bf16``)."""
    flags = get_cpu_flags()
    return "bf16" in flags or "avx512_bf16" in flags


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return machine


def log_cpu_details() -> None:
    logger.info("arch=%s", host_arch())
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fp:
            model = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in fp
                    if line.lower().startswith("model name")
                    or line.lower().startswith("cpu part")
                ),
                "unknown",
            )
        logger.info("cpu_model=%s", model)
        flags = get_cpu_flags()
        for flag in ("avx512f", "avx512_bf16", "avx2", "asimd", "bf16"):
            if flag in flags:
                logger.info("cpu_flag_%s=yes", flag)
        logger.info("cpu_bf16=%s", cpu_has_bf16())
    except OSError as e:
        logger.debug("cpuinfo: %s", e)
    mem = virtual_memory()
    logger.info("ram_total_gb=%.2f ram_available_gb=%.2f", mem.total / 1e9, mem.available / 1e9)


def check_cpu_isa_compat(mode: str) -> None:
    if mode != "cpu":
        return
    if host_arch() != "amd64":
        return
    if environ.get("BENCHMARK_VLLM_ALLOW_AVX2_ONLY", "").lower() in ("1", "true", "yes"):
        return
    if cpu_has_avx512():
        return
    logger.error(
        "amd64 host has AVX2 but not AVX-512 (avx512f). Hub vllm-openai-cpu requires "
        "AVX-512. Use ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main."
    )
    sys_exit(2)


def cpu_server_env(
    base: dict[str, str],
    tuning: BenchmarkTuning | None = None,
    *,
    spec: ModelSpec | None = None,
) -> dict[str, str]:
    env = dict(base)
    spec = spec or DEFAULT_MODELS[0]
    tp = cpu_tensor_parallel_size(spec)
    dp = cpu_data_parallel_size(spec)
    env["VLLM_CPU_OMP_THREADS_BIND"] = cpu_omp_threads_bind(spec, tp, dp)
    if environ.get("VLLM_CPU_KVCACHE_SPACE"):
        env.setdefault("VLLM_CPU_KVCACHE_SPACE", environ["VLLM_CPU_KVCACHE_SPACE"])
    elif tuning and tuning.kv_cache_gib is not None:
        env.setdefault("VLLM_CPU_KVCACHE_SPACE", str(tuning.kv_cache_gib))
    return env


def cpu_gpu_memory_utilization() -> float:
    override = environ.get("VLLM_CPU_GPU_MEMORY_UTILIZATION")
    if override:
        return float(override)
    mem = virtual_memory()
    if mem.total <= 0:
        return 0.35
    util = (mem.available / mem.total) * 0.8
    return max(0.12, min(0.45, util))


def cpu_min_kv_gb(spec: ModelSpec, max_model_len: int) -> float:
    """Smallest KV cache worth serving with, per rank."""
    ctx_scale = max_model_len / 2048.0
    return 0.4 * ctx_scale * max(1.0, spec.params_b / 2.0)


def cpu_kv_memory_util(
    spec: ModelSpec,
    hp: HostProfile,
    max_model_len: int = 2048,
    ranks_per_node: int = 1,
) -> float:
    """Per-rank ``--gpu-memory-utilization`` for the CPU backend.

    The fraction is applied by every worker against the total RAM of its own
    NUMA node, so it is divided by the number of ranks sharing that node and
    capped so their sum stays inside the node's free RAM with sequential-alloc
    headroom (vLLM loads weights then reserves util×MemTotal − RSS for KV).
    """
    del hp  # reserved for future host-wide caps
    override = environ.get("VLLM_CPU_GPU_MEMORY_UTILIZATION")
    if override:
        return float(override)
    tp = cpu_tensor_parallel_size(spec)
    dp = cpu_data_parallel_size(spec)
    plan = cpu_layout_memory_plan(
        spec,
        tp=tp,
        dp=dp,
        max_model_len=max_model_len,
    )
    ceilings: list[float] = []
    for node in plan["nodes"]:
        total = float(node["total_gib"])
        ranks = max(1, int(node["ranks"]))
        if total > 0:
            multi = CPU_NODE_MULTI_RANK_UTIL_FACTOR if ranks > 1 else 1.0
            ceilings.append(
                min(
                    0.50,
                    float(node["available_gib"])
                    * CPU_NODE_MEMORY_BUDGET
                    * multi
                    / (ranks * total),
                )
            )
    return min(
        ceilings,
        default=min(0.50, CPU_NODE_MEMORY_BUDGET / max(1, ranks_per_node)),
    )


def cpu_serve_dtype(spec: ModelSpec | None = None) -> str:
    """Pick a CPU dtype the host ISA can actually execute.

    Graviton2 (Neoverse-N1) and older ARM lack ``bf16``; forcing bfloat16 there
    makes oneDNN fail and the API server never binds :8000.
    """
    del spec  # reserved for model-specific overrides later
    if override := environ.get("VLLM_CPU_DTYPE", "").strip():
        return override
    if cpu_has_bf16():
        return "bfloat16"
    return "float16"


def gpu_memory_utilization(spec: ModelSpec) -> float:
    override = environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    if override:
        return float(override)
    # Match vLLM 0.22's default. Higher/lower values belong in the experiment
    # sweep; preflight should not silently change the server's memory contract.
    return 0.92


def log_docker_cpu_hints() -> None:
    if not os.path.exists("/.dockerenv"):
        return
    try:
        shm = disk_usage("/dev/shm")
        if shm.total < 1024**3:
            logger.warning(
                "/dev/shm is only %.0f MiB; use docker --shm-size=4g",
                shm.total / (1024**2),
            )
    except OSError:
        pass


# Current vllm/vllm-openai CUDA wheels ship Ampere+ kernels (sm_80+).
VLLM_MIN_GPU_COMPUTE_CAP: tuple[int, int] = (8, 0)


@cache
def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_count": 0,
        "gpu_model": None,
        "vram_gb": 0.0,
        "total_vram_gb": 0.0,
        "vram_per_device_gb": [],
        "available_vram_per_device_gb": [],
        "compute_capability": None,
        "compute_capabilities": [],
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        count = torch.cuda.device_count()
        info["gpu_count"] = count
        if count:
            total = 0.0
            per_device: list[float] = []
            available_per_device: list[float] = []
            caps: list[tuple[int, int]] = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total += props.total_memory
                per_device.append(props.total_memory / (1024**3))
                try:
                    free, _device_total = torch.cuda.mem_get_info(i)
                    available_per_device.append(free / (1024**3))
                except Exception:
                    available_per_device.append(props.total_memory / (1024**3))
                caps.append((int(props.major), int(props.minor)))
                if i == 0:
                    info["gpu_model"] = props.name
                    info["compute_capability"] = caps[0]
            info["total_vram_gb"] = total / (1024**3)
            info["vram_gb"] = min(per_device)
            info["vram_per_device_gb"] = per_device
            info["available_vram_per_device_gb"] = available_per_device
            info["compute_capabilities"] = caps
    except Exception as e:
        logger.debug("gpu_info: %s", e)
    return info


def gpu_supports_current_image(info: dict[str, Any] | None = None) -> bool:
    """False when every visible GPU is below the image's minimum CUDA arch."""
    caps = list((info or gpu_info()).get("compute_capabilities") or [])
    if not caps:
        return True
    return min(caps) >= VLLM_MIN_GPU_COMPUTE_CAP


def check_gpu_compat(mode: str) -> None:
    if mode != "gpu":
        return
    info = gpu_info()
    if gpu_supports_current_image(info):
        return
    caps = info.get("compute_capabilities") or []
    logger.error(
        "GPU compute capability %s is below %s.%s required by benchmark-vllm-gpu "
        "(T4/T4G sm_75 and older are unsupported). Use an Ampere+ instance "
        "(g5/g6/p4/…) or the CPU image.",
        caps[0] if caps else "unknown",
        VLLM_MIN_GPU_COMPUTE_CAP[0],
        VLLM_MIN_GPU_COMPUTE_CAP[1],
    )
    sys_exit(1)


def _metadata_cache_path(spec: ModelSpec) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "--", spec.model_id)
    return Path(cli_args.models_dir) / ".vllm-footprints" / f"{safe}.json"


def _weight_files_from_repo(model_id: str) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """Return the exact deployed weight-format files and config from HF metadata.

    File sizes come from the Hub API (no weight download). Prefer safetensors;
    use an index's weight_map when present so repos carrying duplicate formats
    are never double-counted.
    """
    from huggingface_hub import HfApi, hf_hub_download

    token = environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    info = api.model_info(model_id, files_metadata=True)
    sizes = {
        sibling.rfilename: int(sibling.size or 0)
        for sibling in info.siblings or []
        if sibling.rfilename
    }
    config_path = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        cache_dir=cli_args.models_dir,
        token=token,
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    def indexed(index_name: str) -> list[tuple[str, int]]:
        if index_name not in sizes:
            return []
        path = hf_hub_download(
            repo_id=model_id,
            filename=index_name,
            cache_dir=cli_args.models_dir,
            token=token,
        )
        index = json.loads(Path(path).read_text(encoding="utf-8"))
        names = sorted(set((index.get("weight_map") or {}).values()))
        return [(name, sizes[name]) for name in names if sizes.get(name, 0) > 0]

    files = indexed("model.safetensors.index.json")
    if not files:
        files = [
            (name, size)
            for name, size in sizes.items()
            if size > 0
            and name.endswith(".safetensors")
            and not name.startswith(("adapter_", "optimizer"))
        ]
    if not files:
        files = indexed("pytorch_model.bin.index.json")
    if not files:
        files = [
            (name, size)
            for name, size in sizes.items()
            if size > 0 and re.search(r"(?:^|/)pytorch_model.*\.bin$", name)
        ]
    if not files:
        raise RuntimeError(f"No weight files with size metadata in {model_id}")
    return files, config


@cache
def model_metadata(spec: ModelSpec) -> ModelMetadata:
    """Resolve and persist model config + exact repository weight bytes."""
    cache_path = _metadata_cache_path(spec)
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return ModelMetadata(**cached)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    if __name__ != "__main__" and not environ.get("BENCHMARK_VLLM_TEST_HF_METADATA"):
        return ModelMetadata(
            weight_bytes=int(model_memory_gb_fallback(spec) * 1024**3),
            num_hidden_layers=None,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_attention_heads,
            hidden_size=None,
            torch_dtype=None,
            source="declared-test-fallback",
        )

    try:
        files, config = _weight_files_from_repo(spec.model_id)
        architecture = config.get("text_config") or config
        metadata = ModelMetadata(
            weight_bytes=sum(size for _name, size in files),
            num_hidden_layers=int(architecture["num_hidden_layers"])
            if architecture.get("num_hidden_layers") is not None
            else None,
            num_attention_heads=int(architecture["num_attention_heads"])
            if architecture.get("num_attention_heads") is not None
            else spec.num_attention_heads,
            num_key_value_heads=int(
                architecture.get("num_key_value_heads")
                or architecture.get("num_attention_heads")
            )
            if (
                architecture.get("num_key_value_heads")
                or architecture.get("num_attention_heads")
            )
            is not None
            else None,
            hidden_size=int(architecture["hidden_size"])
            if architecture.get("hidden_size") is not None
            else None,
            torch_dtype=str(
                architecture.get("torch_dtype") or config.get("torch_dtype")
            )
            if (architecture.get("torch_dtype") or config.get("torch_dtype"))
            is not None
            else None,
            source="huggingface:file-metadata+config",
            head_dim=int(architecture["head_dim"])
            if architecture.get("head_dim") is not None
            else None,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            key: value
            for key, value in metadata.as_dict().items()
            if key != "weight_gib"
        }
        cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        return metadata
    except Exception as exc:
        logger.warning("metadata lookup failed for %s: %s; using declared fallback", spec.model_id, exc)
        return ModelMetadata(
            weight_bytes=int(model_memory_gb_fallback(spec) * 1024**3),
            num_hidden_layers=None,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_attention_heads,
            hidden_size=None,
            torch_dtype=None,
            source="declared-fallback",
        )


def model_memory_gb_fallback(spec: ModelSpec) -> float:
    if spec.memory_gb is not None:
        return spec.memory_gb
    return spec.params_b * 2.0 * 1.2


def model_memory_gb(spec: ModelSpec) -> float:
    return model_metadata(spec).weight_bytes / (1024**3)


def dtype_bytes(dtype: str | None) -> int:
    value = (dtype or "").lower()
    if any(name in value for name in ("float32", "fp32")):
        return 4
    if any(name in value for name in ("float8", "fp8", "int8", "uint8")):
        return 1
    return 2


def kv_bytes_per_token(
    spec: ModelSpec,
    tp: int = 1,
    dtype: str | None = None,
) -> int | None:
    """KV bytes per cached token and worker from model config.

    Formula: K+V × layers × KV heads assigned to the TP rank × head_dim × dtype.
    """
    meta = model_metadata(spec)
    if not (
        meta.num_hidden_layers
        and meta.num_attention_heads
        and meta.num_key_value_heads
        and meta.hidden_size
    ):
        return None
    head_dim = meta.head_dim or math.ceil(meta.hidden_size / meta.num_attention_heads)
    kv_heads_per_rank = math.ceil(meta.num_key_value_heads / max(1, tp))
    return (
        2
        * meta.num_hidden_layers
        * kv_heads_per_rank
        * head_dim
        * dtype_bytes(dtype or meta.torch_dtype or cpu_serve_dtype(spec))
    )


def model_worker_memory_gb(
    spec: ModelSpec,
    *,
    tp: int,
    max_model_len: int,
    min_sequences: int = 1,
    dtype: str | None = None,
    pp: int = 1,
    block_size: int = 16,
) -> float:
    """Conservative per-worker startup footprint: TP weight shard + KV + runtime."""
    parallel_shards = max(1, tp * pp)
    weights = model_memory_gb(spec) * 1.10 / parallel_shards
    per_token = kv_bytes_per_token(spec, tp, dtype=dtype)
    if per_token is None:
        kv = cpu_min_kv_gb(spec, max_model_len)
    else:
        block_tokens = math.ceil(max_model_len / max(1, block_size)) * max(1, block_size)
        kv = (
            per_token
            * block_tokens
            * max(1, min_sequences)
            / max(1, pp)
            / (1024**3)
        )
    return weights + kv + 0.5


def available_memory_gb(mode: str) -> float:
    if mode == "gpu":
        total = gpu_info()["total_vram_gb"]
        if total > 0:
            return total * 0.9
    return virtual_memory().available / 1e9 * 0.85


def max_sequences_for_layout(
    spec: ModelSpec,
    *,
    mode: str,
    tp: int,
    dp: int,
    max_model_len: int,
    cpu_memory_util: float,
    gpu_memory_util_value: float,
) -> int:
    """Full-context sequences that fit in one worker's planned KV budget."""
    per_token = kv_bytes_per_token(spec, tp)
    if not per_token:
        return 1
    pp = (
        max(1, int(gpu_info().get("gpu_count") or 1))
        if mode == "gpu" and model_requires_gpu(spec)
        else 1
    )
    weights_and_runtime = model_memory_gb(spec) * 1.10 / max(1, tp * pp) + 0.5
    if mode == "cpu":
        counts = cpu_rank_counts_by_memory_node(spec, tp, dp)
        memories = host_memory_by_numa_gb()
        budgets = [
            memories[min(node, len(memories) - 1)][0] * cpu_memory_util
            for node in counts
        ]
    else:
        gpu = gpu_info()
        totals = list(gpu.get("vram_per_device_gb") or [])
        available = list(gpu.get("available_vram_per_device_gb") or totals)
        budgets = [
            min(total * gpu_memory_util_value, available[index] * 0.95)
            for index, total in enumerate(totals[: max(1, tp * dp)])
        ]
    if not budgets:
        return 1
    kv_gib = min(budgets) - weights_and_runtime
    block_size = 128 if mode == "cpu" else 16
    block_tokens = math.ceil(max_model_len / block_size) * block_size
    bytes_per_sequence = per_token * block_tokens / pp
    if kv_gib <= 0 or bytes_per_sequence <= 0:
        return 1
    return max(1, int(kv_gib * (1024**3) / bytes_per_sequence))


def gpu_layout_memory_plan(
    spec: ModelSpec,
    *,
    tp: int,
    dp: int,
    max_model_len: int,
    min_sequences: int = 1,
    pp: int = 1,
) -> dict[str, Any]:
    gpu = gpu_info()
    per_device = list(gpu.get("vram_per_device_gb") or [])
    available_per_device = list(gpu.get("available_vram_per_device_gb") or per_device)
    if not per_device and gpu.get("gpu_count"):
        per_device = [float(gpu.get("vram_gb") or 0)] * int(gpu["gpu_count"])
        available_per_device = list(per_device)
    workers = max(1, tp * dp)
    required = model_worker_memory_gb(
        spec,
        tp=tp,
        max_model_len=max_model_len,
        min_sequences=min_sequences,
        pp=pp,
        block_size=16,
    )
    workers = max(workers, pp)
    used = per_device[:workers]
    fits = (
        workers <= len(per_device)
        and bool(used)
        and all(
            required
            <= min(
                memory * gpu_memory_utilization(spec),
                available_per_device[index] * 0.95,
            )
            for index, memory in enumerate(used)
        )
    )
    return {
        "fits": fits,
        "tp": tp,
        "dp": dp,
        "pp": pp,
        "world_size": workers,
        "per_worker_gib": required,
        "vram_per_device_gib": used,
        "available_vram_per_device_gib": available_per_device[:workers],
        "gpu_memory_utilization": gpu_memory_utilization(spec),
        "model": model_metadata(spec).as_dict(),
    }


def model_host_plan(spec: ModelSpec, mode: str) -> dict[str, Any]:
    """Whether at least one valid layout can host the model for chat."""
    supported = model_supported_on_mode(spec, mode)
    if not supported:
        return {
            "model": spec.short_name,
            "model_id": spec.model_id,
            "mode": mode,
            "runnable": False,
            "reason": "unsupported serve configuration for mode",
            "metadata": None,
        }
    metadata = model_metadata(spec)
    if metadata.source in {"declared-fallback"} and not environ.get(
        "BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA"
    ):
        return {
            "model": spec.short_name,
            "model_id": spec.model_id,
            "mode": mode,
            "runnable": False,
            "reason": "authoritative Hugging Face weight/config metadata unavailable",
            "metadata": metadata.as_dict(),
        }
    max_len = min(w.max_model_len for w in workloads_for_mode(mode))
    candidates: list[dict[str, Any]] = []
    if mode == "cpu":
        heads = metadata.num_attention_heads or spec.num_attention_heads
        for tp in range(1, max(1, host_numa_count()) + 1):
            if heads and heads % tp:
                continue
            for dp in range(
                1,
                min(MAX_CPU_DP, max(1, host_profile().vcpus // tp)) + 1,
            ):
                candidates.append(
                    cpu_layout_memory_plan(
                        spec,
                        tp=tp,
                        dp=dp,
                        max_model_len=max_len,
                    )
                )
    else:
        gpus = max(1, int(gpu_info().get("gpu_count") or 1))
        heads = metadata.num_attention_heads or spec.num_attention_heads
        if model_requires_gpu(spec) and gpus > 1:
            candidates.append(
                gpu_layout_memory_plan(
                    spec,
                    tp=1,
                    dp=1,
                    pp=gpus,
                    max_model_len=max_len,
                )
            )
        else:
            for tp in range(1, gpus + 1):
                if heads and heads % tp:
                    continue
                for dp in range(1, gpus // tp + 1):
                    candidates.append(
                        gpu_layout_memory_plan(
                            spec,
                            tp=tp,
                            dp=dp,
                            max_model_len=max_len,
                        )
                    )
    runnable = any(candidate["fits"] for candidate in candidates)
    feasible = [
        {
            "tp": candidate["tp"],
            "dp": candidate["dp"],
            "pp": candidate.get("pp", 1),
        }
        for candidate in candidates
        if candidate["fits"]
    ]
    return {
        "model": spec.short_name,
        "model_id": spec.model_id,
        "mode": mode,
        "runnable": runnable,
        "reason": None if runnable else "weight+minimum-KV footprint exceeds host resources",
        "metadata": metadata.as_dict(),
        "feasible_layouts": feasible,
        "candidate_layouts": candidates,
    }


def model_fits(spec: ModelSpec, mode: str) -> bool:
    plan = model_host_plan(spec, mode)
    need = model_memory_gb(spec)
    have = available_memory_gb(mode)
    logger.info(
        "memory check %s: weights=%.1f GB aggregate_available=%.1f GB runnable=%s source=%s",
        spec.short_name,
        need,
        have,
        plan["runnable"],
        plan["metadata"]["source"],
    )
    return bool(plan["runnable"])


def workload_kv_fits(
    spec: ModelSpec,
    mode: str,
    max_model_len: int,
    kv_util: float | None = None,
) -> bool:
    """Estimate vLLM CPU KV cache headroom after loading weights (see cpu_worker.py)."""
    if mode == "gpu":
        pp = (
            max(1, int(gpu_info().get("gpu_count") or 1))
            if model_requires_gpu(spec)
            else 1
        )
        plan = gpu_layout_memory_plan(
            spec,
            tp=tensor_parallel_size(mode, spec),
            dp=gpu_data_parallel_size(spec),
            pp=pp,
            max_model_len=max_model_len,
        )
        if not plan["fits"]:
            logger.info(
                "GPU resource plan rejected %s max_model_len=%s plan=%s",
                spec.short_name,
                max_model_len,
                plan,
            )
        return bool(plan["fits"])
    tp = cpu_tensor_parallel_size(spec)
    dp = cpu_data_parallel_size(spec)
    plan = cpu_layout_memory_plan(
        spec,
        tp=tp,
        dp=dp,
        max_model_len=max_model_len,
    )
    if plan["fits"]:
        return True
    logger.info(
        "resource plan rejected %s max_model_len=%s tp=%s dp=%s bind=%s nodes=%s",
        spec.short_name,
        max_model_len,
        tp,
        dp,
        plan["binding"],
        plan["nodes"],
    )
    return False


def model_requires_gpu(spec: ModelSpec) -> bool:
    """Serve flags that vLLM CPU backend cannot use (e.g. bitsandbytes quant)."""
    return any("bitsandbytes" in a for a in spec.serve_extra_args)


def model_supported_on_mode(spec: ModelSpec, mode: str) -> bool:
    if spec.cpu_only and mode != "cpu":
        return False
    if spec.gpu_only and mode != "gpu":
        return False
    if mode == "cpu" and model_requires_gpu(spec):
        return False
    return True


def probe_model_spec() -> ModelSpec:
    override = environ.get("VLLM_PROBE_MODEL", "").strip()
    if override:
        return ModelSpec("probe", override, 0.135)
    return DEFAULT_MODELS[0]


def models_to_run(mode: str) -> list[ModelSpec]:
    if cli_args.models:
        by_id = {s.model_id: s for s in DEFAULT_MODELS}
        by_short = {s.short_name: s for s in DEFAULT_MODELS}
        out: list[ModelSpec] = []
        for m in cli_args.models:
            if m in by_id:
                out.append(by_id[m])
            elif m in by_short:
                out.append(by_short[m])
            else:
                out.append(
                    ModelSpec(os.path.basename(m.rstrip("/")), m, max(0.135, len(m) / 20))
                )
        return out
    return [spec for spec in DEFAULT_MODELS if model_supported_on_mode(spec, mode)]


def workloads_for_mode(mode: str) -> list[WorkloadSpec]:
    out = [w for w in WORKLOADS if not w.gpu_only or mode == "gpu"]
    raw = environ.get("BENCHMARK_VLLM_WORKLOADS", "").strip()
    if not raw:
        return out
    names = {n.strip() for n in raw.split(",") if n.strip()}
    filtered = [w for w in out if w.name in names]
    if not filtered:
        logger.warning("BENCHMARK_VLLM_WORKLOADS=%r matched nothing; using %s", raw, [w.name for w in out])
        return out
    return filtered


def guidellm_sweep_size(mode: str) -> str:
    """Sweep steps (sync + throughput + constant interpolations). Minimum useful size is 2."""
    if mode == "gpu":
        return environ.get("GUIDELLM_GPU_SWEEP_SIZE") or environ.get("GUIDELLM_SWEEP_SIZE", "3")
    return environ.get("GUIDELLM_CPU_SWEEP_SIZE") or environ.get("GUIDELLM_SWEEP_SIZE", "3")


def guidellm_sweep_profile(mode: str, tuning: BenchmarkTuning) -> str:
    if not tuning.autoconfig:
        return f"sweep,{guidellm_sweep_size(mode)}"
    return f"sweep,{tuning.sweep_size}"


def guidellm_throughput_rate(mode: str) -> str:
    """max_concurrency for standalone throughput profile (legacy plan only)."""
    default = "8" if mode == "cpu" else "16"
    return environ.get("GUIDELLM_THROUGHPUT_RATE", default)


def _guidellm_profile_override(mode: str) -> str:
    return (
        environ.get("GUIDELLM_PROFILES", "").strip().lower()
        or (environ.get("GUIDELLM_CPU_PROFILES", "").strip().lower() if mode == "cpu" else "")
    )


def guidellm_plan(mode: str, tuning: BenchmarkTuning) -> list[tuple[str, str | None]]:
    """(profile kind, rate-or-sweep-size) runs per model/workload.

    GuideLLM 0.7+ puts sweep size / concurrency on ``--profile kind=...,...``.
    """
    override = _guidellm_profile_override(mode)
    if override in ("legacy", "sync-throughput", "sync"):
        return [("synchronous", None), ("throughput", guidellm_throughput_rate(mode))]
    if tuning.autoconfig:
        return [("sweep", str(tuning.sweep_size))]
    return [("sweep", guidellm_sweep_size(mode))]


def guidellm_profile_spec(
    profile: str,
    rate: str | None,
    tuning: BenchmarkTuning,
) -> str:
    """Build a GuideLLM 0.7+ ``--profile kind=...,key=value`` string."""
    parts = [f"kind={profile}"]
    if profile == "sweep":
        size = rate if rate is not None else str(tuning.sweep_size)
        parts.append(f"sweep_size={size}")
        if tuning.autoconfig:
            parts.append(f"max_concurrency={tuning.max_concurrency}")
            if tuning.rampup_duration > 0:
                parts.append(f"rampup_duration={tuning.rampup_duration}")
    elif profile == "throughput":
        conc = rate if rate is not None else str(tuning.max_concurrency)
        parts.append(f"max_concurrency={conc}")
        if tuning.autoconfig and tuning.rampup_duration > 0:
            parts.append(f"rampup_duration={tuning.rampup_duration}")
    elif rate is not None and profile in ("concurrent", "constant", "async", "poisson"):
        key = "streams" if profile == "concurrent" else "rate"
        parts.append(f"{key}={rate}")
    if tuning.warmup:
        parts.append(f"warmup={tuning.warmup}")
    return ",".join(parts)


def guidellm_max_seconds(mode: str, spec: ModelSpec, tuning: BenchmarkTuning) -> int:
    if tuning.autoconfig:
        return tuning.max_seconds_per_strategy
    if mode == "gpu":
        base = 40 + int(spec.params_b * 8)
    else:
        base = 45 + int(spec.params_b * 12)
    return base * max(1, cli_args.benchmark_timeout_scale)


def guidellm_subprocess_timeout(tuning: BenchmarkTuning) -> int:
    """Wall timeout for the full GuideLLM sweep subprocess."""
    scale = max(1, cli_args.benchmark_timeout_scale)
    warmup_sec = int(tuning.warmup) if tuning.warmup.isdigit() else 15
    rampup_sec = int(tuning.rampup_duration) if tuning.rampup_duration > 0 else 0
    # Stages run up to max_seconds each; slow models may drain in-flight requests after.
    stage_wall = tuning.max_seconds_per_strategy * tuning.sweep_size * scale
    drain_pad = tuning.max_seconds_per_strategy * scale
    return int(
        max(
            tuning.per_run_budget_sec * scale * 1.5,
            stage_wall + warmup_sec + rampup_sec + drain_pad + 180,
        )
    )


def guidellm_max_requests(mode: str, tuning: BenchmarkTuning) -> int | None:
    if tuning.autoconfig:
        return tuning.max_requests
    if mode == "gpu":
        return int(environ.get("GUIDELLM_MAX_REQUESTS", "120"))
    return int(environ.get("GUIDELLM_MAX_REQUESTS_CPU", "25"))


def emit_jsonl(record: dict[str, Any]) -> None:
    stdout.write(json.dumps(record) + "\n")
    stdout.flush()


def wait_for_health(timeout_sec: float, server: Optional[Popen[Any]] = None) -> bool:
    deadline = monotonic() + timeout_sec
    while monotonic() < deadline:
        if server is not None and server.poll() is not None:
            logger.warning("vLLM server exited with code %s", server.returncode)
            return False
        try:
            with urlopen(HEALTH_URL, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        sleep(2)
    return False


def _server_stderr_tail(max_chars: int = 4000, max_lines: int = 8) -> list[str]:
    path = _SERVER_STDERR_PATH
    if path is None or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-max_lines:]


def _server_memory_observed() -> dict[str, Any]:
    """Extract vLLM's profiled memory results; runtime init is authoritative."""
    path = _SERVER_STDERR_PATH
    if path is None or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    observed: dict[str, Any] = {}
    gib_values = [
        float(value)
        for value in re.findall(
            r"(?:Available KV cache memory|KV cache memory)[^:\n]*:\s*"
            r"([0-9.]+)\s*GiB",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if gib_values:
        observed["kv_cache_gib_per_worker"] = gib_values
    token_values = [
        int(value.replace(",", ""))
        for value in re.findall(
            r"(?:GPU|CPU) KV cache size:\s*([0-9,]+)\s*tokens",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if token_values:
        observed["kv_cache_tokens_per_worker"] = token_values
    byte_values = [
        int(value.replace(",", ""))
        for value in re.findall(
            r"--kv-cache-memory(?:-bytes)?[=\s]+([0-9,]+)",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if byte_values:
        observed["suggested_kv_cache_memory_bytes"] = byte_values
    return observed


_SERVER_CAUSE_RE = re.compile(
    r"(ValueError|RuntimeError|MemoryError|AssertionError|ImportError|OSError|"
    r"NotImplementedError|OutOfMemoryError|CUDA error|Available memory|"
    r"Illegal instruction|Killed)"
)


def _server_stderr_causes(max_chars: int = 400_000, max_lines: int = 6) -> list[str]:
    """First error lines in the server log, not just its last words.

    vLLM reports worker failures through a wrapper ("Engine core initialization
    failed. See root cause above."), so the tail alone hides why a rank died.
    """
    path = _SERVER_STDERR_PATH
    if path is None or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return []
    causes: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "resource_tracker" in line:
            continue
        if _SERVER_CAUSE_RE.search(line):
            causes.append(line)
        if len(causes) >= max_lines:
            break
    return causes


def _classify_server_failure(tail: list[str]) -> str | None:
    blob = "\n".join(tail).lower()
    if "kv cache" in blob and "less than requested" in blob:
        return "KV cache OOM"
    if "less than desired cpu memory utilization" in blob:
        return "CPU memory reservation too high (--gpu-memory-utilization)"
    if "less than desired gpu memory utilization" in blob:
        return "memory reservation too high"
    if "available memory" in blob and "on startup" in blob:
        return "insufficient RAM at startup"
    return None


def log_server_start_failure(
    mode: str,
    spec: ModelSpec | None = None,
    workload: WorkloadSpec | None = None,
    server: Optional[Popen[Any]] = None,
) -> None:
    causes = _server_stderr_causes()
    tail = _server_stderr_tail()
    kind = _classify_server_failure(causes + tail)
    label = spec.short_name if spec else "model"
    ctx = f" workload={workload.name}" if workload else ""
    if kind:
        summary = f"vLLM {mode} server failed for {label}{ctx}: {kind}."
    else:
        summary = f"vLLM {mode} server failed for {label}{ctx}."
    hints = [summary]
    if server is not None and server.poll() is not None:
        hints.append(f"exit_code={server.returncode}")
    if mode == "cpu":
        hints.append("Try --privileged --shm-size=4g")
        if host_arch() == "amd64" and not cpu_has_avx512():
            hints.append("BENCHMARK_VLLM_ALLOW_AVX2_ONLY on AVX2-only amd64")
    for line in dict.fromkeys(causes + tail):
        hints.append(f"stderr: {line[:240]}")
    logger.error(" | ".join(hints))


def gpu_data_parallel_size(spec: ModelSpec) -> int:
    """GPU DP replicas. Default 1 (current fleet); override with BENCHMARK_VLLM_GPU_DP.

    vLLM applies ``--max-num-seqs`` per DP rank. Independent replicas are the
    way to use leftover GPUs when attention heads are not divisible by GPU count
    (e.g. SmolLM2's 9 heads on 2 GPUs).
    """
    override = _env_int("BENCHMARK_VLLM_GPU_DP", "VLLM_GPU_DATA_PARALLEL_SIZE")
    if override is not None:
        return max(1, override)
    return 1


def tensor_parallel_size(mode: str, spec: ModelSpec) -> int:
    """Largest TP for the active backend (GPU count or CPU NUMA nodes)."""
    if mode == "cpu":
        return cpu_tensor_parallel_size(spec)
    override = _env_int("BENCHMARK_VLLM_GPU_TP", "VLLM_GPU_TENSOR_PARALLEL_SIZE")
    if override is not None:
        return max(1, override)
    gpus = max(1, int(gpu_info()["gpu_count"] or 1))
    if gpus <= 1:
        return 1
    heads = model_metadata(spec).num_attention_heads or spec.num_attention_heads
    if not heads:
        return 1
    for tp in range(min(gpus, heads), 0, -1):
        if heads % tp == 0:
            return tp
    return 1


def start_server(model_id: str, mode: str, max_model_len: int, spec: ModelSpec) -> Popen[Any]:
    global _SERVER_STDERR_PATH
    err_log = tempfile.NamedTemporaryFile(
        mode="w+",
        prefix="vllm-serve-",
        suffix=".log",
        delete=False,
    )
    _SERVER_STDERR_PATH = Path(err_log.name)
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(max_model_len),
        *spec.serve_extra_args,
    ]
    gpus = max(1, int(gpu_info()["gpu_count"] or 1))
    tuning = current_tuning(mode, spec, max_model_len=max_model_len)
    if mode == "gpu":
        if any("bitsandbytes" in a for a in spec.serve_extra_args) and gpus > 1:
            cmd.extend(["--pipeline-parallel-size", str(gpus)])
        else:
            tp = tensor_parallel_size(mode, spec)
            cmd.extend(
                [
                    "--tensor-parallel-size",
                    str(tp),
                    "--gpu-memory-utilization",
                    f"{tuning.gpu_memory_util:.2f}",
                ]
            )
            dp = gpu_data_parallel_size(spec)
            if dp > 1:
                cmd.extend(["--data-parallel-size", str(dp)])
            if seqs_env := _env_int("BENCHMARK_VLLM_MAX_NUM_SEQS"):
                cmd.extend(["--max-num-seqs", str(max(1, seqs_env))])
            if batched_env := _env_int("BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS"):
                cmd.extend(["--max-num-batched-tokens", str(max(1, batched_env))])
    else:
        mem_util = tuning.kv_memory_util if tuning.autoconfig else cpu_gpu_memory_utilization()
        cmd.extend(
            [
                "--dtype",
                tuning.dtype if tuning.autoconfig else cpu_serve_dtype(spec),
                "--gpu-memory-utilization",
                f"{mem_util:.2f}",
            ]
        )
        tp = tensor_parallel_size(mode, spec)
        if tp > 1:
            cmd.extend(["--tensor-parallel-size", str(tp)])
        dp = cpu_data_parallel_size(spec)
        if dp > 1:
            cmd.extend(["--data-parallel-size", str(dp)])
        if tuning.autoconfig:
            cmd.extend(
                [
                    "--max-num-seqs",
                    str(tuning.max_num_seqs),
                    "--max-num-batched-tokens",
                    str(tuning.max_num_batched_tokens),
                ]
            )

    env = os.environ.copy()
    env.setdefault("HF_HOME", cli_args.models_dir)
    env.setdefault("HUGGINGFACE_HUB_CACHE", cli_args.models_dir)
    if mode == "cpu":
        env = cpu_server_env(env, tuning, spec=spec)
    os.makedirs(cli_args.models_dir, exist_ok=True)
    logger.info("Starting server: %s", " ".join(cmd))
    if mode == "cpu":
        logger.info(
            "cpu parallel: numa=%s tp=%s dp=%s ranks/node=%s mem_util=%s omp_bind=%s",
            host_numa_count(),
            tensor_parallel_size(mode, spec),
            cpu_data_parallel_size(spec),
            cpu_ranks_per_memory_node(spec),
            f"{tuning.kv_memory_util:.2f}",
            env.get("VLLM_CPU_OMP_THREADS_BIND", ""),
        )
    else:
        logger.info(
            "gpu parallel: gpus=%s tp=%s dp=%s mem_util=%s",
            gpus,
            tensor_parallel_size(mode, spec),
            gpu_data_parallel_size(spec),
            f"{tuning.gpu_memory_util:.2f}",
        )
    return Popen(
        cmd,
        stdout=DEVNULL,
        stderr=err_log,
        env=env,
        start_new_session=True,
    )


def stop_server(proc: Optional[Popen[Any]]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
    if proc.stderr is not None:
        try:
            proc.stderr.flush()
            proc.stderr.close()
        except OSError:
            pass


def run_guidellm(
    spec: ModelSpec,
    workload: WorkloadSpec,
    profile: str,
    rate: str | None,
    mode: str,
    out_dir: Path,
    tuning: BenchmarkTuning,
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_seconds = guidellm_max_seconds(mode, spec, tuning)
    report_path = out_dir / "benchmarks.json"
    cmd = [
        "guidellm",
        "run",
        "--backend",
        (
            f"kind=openai_http,target={TARGET_URL},model={spec.model_id},"
            "request_format=/v1/completions"
        ),
        "--data",
        (
            "kind=synthetic_text,"
            f"prompt_tokens={workload.prompt_tokens},"
            f"output_tokens={workload.output_tokens}"
        ),
        "--profile",
        guidellm_profile_spec(profile, rate, tuning),
        "--seed",
        "kind=static,value=42",
        "--constraint",
        f"kind=max_duration,seconds={max_seconds}",
        "--output",
        f"kind=json,path={report_path}",
        "--metrics",
        "kind=generative,sample_size=10",
        "--disable-console",
    ]
    if (max_req := guidellm_max_requests(mode, tuning)) is not None:
        cmd.extend(["--constraint", f"kind=max_requests,count={max_req}"])

    logger.info("GuideLLM: %s", " ".join(cmd))
    subprocess_timeout = guidellm_subprocess_timeout(tuning)
    try:
        result = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=subprocess_timeout,
            check=False,
            env=guidellm_env(),
            cwd=str(out_dir),
        )
    except TimeoutExpired:
        logger.warning(
            "GuideLLM timed out profile=%s workload=%s after %ss",
            profile,
            workload.name,
            subprocess_timeout,
        )
        report = find_guidellm_report(out_dir)
        if report is not None:
            logger.warning("Using partial GuideLLM report after timeout: %s", report)
            return report
        return None

    if result.returncode != 0:
        logger.warning(
            "GuideLLM failed profile=%s workload=%s: %s",
            profile,
            workload.name,
            (result.stderr or result.stdout)[-3000:],
        )
        report = find_guidellm_report(out_dir)
        if report is not None:
            logger.warning("Using partial GuideLLM report after failure: %s", report)
            return report
        return None

    report = find_guidellm_report(out_dir)
    if report is None:
        logger.warning("No benchmarks.json under %s", out_dir)
    return report


def find_guidellm_report(out_dir: Path) -> Path | None:
    report = out_dir / "benchmarks.json"
    if report.is_file():
        return report
    candidates = list(out_dir.glob("**/benchmarks.json"))
    return candidates[0] if candidates else None


def _dist_block(metrics: dict[str, Any], key: str) -> dict[str, Any] | None:
    raw = metrics.get(key)
    if not isinstance(raw, dict):
        return None
    for branch in ("successful", "total"):
        block = raw.get(branch)
        if isinstance(block, dict):
            return block
    return raw if "mean" in raw else None


def _strategy_label(bench: dict[str, Any]) -> str:
    config = bench.get("config") or {}
    strategy = config.get("strategy") or {}
    if isinstance(strategy, dict):
        return str(strategy.get("type_") or strategy.get("type") or "unknown")
    return "unknown"


def _target_rate(bench: dict[str, Any]) -> float | str | None:
    state = bench.get("scheduler_state") or {}
    if isinstance(state, dict):
        for key in ("target_rate", "rate", "requests_per_second"):
            if key in state and state[key] is not None:
                return state[key]
    config = bench.get("config") or {}
    constraints = config.get("constraints") or {}
    if isinstance(constraints, dict):
        rate = constraints.get("rate")
        if rate is not None:
            return rate
    return None


def _concurrency_mean(metrics: dict[str, Any]) -> float | None:
    block = _dist_block(metrics, "request_concurrency")
    if block and block.get("mean") is not None:
        return float(block["mean"])
    return None


def report_to_jsonl(
    report_path: Path,
    spec: ModelSpec,
    workload: WorkloadSpec,
    profile: str,
    mode: str,
    tuning: BenchmarkTuning,
) -> int:
    with open(report_path, encoding="utf-8") as fp:
        report = json.load(fp)

    gi = gpu_info()
    metadata = model_metadata(spec)
    base: dict[str, Any] = {
        "benchmark": "vllm_serving",
        "model": spec.short_name,
        "model_id": spec.model_id,
        "workload": workload.name,
        "prompt_tokens": workload.prompt_tokens,
        "output_tokens": workload.output_tokens,
        "profile": profile,
        "mode": mode,
        "arch": host_arch(),
        "max_model_len": tuning.max_model_len,
        "tuning_version": tuning.tuning_version,
        "tuning": tuning.as_dict(),
        "model_weight_gib": round(metadata.weight_bytes / (1024**3), 4),
        "model_metadata_source": metadata.source,
        "kv_bytes_per_token_per_tp_rank": kv_bytes_per_token(
            spec,
            tensor_parallel_size(mode, spec),
            dtype=tuning.dtype if mode == "cpu" else None,
        ),
        "runtime_memory": _server_memory_observed(),
        "avx512": cpu_has_avx512() if host_arch() == "amd64" else None,
        "avx2_only_image": environ.get("BENCHMARK_VLLM_ALLOW_AVX2_ONLY", "").lower()
        in ("1", "true", "yes"),
        "vllm_version": read_vllm_version(),
        "guidellm_version": read_guidellm_version(),
        "tensor_parallel": tensor_parallel_size(mode, spec),
        "pipeline_parallel": (
            max(1, int(gi["gpu_count"] or 1))
            if mode == "gpu" and model_requires_gpu(spec)
            else 1
        ),
        "data_parallel": (
            cpu_data_parallel_size(spec) if mode == "cpu" else gpu_data_parallel_size(spec)
        ),
        "omp_threads_bind": (
            cpu_omp_threads_bind(spec) if mode == "cpu" else None
        ),
        "gpu_count": gi["gpu_count"],
        "gpu_model": gi["gpu_model"],
        "total_vram_gb": round(float(gi["total_vram_gb"]), 2),
    }

    count = 0
    for bench in report.get("benchmarks") or []:
        if not isinstance(bench, dict):
            continue
        metrics = bench.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue

        row_base = {
            **base,
            "strategy": _strategy_label(bench),
            "target_rate": _target_rate(bench),
            "concurrency": _concurrency_mean(metrics),
        }

        for short, key, unit, scale in LATENCY_METRICS:
            block = _dist_block(metrics, key)
            if not block:
                continue
            percentiles = block.get("percentiles") or {}
            for pct in PERCENTILES:
                val = percentiles.get(pct)
                if val is None:
                    continue
                emit_jsonl(
                    {
                        **row_base,
                        "measurement": short,
                        "percentile": pct,
                        "score": float(val) * scale,
                        "unit": unit,
                    }
                )
                count += 1
            if block.get("mean") is not None:
                emit_jsonl(
                    {
                        **row_base,
                        "measurement": short,
                        "percentile": "mean",
                        "score": float(block["mean"]) * scale,
                        "unit": unit,
                    }
                )
                count += 1

        for short, key, unit in THROUGHPUT_METRICS:
            block = _dist_block(metrics, key)
            if not block or block.get("mean") is None:
                continue
            emit_jsonl(
                {
                    **row_base,
                    "measurement": short,
                    "percentile": None,
                    "score": float(block["mean"]),
                    "unit": unit,
                }
            )
            count += 1

    return count


def _run_guidellm_sweeps(
    spec: ModelSpec,
    workload: WorkloadSpec,
    mode: str,
    tuning: BenchmarkTuning,
    start_time: float,
) -> float:
    global _EMITTED_ROWS
    peak_output_tps = 0.0
    for profile, rate in guidellm_plan(mode, tuning):
        if monotonic() - start_time > OVERALL_TIMEOUT_SEC:
            break
        with tempfile.TemporaryDirectory(prefix="guidellm-") as tmp:
            report = run_guidellm(
                spec,
                workload,
                profile,
                rate,
                mode,
                Path(tmp),
                tuning,
            )
            if not report:
                continue
            n = report_to_jsonl(report, spec, workload, profile, mode, tuning)
            _EMITTED_ROWS += n
            logger.info(
                "GuideLLM emitted %s rows model=%s workload=%s profile=%s",
                n,
                spec.short_name,
                workload.name,
                profile,
            )
            with open(report, encoding="utf-8") as fp:
                raw = json.load(fp)
            for bench in raw.get("benchmarks") or []:
                metrics = bench.get("metrics") or {}
                block = _dist_block(metrics, "output_tokens_per_second")
                if block and block.get("mean") is not None:
                    peak_output_tps = max(peak_output_tps, float(block["mean"]))
    return peak_output_tps


def run_model(spec: ModelSpec, mode: str, start_time: float) -> bool:
    if monotonic() - start_time > OVERALL_TIMEOUT_SEC:
        logger.warning("Overall timeout reached")
        return False

    workloads = workloads_for_mode(mode)
    peak_output_tps = 0.0

    if per_workload_server_enabled():
        for workload in workloads:
            if monotonic() - start_time > OVERALL_TIMEOUT_SEC:
                return False
            tuning = init_benchmark_tuning(mode, spec, max_model_len=workload.max_model_len)
            if not workload_kv_fits(spec, mode, workload.max_model_len, tuning.kv_memory_util):
                logger.info(
                    "Skipping %s workload=%s — insufficient KV cache headroom",
                    spec.short_name,
                    workload.name,
                )
                continue
            server = None
            try:
                server = start_server(
                    spec.model_id, mode, workload.max_model_len, spec
                )
                health_timeout = (
                    SERVER_START_TIMEOUT_CPU_SEC
                    if mode == "cpu"
                    else SERVER_START_TIMEOUT_GPU_SEC
                )
                if not wait_for_health(health_timeout, server):
                    logger.warning(
                        "Server health check failed for %s workload=%s",
                        spec.model_id,
                        workload.name,
                    )
                    log_server_start_failure(mode, spec, workload, server)
                    continue
                peak_output_tps = max(
                    peak_output_tps,
                    _run_guidellm_sweeps(spec, workload, mode, tuning, start_time),
                )
            finally:
                stop_server(server)
    else:
        max_len = max(w.max_model_len for w in workloads)
        tuning = init_benchmark_tuning(mode, spec, max_model_len=max_len)
        server = None
        try:
            server = start_server(spec.model_id, mode, max_len, spec)
            health_timeout = (
                SERVER_START_TIMEOUT_CPU_SEC if mode == "cpu" else SERVER_START_TIMEOUT_GPU_SEC
            )
            if not wait_for_health(health_timeout, server):
                logger.warning("Server health check failed for %s", spec.model_id)
                log_server_start_failure(mode, spec, server=server)
                return True
            for workload in workloads:
                if monotonic() - start_time > OVERALL_TIMEOUT_SEC:
                    return False
                peak_output_tps = max(
                    peak_output_tps,
                    _run_guidellm_sweeps(spec, workload, mode, tuning, start_time),
                )
        finally:
            stop_server(server)

    if peak_output_tps > 0 and peak_output_tps < MIN_OUTPUT_TOKENS_PER_SEC:
        logger.warning(
            "Peak output %.2f tok/s below threshold; stopping ladder",
            peak_output_tps,
        )
        return False
    return True


def probe_health_timeout_sec(mode: str) -> float:
    if mode == "cpu":
        return float(
            environ.get("VLLM_PROBE_HEALTH_TIMEOUT_CPU_SEC")
            or SERVER_START_TIMEOUT_PROBE_CPU_SEC
        )
    return float(
        environ.get("VLLM_PROBE_HEALTH_TIMEOUT_GPU_SEC")
        or SERVER_START_TIMEOUT_PROBE_GPU_SEC
    )


def run_probe_only(mode: str) -> None:
    log_cpu_details()
    check_cpu_isa_compat(mode)
    check_gpu_compat(mode)
    if mode == "cpu":
        log_docker_cpu_hints()
    spec = probe_model_spec()
    if not model_fits(spec, mode):
        logger.error("Probe model %s does not fit in available memory", spec.model_id)
        sys_exit(1)
    init_benchmark_tuning(mode, spec, max_model_len=2048)
    server = None
    try:
        server = start_server(spec.model_id, mode, 2048, spec)
        if wait_for_health(probe_health_timeout_sec(mode), server):
            logger.info("probe_ok model=%s mode=%s", spec.model_id, mode)
            sys_exit(0)
        log_server_start_failure(mode, spec, server=server)
        sys_exit(1)
    finally:
        stop_server(server)


def run_plan_only(mode: str) -> None:
    plans = [model_host_plan(spec, mode) for spec in models_to_run(mode)]
    print(
        json.dumps(
            {
                "mode": mode,
                "host": {
                    "vcpus": host_profile().vcpus,
                    "ram_total_gb": host_profile().ram_total_gb,
                    "ram_available_gb": host_profile().ram_avail_gb,
                    "numa": host_numa_count(),
                    "memory_by_numa_gb": host_memory_by_numa_gb(),
                    "gpu": gpu_info(),
                },
                "models": plans,
            }
        )
    )


def print_versions() -> str:
    # Pinned files match the image build; avoid vllm CLI stderr noise in meta.json.
    version = f"vllm={read_vllm_version()} guidellm={read_guidellm_version()}"
    print(version)
    return version


def main() -> None:
    if cli_args.version:
        print_versions()
        sys_exit(0)

    log_cpu_details()
    mode = detect_mode()
    if cli_args.plan_only:
        run_plan_only(mode)
        return
    if not shutil.which("guidellm"):
        logger.error("guidellm CLI not found in PATH")
        sys_exit(1)
    if cli_args.probe_only:
        if mode == "cpu":
            log_docker_cpu_hints()
        run_probe_only(mode)
        return

    check_cpu_isa_compat(mode)
    check_gpu_compat(mode)
    if mode == "cpu":
        log_docker_cpu_hints()
    logger.info(
        "mode=%s arch=%s vllm=%s guidellm=%s",
        mode,
        host_arch(),
        read_vllm_version(),
        read_guidellm_version(),
    )

    free_disk = disk_usage(cli_args.models_dir).free / 1e9
    if free_disk < 1.0:
        logger.error("Less than 1 GiB free in models_dir")
        sys_exit(1)

    budget = compute_budget(mode) if autoconfig_enabled() else None
    if budget:
        logger.info(
            "benchmark budget overall=%ss reserve=%ss runs=%s per_run=%ss",
            budget.overall_timeout_sec,
            budget.reserve_sec,
            budget.total_runs,
            budget.per_run_sec,
        )

    start = monotonic()
    stop_ladder = False
    attempted = 0
    for spec in models_to_run(mode):
        if stop_ladder:
            break
        if not model_supported_on_mode(spec, mode):
            if mode == "cpu" and model_requires_gpu(spec):
                logger.info(
                    "Skipping %s — GPU-only serve config on CPU (e.g. bitsandbytes)",
                    spec.short_name,
                )
            continue
        if not model_fits(spec, mode):
            logger.info("Skipping %s — insufficient memory", spec.short_name)
            continue
        attempted += 1
        if not run_model(spec, mode, start):
            stop_ladder = True

    if attempted == 0:
        logger.error("No models runnable on this host (memory/mode); not recording empty success")
        sys_exit(1)
    if _EMITTED_ROWS == 0:
        logger.error(
            "Attempted %s model(s) but emitted no measurements; see server stderr above",
            attempted,
        )
        sys_exit(1)
    sys_exit(0)


if __name__ == "__main__":
    main()
