#!/usr/bin/env python3
"""vLLM serving benchmark: start vllm serve, run GuideLLM, emit JSONL metrics."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import signal
import tempfile
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
TUNING_VERSION = 2
BUDGET_RESERVE_STARTUP_SEC = 600
BUDGET_MIN_PER_RUN_SEC = 45
BUDGET_MAX_PER_RUN_SEC = 240
BUDGET_MODEL_START_CPU_SEC = 120
BUDGET_MODEL_START_GPU_SEC = 60

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
cli_args = cli_parser.parse_args()


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
    max_requests: int
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
        ["guidellm", "benchmark", "run", "--help"],
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


def host_vcpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


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


def _fit_sweep_to_budget(desired_sweep: int, per_run_sec: int) -> tuple[int, int]:
    """Return (sweep_size, max_seconds_per_strategy) within per-run wall budget."""
    sweep = max(2, desired_sweep)
    while sweep >= 2:
        max_seconds = max(BUDGET_MIN_PER_RUN_SEC // 2, per_run_sec // sweep)
        max_seconds = min(max_seconds, BUDGET_MAX_PER_RUN_SEC)
        if sweep * max_seconds <= per_run_sec:
            return sweep, max_seconds
        sweep -= 1
    max_seconds = max(BUDGET_MIN_PER_RUN_SEC // 2, min(per_run_sec, BUDGET_MAX_PER_RUN_SEC))
    return 2, max_seconds


def cpu_kv_cache_gib(spec: ModelSpec, hp: HostProfile) -> int | None:
    if environ.get("VLLM_CPU_KVCACHE_SPACE"):
        return None
    remaining = hp.ram_avail_gb - model_memory_gb(spec) * 1.3
    if remaining < 1.0:
        return None
    return max(1, int(remaining * 0.9))


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

    if mode == "cpu":
        max_conc = sublinear_scale(v, 6.0, 0.65, 32)
        max_workers = min(sublinear_scale(v, 2.0, 0.45, 4), max(2, v // 2))
        max_requests = max(max_conc * 2, sublinear_scale(v, 8.0, 0.55, 50))
        rampup = min(30.0, max(5.0, v / 4.0))
        warmup = "10"
    else:
        max_conc = sublinear_scale(v, 8.0, 0.60, 64)
        max_workers = min(sublinear_scale(v, 3.0, 0.50, 8), max(4, v))
        max_requests = max(max_conc * 2, sublinear_scale(v, 12.0, 0.55, 120))
        rampup = 10.0
        warmup = "0.05"

    desired_sweep = max(
        2,
        min(
            3 + int(math.log2(max(v, 2)) // 2),
            6 + int(math.log2(max(v, 2)) // 3),
        ),
    )
    sweep, max_seconds = _fit_sweep_to_budget(desired_sweep, budget.per_run_sec)

    if sweep_env := (
        _env_int("GUIDELLM_CPU_SWEEP_SIZE", "GUIDELLM_SWEEP_SIZE")
        if mode == "cpu"
        else _env_int("GUIDELLM_GPU_SWEEP_SIZE", "GUIDELLM_SWEEP_SIZE")
    ):
        sweep = max(2, sweep_env)
        _, max_seconds = _fit_sweep_to_budget(sweep, budget.per_run_sec)
    if max_conc_env := _env_int("GUIDELLM__MAX_CONCURRENCY"):
        max_conc = max_conc_env
    if max_req_env := _env_int("GUIDELLM_MAX_REQUESTS", "GUIDELLM_MAX_REQUESTS_CPU"):
        max_requests = max_req_env
    if workers_env := _env_int("GUIDELLM__MAX_WORKER_PROCESSES"):
        max_workers = workers_env

    max_num_seqs = min(max_conc, sublinear_scale(v, 4.0, 0.55, 16))
    max_batched = min(max_model_len, max(2048, sublinear_scale(v, 64.0, 0.50, 2048)))

    dtype = cpu_serve_dtype(spec)
    kv_util = cpu_kv_memory_util(spec, hp)
    gpu_util = gpu_memory_utilization(spec)
    kv_gib = cpu_kv_cache_gib(spec, hp)

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
        "autoconfig vcpus=%s budget_per_run=%ss sweep=%s max_conc=%s max_req=%s "
        "workers=%s max_sec/strategy=%s max_model_len=%s per_workload_server=%s",
        host_profile().vcpus,
        _TUNING.per_run_budget_sec,
        _TUNING.sweep_size,
        _TUNING.max_concurrency,
        _TUNING.max_requests,
        _TUNING.max_workers,
        _TUNING.max_seconds_per_strategy,
        _TUNING.max_model_len,
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
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fp:
            for line in fp:
                if line.startswith("flags"):
                    return frozenset(line.split(":", 1)[1].strip().split())
    except OSError:
        pass
    return frozenset()


def cpu_has_avx512() -> bool:
    return "avx512f" in get_cpu_flags()


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
                (line.split(":", 1)[1].strip() for line in fp if line.startswith("model name")),
                "unknown",
            )
        logger.info("cpu_model=%s", model)
        flags = get_cpu_flags()
        for flag in ("avx512f", "avx2", "asimd"):
            if flag in flags:
                logger.info("cpu_flag_%s=yes", flag)
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
) -> dict[str, str]:
    env = dict(base)
    env.setdefault("VLLM_CPU_OMP_THREADS_BIND", "auto")
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


def cpu_kv_memory_util(spec: ModelSpec, hp: HostProfile) -> float:
    override = environ.get("VLLM_CPU_GPU_MEMORY_UTILIZATION")
    if override:
        return float(override)
    remaining = hp.ram_avail_gb - model_memory_gb(spec) * 1.3
    if remaining <= 0 or hp.ram_total_gb <= 0:
        return cpu_gpu_memory_utilization()
    fraction = (remaining / hp.ram_total_gb) * 0.85
    return max(0.12, min(0.55, fraction))


def cpu_serve_dtype(spec: ModelSpec | None = None) -> str:
    if override := environ.get("VLLM_CPU_DTYPE", "").strip():
        return override
    if spec is not None and spec.short_name == "gemma-2b":
        return "bfloat16"
    if host_arch() == "arm64":
        return "bfloat16"
    if host_arch() == "amd64" and cpu_has_avx512():
        return "bfloat16"
    return "float16"


def gpu_memory_utilization(spec: ModelSpec) -> float:
    override = environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    if override:
        return float(override)
    vram = float(gpu_info()["total_vram_gb"])
    if vram <= 0:
        return 0.9
    if model_memory_gb(spec) > 0.7 * vram:
        return 0.95
    return 0.90


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


@cache
def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_count": 0,
        "gpu_model": None,
        "vram_gb": 0.0,
        "total_vram_gb": 0.0,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        count = torch.cuda.device_count()
        info["gpu_count"] = count
        if count:
            total = 0.0
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total += props.total_memory
                if i == 0:
                    info["gpu_model"] = props.name
            info["total_vram_gb"] = total / 1e9
            info["vram_gb"] = total / 1e9
    except Exception as e:
        logger.debug("gpu_info: %s", e)
    return info


def model_memory_gb(spec: ModelSpec) -> float:
    if spec.memory_gb is not None:
        return spec.memory_gb
    return spec.params_b * 2.0 * 1.2


def available_memory_gb(mode: str) -> float:
    if mode == "gpu":
        total = gpu_info()["total_vram_gb"]
        if total > 0:
            return total * 0.9
    return virtual_memory().available / 1e9 * 0.85


def model_fits(spec: ModelSpec, mode: str) -> bool:
    need = model_memory_gb(spec)
    have = available_memory_gb(mode)
    logger.info(
        "memory check %s: need~%.1f GiB have~%.1f GiB",
        spec.short_name,
        need,
        have,
    )
    return need <= have


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
        return [
            ModelSpec(os.path.basename(m.rstrip("/")), m, max(0.135, len(m) / 20))
            for m in cli_args.models
        ]
    return [spec for spec in DEFAULT_MODELS if model_supported_on_mode(spec, mode)]


def workloads_for_mode(mode: str) -> list[WorkloadSpec]:
    return [w for w in WORKLOADS if not w.gpu_only or mode == "gpu"]


def guidellm_sweep_size(mode: str) -> str:
    """Sweep steps (sync + throughput + constant interpolations). Minimum useful size is 2."""
    if mode == "gpu":
        return environ.get("GUIDELLM_GPU_SWEEP_SIZE") or environ.get("GUIDELLM_SWEEP_SIZE", "3")
    return environ.get("GUIDELLM_CPU_SWEEP_SIZE") or environ.get("GUIDELLM_SWEEP_SIZE", "3")


def guidellm_sweep_profile(mode: str, tuning: BenchmarkTuning) -> str:
    if not tuning.autoconfig:
        return f"sweep,{guidellm_sweep_size(mode)}"
    return (
        f"kind=sweep,sweep_size={tuning.sweep_size},"
        f"max_concurrency={tuning.max_concurrency},"
        f"rampup_duration={tuning.rampup_duration:g}"
    )


def guidellm_throughput_rate(mode: str) -> str:
    """Concurrent streams for standalone throughput profile (legacy plan only; GuideLLM 0.6+)."""
    default = "8" if mode == "cpu" else "16"
    return environ.get("GUIDELLM_THROUGHPUT_RATE", default)


def _guidellm_profile_override(mode: str) -> str:
    return (
        environ.get("GUIDELLM_PROFILES", "").strip().lower()
        or (environ.get("GUIDELLM_CPU_PROFILES", "").strip().lower() if mode == "cpu" else "")
    )


def guidellm_plan(mode: str, tuning: BenchmarkTuning) -> list[tuple[str, str | None]]:
    """(profile, rate) runs per model/workload."""
    override = _guidellm_profile_override(mode)
    if override in ("legacy", "sync-throughput", "sync"):
        return [("synchronous", None), ("throughput", guidellm_throughput_rate(mode))]
    if tuning.autoconfig:
        return [(guidellm_sweep_profile(mode, tuning), None)]
    return [("sweep", guidellm_sweep_size(mode))]


def guidellm_max_seconds(mode: str, spec: ModelSpec, tuning: BenchmarkTuning) -> int:
    if mode == "gpu":
        base = 40 + int(spec.params_b * 8)
    else:
        base = 45 + int(spec.params_b * 12)
    base *= max(1, cli_args.benchmark_timeout_scale)
    if tuning.autoconfig:
        return min(base, tuning.max_seconds_per_strategy)
    return base


def guidellm_max_requests(mode: str, tuning: BenchmarkTuning) -> int:
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


def log_server_start_failure(mode: str, spec: ModelSpec | None = None) -> None:
    if mode != "cpu":
        return
    hints = [
        "vLLM CPU server failed to start.",
        "Use --privileged --shm-size=4g",
    ]
    if host_arch() == "amd64" and not cpu_has_avx512():
        hints.append("BENCHMARK_VLLM_ALLOW_AVX2_ONLY on AVX2-only amd64")
    if spec and spec.short_name == "gemma-2b":
        hints.append("gemma-2b needs bfloat16 on CPU (autoconfig sets VLLM_CPU_DTYPE)")
    logger.error(" ".join(hints))


def tensor_parallel_size(mode: str, spec: ModelSpec) -> int:
    """Largest TP <= GPU count where num_attention_heads % TP == 0 (vLLM requirement)."""
    gpus = max(1, int(gpu_info()["gpu_count"] or 1))
    if mode != "gpu" or gpus <= 1:
        return 1
    heads = spec.num_attention_heads
    if not heads:
        return 1
    for tp in range(min(gpus, heads), 0, -1):
        if heads % tp == 0:
            return tp
    return 1


def start_server(model_id: str, mode: str, max_model_len: int, spec: ModelSpec) -> Popen[Any]:
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
        env = cpu_server_env(env, tuning)
    os.makedirs(cli_args.models_dir, exist_ok=True)
    logger.info("Starting server: %s", " ".join(cmd))
    return Popen(
        cmd,
        stdout=DEVNULL,
        stderr=stderr,
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
    cmd = [
        "guidellm",
        "benchmark",
        "run",
        "--target",
        TARGET_URL,
        "--model",
        spec.model_id,
        "--backend",
        "openai_http",
        "--request-format",
        "/v1/completions",
        "--data",
        f"prompt_tokens={workload.prompt_tokens},output_tokens={workload.output_tokens}",
        "--profile",
        profile,
        "--random-seed",
        "42",
        "--max-seconds",
        str(max_seconds),
        "--max-requests",
        str(guidellm_max_requests(mode, tuning)),
        "--warmup",
        tuning.warmup,
        "--output-dir",
        str(out_dir),
        "--outputs",
        "json",
        "--disable-console",
        "--sample-requests",
        "10",
    ]
    if rate is not None:
        cmd.extend(["--rate", rate])
    elif tuning.autoconfig and tuning.rampup_duration > 0 and profile.startswith("kind=sweep"):
        cmd.extend(["--rampup", str(tuning.rampup_duration)])

    logger.info("GuideLLM: %s", " ".join(cmd))
    try:
        result = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(max_seconds * (tuning.sweep_size + 2), 120),
            check=False,
            env=guidellm_env(),
            cwd=str(out_dir),
        )
    except TimeoutExpired:
        logger.warning("GuideLLM timed out profile=%s workload=%s", profile, workload.name)
        return None

    if result.returncode != 0:
        logger.warning(
            "GuideLLM failed profile=%s workload=%s: %s",
            profile,
            workload.name,
            (result.stderr or result.stdout)[-3000:],
        )
        return None

    report = out_dir / "benchmarks.json"
    if not report.is_file():
        candidates = list(out_dir.glob("**/benchmarks.json"))
        if not candidates:
            logger.warning("No benchmarks.json under %s", out_dir)
            return None
        report = candidates[0]
    return report


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
        "avx512": cpu_has_avx512() if host_arch() == "amd64" else None,
        "avx2_only_image": environ.get("BENCHMARK_VLLM_ALLOW_AVX2_ONLY", "").lower()
        in ("1", "true", "yes"),
        "vllm_version": read_vllm_version(),
        "guidellm_version": read_guidellm_version(),
        "tensor_parallel": tensor_parallel_size(mode, spec) if mode == "gpu" else 0,
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
                    log_server_start_failure(mode, spec)
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
                log_server_start_failure(mode, spec)
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
        log_server_start_failure(mode, spec)
        sys_exit(1)
    finally:
        stop_server(server)


def print_versions() -> str:
    # Pinned files match the image build; avoid vllm CLI stderr noise in meta.json.
    version = f"vllm={read_vllm_version()} guidellm={read_guidellm_version()}"
    print(version)
    return version


def main() -> None:
    if not shutil.which("guidellm"):
        logger.error("guidellm CLI not found in PATH")
        sys_exit(1)

    if cli_args.version:
        print_versions()
        sys_exit(0)

    log_cpu_details()
    mode = detect_mode()
    if cli_args.probe_only:
        if mode == "cpu":
            log_docker_cpu_hints()
        run_probe_only(mode)
        return

    check_cpu_isa_compat(mode)
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
        if not run_model(spec, mode, start):
            stop_ladder = True

    sys_exit(0)


if __name__ == "__main__":
    main()
