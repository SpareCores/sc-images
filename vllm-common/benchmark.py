#!/usr/bin/env python3
"""vLLM serving benchmark: start vllm serve, run GuideLLM, emit JSONL metrics."""

from __future__ import annotations

import json
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
    serve_extra_args: tuple[str, ...] = ()
    gpu_only: bool = False
    cpu_only: bool = False


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("smol-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", 0.135),
    ModelSpec("qwen-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 0.5),
    ModelSpec("gemma-2b", "google/gemma-2-2b-it", 2.0),
    ModelSpec("llama-8b", "meta-llama/Llama-3.1-8B-Instruct", 8.0),
    ModelSpec("phi-4", "microsoft/phi-4", 14.0),
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
    result = run(["vllm", "--version"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return (result.stdout or result.stderr).strip().splitlines()[0]
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
    return env


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


def cpu_server_env(base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.setdefault("VLLM_CPU_OMP_THREADS_BIND", "auto")
    if environ.get("VLLM_CPU_KVCACHE_SPACE"):
        env.setdefault("VLLM_CPU_KVCACHE_SPACE", environ["VLLM_CPU_KVCACHE_SPACE"])
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


def cpu_serve_dtype() -> str:
    return environ.get("VLLM_CPU_DTYPE", "float16")


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


def guidellm_throughput_rate(mode: str) -> str:
    """Concurrent streams for standalone throughput profile (legacy plan only; GuideLLM 0.6+)."""
    default = "8" if mode == "cpu" else "16"
    return environ.get("GUIDELLM_THROUGHPUT_RATE", default)


def _guidellm_profile_override(mode: str) -> str:
    return (
        environ.get("GUIDELLM_PROFILES", "").strip().lower()
        or (environ.get("GUIDELLM_CPU_PROFILES", "").strip().lower() if mode == "cpu" else "")
    )


def guidellm_plan(mode: str) -> list[tuple[str, str | None]]:
    """(profile, rate) runs per model/workload."""
    override = _guidellm_profile_override(mode)
    if override in ("legacy", "sync-throughput", "sync"):
        return [("synchronous", None), ("throughput", guidellm_throughput_rate(mode))]
    return [("sweep", guidellm_sweep_size(mode))]


def guidellm_max_seconds(mode: str, spec: ModelSpec) -> int:
    if mode == "gpu":
        base = 40 + int(spec.params_b * 8)
    else:
        base = 45 + int(spec.params_b * 12)
    return base * max(1, cli_args.benchmark_timeout_scale)


def guidellm_max_requests(mode: str) -> int:
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


def log_server_start_failure(mode: str) -> None:
    if mode != "cpu":
        return
    logger.error(
        "vLLM CPU server failed to start. Use --privileged --shm-size=4g and "
        "BENCHMARK_VLLM_ALLOW_AVX2_ONLY on AVX2-only amd64."
    )


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
    if mode == "gpu":
        if any("bitsandbytes" in a for a in spec.serve_extra_args) and gpus > 1:
            cmd.extend(["--pipeline-parallel-size", str(gpus)])
        else:
            cmd.extend(
                [
                    "--tensor-parallel-size",
                    str(gpus),
                    "--gpu-memory-utilization",
                    "0.9",
                ]
            )
    else:
        mem_util = cpu_gpu_memory_utilization()
        cmd.extend(
            [
                "--dtype",
                cpu_serve_dtype(),
                "--gpu-memory-utilization",
                f"{mem_util:.2f}",
            ]
        )

    env = os.environ.copy()
    env.setdefault("HF_HOME", cli_args.models_dir)
    env.setdefault("HUGGINGFACE_HUB_CACHE", cli_args.models_dir)
    if mode == "cpu":
        env = cpu_server_env(env)
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
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
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
        str(guidellm_max_seconds(mode, spec)),
        "--max-requests",
        str(guidellm_max_requests(mode)),
        "--warmup",
        "0.05",
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

    logger.info("GuideLLM: %s", " ".join(cmd))
    try:
        result = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=guidellm_max_seconds(mode, spec) * 12,
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
        "avx512": cpu_has_avx512() if host_arch() == "amd64" else None,
        "avx2_only_image": environ.get("BENCHMARK_VLLM_ALLOW_AVX2_ONLY", "").lower()
        in ("1", "true", "yes"),
        "vllm_version": read_vllm_version(),
        "guidellm_version": read_guidellm_version(),
        "tensor_parallel": gi["gpu_count"] if mode == "gpu" else 0,
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


def run_model(spec: ModelSpec, mode: str, start_time: float) -> bool:
    if monotonic() - start_time > OVERALL_TIMEOUT_SEC:
        logger.warning("Overall timeout reached")
        return False

    max_len = max(w.max_model_len for w in workloads_for_mode(mode))
    server = None
    try:
        server = start_server(spec.model_id, mode, max_len, spec)
        health_timeout = (
            SERVER_START_TIMEOUT_CPU_SEC if mode == "cpu" else SERVER_START_TIMEOUT_GPU_SEC
        )
        if not wait_for_health(health_timeout, server):
            logger.warning("Server health check failed for %s", spec.model_id)
            log_server_start_failure(mode)
            return True

        peak_output_tps = 0.0
        for workload in workloads_for_mode(mode):
            for profile, rate in guidellm_plan(mode):
                if monotonic() - start_time > OVERALL_TIMEOUT_SEC:
                    return False
                with tempfile.TemporaryDirectory(prefix="guidellm-") as tmp:
                    report = run_guidellm(
                        spec,
                        workload,
                        profile,
                        rate,
                        mode,
                        Path(tmp),
                    )
                    if not report:
                        continue
                    n = report_to_jsonl(report, spec, workload, profile, mode)
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
                            peak_output_tps = max(
                                peak_output_tps, float(block["mean"])
                            )

        if peak_output_tps > 0 and peak_output_tps < MIN_OUTPUT_TOKENS_PER_SEC:
            logger.warning(
                "Peak output %.2f tok/s below threshold; stopping ladder",
                peak_output_tps,
            )
            return False
        return True
    finally:
        stop_server(server)


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
    server = None
    try:
        server = start_server(spec.model_id, mode, 2048, spec)
        if wait_for_health(probe_health_timeout_sec(mode), server):
            logger.info("probe_ok model=%s mode=%s", spec.model_id, mode)
            sys_exit(0)
        log_server_start_failure(mode)
        sys_exit(1)
    finally:
        stop_server(server)


def print_versions() -> None:
    print(f"vllm={get_vllm_runtime_version()} guidellm={guidellm_runtime_version()}")


def main() -> None:
    if not shutil.which("guidellm"):
        logger.error("guidellm CLI not found in PATH")
        sys_exit(1)

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

    if cli_args.version:
        print_versions()
        sys_exit(0)

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
