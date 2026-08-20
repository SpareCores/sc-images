#!/usr/bin/env python3
"""FFmpeg aggregate transcoding-capacity benchmark.

The benchmark launches independent, single-threaded FFmpeg jobs behind a
shared start barrier. Aggregate throughput is completed media divided by the
group makespan; individual FFmpeg speed values are deliberately not summed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from logging import DEBUG, StreamHandler, basicConfig, getLogger
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - image always has psutil
    psutil = None  # type: ignore[assignment]

basicConfig(
    level=DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler(sys.stderr)],
)
logger = getLogger("benchmark-ffmpeg")

BENCHMARK_NAME = "ffmpeg_transcoding"
BENCHMARK_VERSION = "3.1.0"

VIDEO_CALIBRATION_DURATION_SEC = float(
    os.environ.get("FFMPEG_BENCH_VIDEO_CALIBRATION_SECONDS", "1")
)
AUDIO_SAMPLE_RATE = 44_100
AUDIO_CHANNELS = 2
AUDIO_CALIBRATION_DURATION_SEC = float(
    os.environ.get("FFMPEG_BENCH_AUDIO_CALIBRATION_SECONDS", "5")
)
TARGET_REPETITION_SEC = float(os.environ.get("FFMPEG_BENCH_TARGET_SECONDS", "5"))
MIN_MEDIA_DURATION_SEC = float(os.environ.get("FFMPEG_BENCH_MIN_MEDIA_SECONDS", "0.5"))
MAX_MEDIA_DURATION_SEC = float(os.environ.get("FFMPEG_BENCH_MAX_MEDIA_SECONDS", "1800"))
REPETITIONS = max(1, int(os.environ.get("FFMPEG_BENCH_REPETITIONS", "3")))
MAX_REPETITIONS = max(
    REPETITIONS,
    int(os.environ.get("FFMPEG_BENCH_MAX_REPETITIONS", "5")),
)
CV_THRESHOLD = max(0.0, float(os.environ.get("FFMPEG_BENCH_CV_THRESHOLD", "0.10")))
OVERSUBSCRIPTION = max(1.0, float(os.environ.get("FFMPEG_BENCH_OVERSUBSCRIPTION", "1")))
OVERALL_TIMEOUT_SEC = int(os.environ.get("FFMPEG_BENCH_TIMEOUT_SECONDS", str(2 * 60 * 60)))
REPETITION_TIMEOUT_SEC = float(
    os.environ.get(
        "FFMPEG_BENCH_REPETITION_TIMEOUT_SECONDS",
        str(max(30.0, TARGET_REPETITION_SEC * 6)),
    )
)
MEMORY_PER_CPU_AUDIO_WORKER_MB = 64
MEMORY_PER_VIDEO_WORKER_MB = 320
GPU_ENCODE_SESSIONS_PER_GPU = max(
    1, int(os.environ.get("FFMPEG_BENCH_GPU_ENCODE_SESSIONS_PER_GPU", "8"))
)
GPU_DECODE_SESSIONS_PER_GPU = max(
    1, int(os.environ.get("FFMPEG_BENCH_GPU_DECODE_SESSIONS_PER_GPU", "8"))
)
# Keep searching only while the next larger pool beats the previous by this ratio.
SCALE_CONTINUE_RATIO = max(1.0, float(os.environ.get("FFMPEG_BENCH_SCALE_CONTINUE_RATIO", "1.08")))
PID_TASKS_PER_WORKER = max(2, int(os.environ.get("FFMPEG_BENCH_PID_TASKS_PER_WORKER", "4")))
FIXTURE_PATH = Path(os.environ.get("FFMPEG_BENCH_AUDIO_SOURCE", "/opt/benchmark-ffmpeg/source.flac"))
FIXTURE_VIDEO_PATH = Path(
    os.environ.get("FFMPEG_BENCH_VIDEO_SOURCE", "/opt/benchmark-ffmpeg/source.mp4")
)
FIXTURE_VIDEO_HEVC_PATH = Path(
    os.environ.get(
        "FFMPEG_BENCH_VIDEO_HEVC_SOURCE", "/opt/benchmark-ffmpeg/source-hevc.mp4"
    )
)
FIXTURE_VIDEO_SHA256 = os.environ.get(
    "FFMPEG_BENCH_VIDEO_SHA256",
    "a49fe8b82c96bafcc344d374facb716f481e3fa9c6753d56f6d1e0ed509a14e7",
)
FIXTURE_VIDEO_HEVC_SHA256 = os.environ.get(
    "FFMPEG_BENCH_VIDEO_HEVC_SHA256",
    "79a70a6aa81e745d650621768dee5cd3fc1da7d2c15d39871125737064f6cde7",
)
CDN_FIXTURE_BASE = os.environ.get(
    "FFMPEG_BENCH_CDN_BASE_URL",
    "https://cdn.sparecores.net/sc-inspector/benchmark-ffmpeg",
).rstrip("/")
FIXTURE_AUDIO_SHA256 = "4445399abe62c9d7c546711a853fccfab8ab274226d2e80aa0e5ad948589e516"

CPU_LIST_PART_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    uuid: str = ""


@dataclass(frozen=True)
class NumaNode:
    index: int
    cpus: tuple[int, ...]


@dataclass(frozen=True)
class HostProfile:
    cpu_count: int
    affinity_cpu_count: int
    affinity_cpus: tuple[int, ...]
    cpu_quota_cores: float | None
    cpu_capacity: float
    architecture: str
    ram_total_gb: float
    ram_avail_gb: float
    memory_limit_gb: float | None
    pids_limit: int | None
    pids_current: int | None
    cgroup_cpu_max: str
    cgroup_cpuset_effective: str
    cgroup_cpu_stat_before: dict[str, int]
    numa_nodes: tuple[NumaNode, ...]
    numa_binding: bool
    gpus: tuple[GpuInfo, ...]
    physical_cpu_count: int
    logical_cpu_count: int


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    backend: str  # cpu | gpu
    operation: str  # encode | decode
    media_type: str  # video | audio
    codec: str
    requires_encoder: str | None = None
    requires_decoder: str | None = None
    # Which pinned video fixture to feed: "h264" (source.mp4) or "hevc" (source-hevc.mp4).
    # Encode scenarios always start from the H.264 excerpt; H.265 decode needs HEVC bits.
    source_codec: str = "h264"
    vbench_scenario: str | None = None
    encode_args: tuple[str, ...] = ()
    bitrate_kbps: int | None = None
    compression_level: int | None = None
    sample_rate_hz: int = AUDIO_SAMPLE_RATE
    channels: int = AUDIO_CHANNELS


VIDEO_SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "cpu_h264_encode", "cpu", "encode", "video", "libx264",
        requires_encoder="libx264", vbench_scenario="upload",
        encode_args=("-crf", "18"),
    ),
    ScenarioSpec(
        "cpu_h265_encode", "cpu", "encode", "video", "libx265",
        requires_encoder="libx265", vbench_scenario="upload",
        encode_args=("-crf", "18"),
    ),
    ScenarioSpec(
        "cpu_h264_decode", "cpu", "decode", "video", "h264",
        requires_decoder="h264",
    ),
    ScenarioSpec(
        "cpu_h265_decode", "cpu", "decode", "video", "hevc",
        requires_decoder="hevc", source_codec="hevc",
    ),
    ScenarioSpec(
        "gpu_h264_encode", "gpu", "encode", "video", "h264_nvenc",
        requires_encoder="h264_nvenc",
    ),
    ScenarioSpec(
        "gpu_h264_decode", "gpu", "decode", "video", "h264_cuvid",
        requires_decoder="h264_cuvid",
    ),
    ScenarioSpec(
        "gpu_h265_encode", "gpu", "encode", "video", "hevc_nvenc",
        requires_encoder="hevc_nvenc",
    ),
    ScenarioSpec(
        "gpu_h265_decode", "gpu", "decode", "video", "hevc_cuvid",
        requires_decoder="hevc_cuvid", source_codec="hevc",
    ),
)

AUDIO_SCENARIOS: tuple[ScenarioSpec, ...] = tuple(
    ScenarioSpec(
        f"ogg_vorbis_{bitrate}k",
        "cpu",
        "encode",
        "audio",
        "libvorbis",
        requires_encoder="libvorbis",
        encode_args=("-b:a", f"{bitrate}k"),
        bitrate_kbps=bitrate,
        # libvorbis rejects 24 kbps stereo at 44.1 kHz. A 16 kHz low-bandwidth
        # profile is the lowest exact managed-bitrate profile it supports.
        sample_rate_hz=16_000 if bitrate == 24 else AUDIO_SAMPLE_RATE,
    )
    for bitrate in (24, 96, 160, 320)
) + (
    ScenarioSpec(
        "flac_lossless",
        "cpu",
        "encode",
        "audio",
        "flac",
        requires_encoder="flac",
        encode_args=("-compression_level", "5"),
        compression_level=5,
    ),
)

SCENARIOS = VIDEO_SCENARIOS + AUDIO_SCENARIOS


@dataclass
class RepetitionResult:
    repetition: int
    wall_time_sec: float
    processed_frames: int | None
    aggregate_fps: float | None
    processed_audio_seconds: float | None
    audio_seconds_per_sec: float | None
    finish_spread_sec: float
    successful_workers: int
    failed_workers: int
    timed_out_workers: int
    error: str = ""


@dataclass
class ScalingStep:
    workers: int
    media_duration_sec: float
    failed_workers: int
    repetitions: list[RepetitionResult] = field(default_factory=list)


@dataclass(frozen=True)
class VideoFixture:
    path: Path
    width: int
    height: int
    fps: float
    duration_sec: float
    codec: str
    title: str
    license_url: str
    sha256: str


@dataclass
class ScenarioResult:
    name: str
    backend: str
    operation: str
    media_type: str
    codec: str
    resolution: str
    source_fps: float
    threads_per_worker: int
    vbench_scenario: str | None = None
    bitrate_kbps: int | None = None
    compression_level: int | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    scaling_steps: list[ScalingStep] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG", "ffmpeg")


def ffprobe_bin() -> str:
    return os.environ.get("FFPROBE", "ffprobe")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def read_int(path: Path) -> int | None:
    value = read_text(path)
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for part in value.split(","):
        match = CPU_LIST_PART_RE.match(part.strip())
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        cpus.update(range(start, end + 1))
    return tuple(sorted(cpus))


def affinity_cpus() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        return tuple(range(os.cpu_count() or 1))


def physical_cpu_count(cpus: tuple[int, ...]) -> int:
    """Count cores, not SMT siblings, within the allowed CPU set.

    Independent single-threaded FFmpeg workers contend on a physical core once
    both hyperthreads are occupied. Using ``nproc``/affinity therefore
    over-subscribes HT systems and can lower decode throughput.
    """
    allowed = set(cpus)
    if not allowed:
        return 1
    groups: set[tuple[int, ...]] = set()
    for cpu in cpus:
        siblings = parse_cpu_list(
            read_text(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
        )
        group = tuple(sorted(cpu for cpu in siblings if cpu in allowed))
        groups.add(group or (cpu,))
    return max(1, len(groups))


def cgroup_cpu_profile() -> tuple[str, float | None]:
    value = read_text(Path("/sys/fs/cgroup/cpu.max"))
    if not value:
        return "", None
    parts = value.split()
    if len(parts) != 2 or parts[0] == "max":
        return value, None
    try:
        quota, period = int(parts[0]), int(parts[1])
    except ValueError:
        return value, None
    return value, quota / period if quota > 0 and period > 0 else None


def cgroup_cpu_stat() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in read_text(Path("/sys/fs/cgroup/cpu.stat")).splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result


def cgroup_limits() -> tuple[float | None, int | None]:
    memory_bytes = read_int(Path("/sys/fs/cgroup/memory.max"))
    pids_limit = read_int(Path("/sys/fs/cgroup/pids.max"))
    memory_gb = memory_bytes / (1024**3) if memory_bytes is not None else None
    return memory_gb, pids_limit


def ram_profile(memory_limit_gb: float | None) -> tuple[float, float]:
    if psutil is None:
        return 0.0, 0.0
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    available = mem.available / (1024**3)
    if memory_limit_gb is not None:
        total = min(total, memory_limit_gb)
        current = read_int(Path("/sys/fs/cgroup/memory.current")) or 0
        available = min(available, max(0.0, memory_limit_gb - current / (1024**3)))
    return total, available


def detect_numa_nodes(cpus_allowed: tuple[int, ...]) -> list[NumaNode]:
    allowed = set(cpus_allowed)
    nodes: list[NumaNode] = []
    node_root = Path("/sys/devices/system/node")
    for path in sorted(node_root.glob("node[0-9]*"), key=lambda item: int(item.name[4:])):
        cpus = tuple(cpu for cpu in parse_cpu_list(read_text(path / "cpulist")) if cpu in allowed)
        if cpus:
            nodes.append(NumaNode(index=int(path.name[4:]), cpus=cpus))
    return nodes or [NumaNode(index=0, cpus=cpus_allowed)]


def run_command(
    args: list[str],
    *,
    timeout: float | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def numa_binding_usable(nodes: list[NumaNode]) -> bool:
    if len(nodes) <= 1 or shutil.which("numactl") is None:
        return False
    for node in nodes:
        proc = run_command(
            ["numactl", f"--cpunodebind={node.index}", f"--membind={node.index}", "true"],
            timeout=5,
        )
        if proc.returncode != 0:
            logger.warning("NUMA binding unavailable: %s", proc.stderr.strip())
            return False
    return True


def detect_gpus() -> list[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    proc = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if proc.returncode != 0:
        logger.warning("nvidia-smi failed: %s", proc.stderr.strip())
        return []
    gpus: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append(GpuInfo(index=index, name=parts[1], uuid=parts[2] if len(parts) > 2 else ""))
    return gpus


def host_profile() -> HostProfile:
    cpus = affinity_cpus()
    cpu_max, quota = cgroup_cpu_profile()
    capacity = min(float(len(cpus)), quota) if quota is not None else float(len(cpus))
    logical = max(1, min(len(cpus), math.ceil(capacity)))
    physical = min(physical_cpu_count(cpus), logical)
    memory_limit_gb, pids_limit = cgroup_limits()
    ram_total_gb, ram_avail_gb = ram_profile(memory_limit_gb)
    nodes = detect_numa_nodes(cpus)
    return HostProfile(
        cpu_count=physical,
        affinity_cpu_count=len(cpus),
        affinity_cpus=cpus,
        cpu_quota_cores=round(quota, 3) if quota is not None else None,
        cpu_capacity=round(capacity, 3),
        architecture=platform.machine(),
        ram_total_gb=round(ram_total_gb, 2),
        ram_avail_gb=round(ram_avail_gb, 2),
        memory_limit_gb=round(memory_limit_gb, 2) if memory_limit_gb is not None else None,
        pids_limit=pids_limit,
        pids_current=read_int(Path("/sys/fs/cgroup/pids.current")),
        cgroup_cpu_max=cpu_max,
        cgroup_cpuset_effective=read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective")),
        cgroup_cpu_stat_before=cgroup_cpu_stat(),
        numa_nodes=tuple(nodes),
        numa_binding=numa_binding_usable(nodes),
        gpus=tuple(detect_gpus()),
        physical_cpu_count=physical,
        logical_cpu_count=logical,
    )


CODEC_LINE_RE = re.compile(r"^\s+[VAS][\.\w]+\s+(\S+)\s")


def ffmpeg_inventory() -> tuple[set[str], set[str]]:
    encoders: set[str] = set()
    decoders: set[str] = set()
    for flag, target in (("-encoders", encoders), ("-decoders", decoders)):
        proc = run_command([ffmpeg_bin(), "-hide_banner", flag], timeout=30)
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            match = CODEC_LINE_RE.match(line)
            if match:
                target.add(match.group(1))
    return encoders, decoders


def ffmpeg_version() -> str:
    proc = run_command([ffmpeg_bin(), "-version"], timeout=15)
    if proc.returncode != 0:
        return "unknown"
    first = proc.stdout.splitlines()
    return first[0].removeprefix("ffmpeg version ").split(" Copyright")[0] if first else "unknown"


def ffmpeg_buildconf() -> list[str]:
    proc = run_command([ffmpeg_bin(), "-buildconf"], timeout=15)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("--")]


def scenario_supported(
    spec: ScenarioSpec,
    encoders: set[str],
    decoders: set[str],
    gpus: list[GpuInfo],
) -> tuple[bool, str]:
    if spec.backend == "gpu" and not gpus:
        return False, "no_cuda_gpu_detected"
    if spec.requires_encoder and spec.requires_encoder not in encoders:
        return False, f"missing_encoder:{spec.requires_encoder}"
    if spec.requires_decoder and spec.requires_decoder not in decoders:
        return False, f"missing_decoder:{spec.requires_decoder}"
    return True, ""


def worker_ladder(cpu_target: int, max_workers: int, *, backend: str = "cpu") -> list[int]:
    """Cheap anchors, then optional adaptive points in ``run_scenario``.

    CPU: 1, P/2, P physical cores. GPU: 1 and one session per GPU. Doubling
    past those anchors happens only while aggregate rate is still rising.
    """
    override = os.environ.get("FFMPEG_BENCH_WORKERS")
    if override:
        requested = [int(value) for value in override.split(",") if int(value) > 0]
        return sorted(set(min(value, max_workers) for value in requested))
    cpu_target = max(1, min(cpu_target, max_workers))
    if backend == "gpu":
        values = {1, cpu_target}
    else:
        values = {1, max(1, cpu_target // 2), cpu_target}
    if OVERSUBSCRIPTION > 1:
        values.add(
            min(max_workers, max(cpu_target, math.ceil(cpu_target * OVERSUBSCRIPTION)))
        )
    return sorted(value for value in values if 0 < value <= max_workers)


def max_workers_for_host(host: HostProfile, spec: ScenarioSpec, gpu_count: int) -> tuple[int, int]:
    if spec.backend == "gpu":
        per_gpu = (
            GPU_DECODE_SESSIONS_PER_GPU
            if spec.operation == "decode"
            else GPU_ENCODE_SESSIONS_PER_GPU
        )
        target = max(1, gpu_count)
        desired = max(target, gpu_count * per_gpu)
    else:
        target = host.cpu_count
        desired = max(
            target,
            host.logical_cpu_count if host.logical_cpu_count else target,
            math.ceil(target * OVERSUBSCRIPTION),
        )
    memory_mb = MEMORY_PER_CPU_AUDIO_WORKER_MB if spec.media_type == "audio" else MEMORY_PER_VIDEO_WORKER_MB
    ram_cap = max(1, int(host.ram_avail_gb * 1024 // memory_mb)) if host.ram_avail_gb > 0 else desired
    pid_cap = (
        max(1, (host.pids_limit - (host.pids_current or 0) - 16) // PID_TASKS_PER_WORKER)
        if host.pids_limit is not None
        else desired
    )
    env_cap = int(os.environ.get("FFMPEG_BENCH_MAX_WORKERS", str(desired)))
    return target, max(1, min(desired, ram_cap, pid_cap, env_cap))


def make_work_dir() -> tempfile.TemporaryDirectory[str]:
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free >= 512 * 1024 * 1024:
                return tempfile.TemporaryDirectory(prefix="ffmpeg-bench-", dir=str(shm))
        except OSError:
            pass
    return tempfile.TemporaryDirectory(prefix="ffmpeg-bench-")


def parse_frame_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        denominator = float(den)
        return float(num) / denominator if denominator else 0.0
    return float(value)


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"fixture checksum mismatch for {path}")


def load_video_fixture(
    path: Path = FIXTURE_VIDEO_PATH,
    *,
    expected_sha256: str = FIXTURE_VIDEO_SHA256,
) -> VideoFixture:
    if not path.is_file():
        raise FileNotFoundError(f"video fixture not found: {path}")
    if expected_sha256:
        verify_sha256(path, expected_sha256)
    probe = probe_source(path, "v:0")
    stream = (probe.get("streams") or [{}])[0]
    fmt = probe.get("format") or {}
    fps = parse_frame_rate(
        str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    )
    duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    codec = str(stream.get("codec_name") or "")
    if fps <= 0 or duration <= 0 or width <= 0 or height <= 0 or not codec:
        raise RuntimeError(f"invalid video fixture probe for {path}: {probe}")
    return VideoFixture(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration,
        codec=codec,
        title="Tears of Steel",
        license_url="https://mango.blender.org/sharing/",
        sha256=expected_sha256,
    )


def video_fixture_for(spec: ScenarioSpec, fixtures: dict[str, VideoFixture]) -> VideoFixture:
    key = spec.source_codec if spec.media_type == "video" else "h264"
    try:
        return fixtures[key]
    except KeyError as exc:
        raise KeyError(f"no video fixture for source_codec={key!r}") from exc


def input_decoder_for(spec: ScenarioSpec) -> str:
    """Forced -c:v before -i so stream_loop decode stays on the intended codec."""
    if spec.backend == "gpu":
        return "hevc_cuvid" if spec.source_codec == "hevc" else "h264_cuvid"
    return "hevc" if spec.source_codec == "hevc" else "h264"


def stage_video_source(source: Path, work_dir: Path) -> Path:
    destination = work_dir / source.name
    shutil.copy2(source, destination)
    return destination


def scenario_video_fields(
    spec: ScenarioSpec, video: VideoFixture | None
) -> tuple[str, float]:
    if spec.media_type != "video" or video is None:
        return "", 0.0
    return f"{video.width}x{video.height}", video.fps


def video_fps_for(spec: ScenarioSpec, video: VideoFixture | None) -> float:
    _, fps = scenario_video_fields(spec, video)
    return fps


def probe_source(path: Path, stream: str) -> dict[str, Any]:
    entries = (
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration"
        if stream == "v:0"
        else "stream=codec_name,sample_rate,channels,channel_layout,bits_per_raw_sample,duration"
    )
    proc = run_command(
        [
            ffprobe_bin(), "-v", "error", "-select_streams", stream,
            "-show_entries", entries, "-show_entries", "format=duration,size,bit_rate",
            "-of", "json", str(path),
        ],
        timeout=30,
    )
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def stage_audio_sources(source: Path, work_dir: Path, host: HostProfile) -> list[Path]:
    if not source.is_file():
        raise FileNotFoundError(f"CC0 FLAC fixture not found: {source}")
    paths: list[Path] = []
    for node in host.numa_nodes:
        destination = work_dir / f"source-node{node.index}.flac"
        if host.numa_binding:
            proc = run_command(
                [
                    "numactl", f"--cpunodebind={node.index}", f"--membind={node.index}",
                    "cp", str(source), str(destination),
                ],
                timeout=30,
            )
            if proc.returncode != 0:
                logger.warning("node-local fixture copy failed; using regular copy")
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)
        paths.append(destination)
    return paths


def assign_gpu(worker_id: int, gpus: list[GpuInfo]) -> int | None:
    return gpus[worker_id % len(gpus)].index if gpus else None


def assign_workers_to_nodes(workers: int, nodes: tuple[NumaNode, ...]) -> list[NumaNode]:
    if not nodes:
        raise ValueError("NUMA nodes are required")
    if len(nodes) == 1:
        return [nodes[0]] * workers
    total_cpus = sum(len(node.cpus) for node in nodes)
    raw = [workers * len(node.cpus) / total_cpus for node in nodes]
    counts = [math.floor(value) for value in raw]
    for idx in sorted(
        range(len(nodes)),
        key=lambda item: (raw[item] - counts[item], len(nodes[item].cpus)),
        reverse=True,
    )[: max(0, workers - sum(counts))]:
        counts[idx] += 1
    while sum(counts) > workers:
        idx = max(range(len(nodes)), key=lambda item: counts[item])
        counts[idx] -= 1
    plan: list[NumaNode] = []
    for node, count in zip(nodes, counts):
        plan.extend([node] * count)
    return plan or [nodes[0]] * workers


def build_worker_command(
    spec: ScenarioSpec,
    input_path: Path,
    media_duration_sec: float,
    *,
    gpu_index: int | None,
) -> list[str]:
    cmd = [
        ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error", "-nostats",
        "-y", "-filter_threads", "1", "-filter_complex_threads", "1",
    ]
    if spec.media_type == "audio":
        cmd += [
            "-stream_loop", "-1", "-i", str(input_path), "-map", "0:a:0",
            "-vn", "-sn", "-dn", "-t", f"{media_duration_sec:.6f}",
            "-ar", str(spec.sample_rate_hz), "-ac", str(spec.channels),
            "-threads:a", "1", "-c:a", spec.codec, *spec.encode_args, "-f", "null", "-",
        ]
        return cmd

    if spec.backend == "cpu":
        cmd += ["-threads", "1"]
    if spec.backend == "gpu":
        assert gpu_index is not None
        cmd += [
            "-hwaccel", "cuda", "-hwaccel_device", str(gpu_index),
            "-hwaccel_output_format", "cuda",
        ]
    cmd += ["-stream_loop", "-1"]
    if spec.media_type == "video":
        # Full GPU decode before NVENC; avoids auto_scale/null failures when
        # stream_loop runs past the end of the source clip. CPU decode/encode
        # pin a single-threaded software decoder for the selected fixture.
        decoder = input_decoder_for(spec)
        if spec.backend == "cpu":
            cmd += ["-threads:v", "1", "-c:v", decoder]
        else:
            cmd += ["-c:v", decoder]
    cmd += ["-i", str(input_path), "-an", "-t", f"{media_duration_sec:.6f}"]
    if spec.operation == "encode":
        if spec.backend == "gpu":
            cmd += ["-c:v", spec.codec, "-preset", "p4", "-gpu", str(gpu_index)]
        else:
            encode_args = list(spec.encode_args)
            if spec.codec == "libx265":
                encode_args += ["-x265-params", "frame-threads=1:pools=none"]
            cmd += ["-threads:v", "1", "-c:v", spec.codec, *encode_args]
    cmd += ["-f", "null", "-"]
    return cmd


def wrap_for_numa(command: list[str], node: NumaNode, enabled: bool) -> list[str]:
    if not enabled:
        return command
    return [
        "numactl", f"--cpunodebind={node.index}", f"--membind={node.index}", *command,
    ]


def terminate_processes(processes: dict[int, subprocess.Popen[bytes]]) -> None:
    for proc in processes.values():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(proc.poll() is None for proc in processes.values()):
        time.sleep(0.01)
    for proc in processes.values():
        if proc.poll() is None:
            proc.kill()
    for proc in processes.values():
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def summarize_repetition(
    spec: ScenarioSpec,
    workers: int,
    media_duration_sec: float,
    repetition: int,
    finish_times: list[float],
    return_codes: list[int],
    *,
    video_fps: float = 0.0,
    timed_out: int = 0,
    error_text: str = "",
) -> RepetitionResult:
    wall = max(finish_times, default=0.0)
    successful = sum(code == 0 for code in return_codes)
    failed = workers - successful
    total_media = successful * media_duration_sec
    processed_frames = (
        round(total_media * video_fps) if spec.media_type == "video" and video_fps > 0 else None
    )
    aggregate_fps = (
        processed_frames / wall
        if processed_frames is not None and wall > 0
        else None
    )
    processed_audio_seconds = total_media if spec.media_type == "audio" else None
    audio_rate = (
        processed_audio_seconds / wall
        if processed_audio_seconds is not None and wall > 0
        else None
    )
    return RepetitionResult(
        repetition=repetition,
        wall_time_sec=wall,
        processed_frames=processed_frames,
        aggregate_fps=aggregate_fps,
        processed_audio_seconds=processed_audio_seconds,
        audio_seconds_per_sec=audio_rate,
        finish_spread_sec=max(finish_times, default=0.0) - min(finish_times, default=0.0),
        successful_workers=successful,
        failed_workers=failed,
        timed_out_workers=timed_out,
        error=error_text,
    )


def run_group_once(
    spec: ScenarioSpec,
    input_paths: list[Path],
    workers: int,
    media_duration_sec: float,
    gpus: list[GpuInfo],
    host: HostProfile,
    repetition: int,
    deadline_monotonic: float,
    *,
    video_fps: float = 0.0,
) -> RepetitionResult:
    read_fd, write_fd = os.pipe()
    processes: dict[int, subprocess.Popen[bytes]] = {}
    finish_times: list[float] = []
    return_codes: list[int] = []
    timed_out = 0
    error_text = ""
    worker_nodes = assign_workers_to_nodes(workers, host.numa_nodes)
    remaining_budget = max(0.0, deadline_monotonic - time.monotonic())
    timeout_sec = min(max(1.0, remaining_budget), REPETITION_TIMEOUT_SEC)
    if remaining_budget <= 0:
        return RepetitionResult(
            repetition, 0.0, None, None, None, None,
            0.0, 0, workers, workers, "overall timeout",
        )
    try:
        with tempfile.TemporaryFile() as error_file:
            try:
                for worker_id in range(workers):
                    node = worker_nodes[worker_id]
                    input_path = input_paths[worker_id % len(input_paths)]
                    command = build_worker_command(
                        spec,
                        input_path,
                        media_duration_sec,
                        gpu_index=assign_gpu(worker_id, gpus) if spec.backend == "gpu" else None,
                    )
                    command = wrap_for_numa(command, node, host.numa_binding)
                    barrier = f'read -r ignored <&{read_fd} || true; exec "$@"'
                    proc = subprocess.Popen(
                        ["/bin/sh", "-c", barrier, "ffmpeg-barrier", *command],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=error_file,
                        pass_fds=(read_fd,),
                    )
                    processes[proc.pid] = proc
            except (OSError, subprocess.SubprocessError) as exc:
                error_text = f"worker launch failed: {exc}"
                terminate_processes(processes)
                return RepetitionResult(
                    repetition, 0.0, None, None, None, None,
                    0.0, 0, workers, 0, error_text,
                )
            finally:
                os.close(read_fd)
                read_fd = -1

            start_ns = time.monotonic_ns()
            os.close(write_fd)
            write_fd = -1
            deadline = time.monotonic() + timeout_sec
            remaining = set(processes)
            while remaining and time.monotonic() < deadline:
                while True:
                    try:
                        pid, status = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        pid = 0
                        break
                    if pid == 0:
                        break
                    proc = processes.get(pid)
                    if proc is not None:
                        proc.returncode = os.waitstatus_to_exitcode(status)
                        return_codes.append(proc.returncode)
                        finish_times.append((time.monotonic_ns() - start_ns) / 1e9)
                        remaining.discard(pid)
                if remaining:
                    time.sleep(0.002)
            if remaining:
                timed_out = len(remaining)
                terminate_processes({pid: processes[pid] for pid in remaining})
                return_codes.extend([124] * timed_out)
                now = (time.monotonic_ns() - start_ns) / 1e9
                finish_times.extend([now] * timed_out)
            error_file.flush()
            if any(code != 0 for code in return_codes):
                error_file.seek(0)
                error_text = error_file.read().decode("utf-8", errors="replace")[-1000:].strip()
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)

    return summarize_repetition(
        spec,
        workers,
        media_duration_sec,
        repetition,
        finish_times,
        return_codes,
        video_fps=video_fps,
        timed_out=timed_out,
        error_text=error_text,
    )


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def repetition_rate(spec: ScenarioSpec, run: RepetitionResult) -> float:
    value = run.aggregate_fps if spec.media_type == "video" else run.audio_seconds_per_sec
    return float(value or 0.0)


def repetition_media_rate(
    spec: ScenarioSpec, run: RepetitionResult, *, video_fps: float = 0.0
) -> float:
    """Processed source seconds per wall second, used only to size later runs."""
    rate = repetition_rate(spec, run)
    return rate / video_fps if spec.media_type == "video" and video_fps > 0 else rate


def step_rates(spec: ScenarioSpec, step: ScalingStep) -> list[float]:
    return [
        repetition_rate(spec, run)
        for run in step.repetitions
        if run.failed_workers == 0 and repetition_rate(spec, run) > 0
    ]


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if len(values) > 1 and mean > 0 else 0.0


def run_scaling_step(
    spec: ScenarioSpec,
    input_paths: list[Path],
    workers: int,
    media_duration_sec: float,
    gpus: list[GpuInfo],
    host: HostProfile,
    repetitions: int,
    deadline_monotonic: float,
    *,
    video_fps: float = 0.0,
) -> ScalingStep:
    runs: list[RepetitionResult] = []
    for repetition in range(1, repetitions + 1):
        run = run_group_once(
            spec,
            input_paths,
            workers,
            media_duration_sec,
            gpus,
            host,
            repetition,
            deadline_monotonic,
            video_fps=video_fps,
        )
        runs.append(run)
        if run.failed_workers:
            break
    while len(runs) < MAX_REPETITIONS and all(run.failed_workers == 0 for run in runs):
        observed = [repetition_rate(spec, run) for run in runs]
        if coefficient_of_variation(observed) <= CV_THRESHOLD:
            break
        runs.append(
            run_group_once(
                spec,
                input_paths,
                workers,
                media_duration_sec,
                gpus,
                host,
                len(runs) + 1,
                deadline_monotonic,
                video_fps=video_fps,
            )
        )
    return ScalingStep(
        workers=workers,
        media_duration_sec=media_duration_sec,
        failed_workers=max((run.failed_workers for run in runs), default=0),
        repetitions=runs,
    )


def clamp_media_duration(value: float) -> float:
    return round(max(MIN_MEDIA_DURATION_SEC, min(MAX_MEDIA_DURATION_SEC, value)), 3)


def calibrated_duration(
    spec: ScenarioSpec,
    calibration: RepetitionResult,
    base_duration: float,
    *,
    video_fps: float = 0.0,
) -> float:
    rate = repetition_media_rate(spec, calibration, video_fps=video_fps)
    if calibration.failed_workers or rate <= 0:
        return clamp_media_duration(base_duration)
    desired = rate * TARGET_REPETITION_SEC
    return clamp_media_duration(desired)


def calibrate_group_duration(
    spec: ScenarioSpec,
    input_paths: list[Path],
    workers: int,
    candidate_duration: float,
    gpus: list[GpuInfo],
    host: HostProfile,
    deadline_monotonic: float,
    *,
    video_fps: float = 0.0,
    attempts: int = 4,
) -> tuple[float, RepetitionResult]:
    """Pilot a worker count and resize media to the target wall-clock duration.

    When a pilot times out, reduce its media duration and retry. Timed-out
    pilots are calibration only and are not emitted as benchmark measurements.
    """
    duration = clamp_media_duration(candidate_duration)
    last: RepetitionResult | None = None
    for attempt in range(attempts):
        last = run_group_once(
            spec,
            input_paths,
            workers,
            duration,
            gpus,
            host,
            -(attempt + 1),
            deadline_monotonic,
            video_fps=video_fps,
        )
        if last.failed_workers == 0 and last.wall_time_sec > 0:
            return clamp_media_duration(
                duration * TARGET_REPETITION_SEC / last.wall_time_sec
            ), last
        if duration <= MIN_MEDIA_DURATION_SEC:
            break
        observed_wall = max(last.wall_time_sec, TARGET_REPETITION_SEC)
        shrink = min(0.5, TARGET_REPETITION_SEC / observed_wall)
        duration = clamp_media_duration(duration * max(0.1, shrink))
    assert last is not None
    return duration, last


def skipped_result(
    spec: ScenarioSpec, reason: str, *, video: VideoFixture | None = None
) -> ScenarioResult:
    resolution, source_fps = scenario_video_fields(spec, video)
    return ScenarioResult(
        name=spec.name,
        backend=spec.backend,
        operation=spec.operation,
        media_type=spec.media_type,
        codec=spec.codec,
        resolution=resolution,
        source_fps=source_fps,
        threads_per_worker=1,
        bitrate_kbps=spec.bitrate_kbps,
        compression_level=spec.compression_level,
        sample_rate_hz=spec.sample_rate_hz if spec.media_type == "audio" else None,
        channels=spec.channels if spec.media_type == "audio" else None,
        skipped=True,
        skip_reason=reason,
    )


def step_median_rate(spec: ScenarioSpec, step: ScalingStep) -> float:
    rates = step_rates(spec, step)
    return float(median(rates)) if rates else 0.0


def maybe_next_workers(
    spec: ScenarioSpec,
    steps: list[ScalingStep],
    *,
    maximum: int,
    anchor: int,
) -> int | None:
    """Grow the pool only while aggregate throughput is still rising.

    GPU: double from the last successful point (1 → G → 2G → 4G …).
    CPU: after the physical-core anchor, try SMT (2·P) once if P still beat P/2.
    """
    if os.environ.get("FFMPEG_BENCH_WORKERS"):
        return None
    if not steps or steps[-1].failed_workers or maximum <= 1:
        return None
    last = steps[-1]
    last_rate = step_median_rate(spec, last)
    if last_rate <= 0:
        return None
    if spec.backend == "gpu":
        nxt = last.workers * 2
        if nxt > maximum or nxt == last.workers:
            return None
        if len(steps) >= 2:
            prev_rate = step_median_rate(spec, steps[-2])
            if prev_rate > 0 and last_rate < prev_rate:
                return None
        return nxt
    if last.workers != anchor:
        return None
    if len(steps) < 2:
        return None
    prev_rate = step_median_rate(spec, steps[-2])
    if prev_rate <= 0 or last_rate < prev_rate * SCALE_CONTINUE_RATIO:
        return None
    nxt = min(maximum, last.workers * 2)
    return nxt if nxt > last.workers else None


def run_scenario(
    spec: ScenarioSpec,
    input_paths: list[Path],
    host: HostProfile,
    gpus: list[GpuInfo],
    deadline_monotonic: float,
    *,
    video: VideoFixture | None = None,
) -> ScenarioResult:
    target, maximum = max_workers_for_host(host, spec, len(gpus))
    ladder = worker_ladder(target, maximum, backend=spec.backend)
    pending = list(ladder)
    seen: set[int] = set()
    video_fps = video_fps_for(spec, video)
    resolution, source_fps = scenario_video_fields(spec, video)
    base_duration = (
        AUDIO_CALIBRATION_DURATION_SEC
        if spec.media_type == "audio"
        else VIDEO_CALIBRATION_DURATION_SEC
    )
    logger.info("calibrating scenario=%s", spec.name)
    calibration = run_group_once(
        spec, input_paths, 1, base_duration, gpus, host, 0, deadline_monotonic,
        video_fps=video_fps,
    )
    if calibration.failed_workers or repetition_media_rate(spec, calibration, video_fps=video_fps) <= 0:
        return skipped_result(spec, calibration.error or "runtime_probe_failed", video=video)
    candidate_duration = calibrated_duration(
        spec, calibration, base_duration, video_fps=video_fps,
    )
    logger.info(
        "scenario=%s initial_media_duration=%.3fs source_rate=%.2fx ladder=%s",
        spec.name,
        candidate_duration,
        repetition_media_rate(spec, calibration, video_fps=video_fps),
        ladder,
    )
    steps: list[ScalingStep] = []
    while pending:
        workers = pending.pop(0)
        if workers in seen:
            continue
        seen.add(workers)
        if time.monotonic() >= deadline_monotonic:
            logger.warning("overall timeout reached during %s", spec.name)
            break
        media_duration, pilot = calibrate_group_duration(
            spec,
            input_paths,
            workers,
            candidate_duration,
            gpus,
            host,
            deadline_monotonic,
            video_fps=video_fps,
        )
        if pilot.failed_workers:
            logger.warning(
                "scenario=%s workers=%d calibration failed at minimum media duration %.3fs",
                spec.name,
                workers,
                media_duration,
            )
            pilot.repetition = 1
            steps.append(
                ScalingStep(
                    workers=workers,
                    media_duration_sec=media_duration,
                    failed_workers=pilot.failed_workers,
                    repetitions=[pilot],
                )
            )
            break
        logger.info(
            "scenario=%s workers=%d media_duration=%.3fs repetitions=%d",
            spec.name,
            workers,
            media_duration,
            REPETITIONS,
        )
        step = run_scaling_step(
            spec,
            input_paths,
            workers,
            media_duration,
            gpus,
            host,
            REPETITIONS,
            deadline_monotonic,
            video_fps=video_fps,
        )
        steps.append(step)
        rates = step_rates(spec, step)
        logger.info(
            "scenario=%s workers=%d rate=%.2f %s cv=%.3f failures=%d",
            spec.name,
            workers,
            median(rates),
            "fps" if spec.media_type == "video" else "audio-seconds/sec",
            coefficient_of_variation(rates),
            step.failed_workers,
        )
        if step.failed_workers:
            break
        media_rates = [
            repetition_media_rate(spec, run, video_fps=video_fps)
            for run in step.repetitions
            if run.failed_workers == 0
        ]
        if media_rates:
            candidate_duration = clamp_media_duration(
                median(media_rates) / workers * TARGET_REPETITION_SEC
            )
        if not pending:
            nxt = maybe_next_workers(spec, steps, maximum=maximum, anchor=target)
            if nxt is not None and nxt not in seen:
                pending.append(nxt)
                logger.info("scenario=%s probing workers=%d", spec.name, nxt)
    return ScenarioResult(
        name=spec.name,
        backend=spec.backend,
        operation=spec.operation,
        media_type=spec.media_type,
        codec=spec.codec,
        resolution=resolution,
        source_fps=source_fps,
        threads_per_worker=1,
        vbench_scenario=spec.vbench_scenario,
        bitrate_kbps=spec.bitrate_kbps,
        compression_level=spec.compression_level,
        sample_rate_hz=spec.sample_rate_hz if spec.media_type == "audio" else None,
        channels=spec.channels if spec.media_type == "audio" else None,
        scaling_steps=steps,
    )


def rounded_dict(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [rounded_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded_dict(item) for key, item in value.items()}
    return value


def scenario_to_dict(result: ScenarioResult) -> dict[str, Any]:
    return rounded_dict(asdict(result))


def host_to_dict(host: HostProfile, cpu_stat_after: dict[str, int]) -> dict[str, Any]:
    return {
        "cpu_count": host.cpu_count,
        "physical_cpu_count": host.physical_cpu_count,
        "affinity_cpu_count": host.affinity_cpu_count,
        "affinity_cpus": host.affinity_cpus,
        "cpu_quota_cores": host.cpu_quota_cores,
        "cpu_capacity": host.cpu_capacity,
        "architecture": host.architecture,
        "ram_total_gb": host.ram_total_gb,
        "ram_avail_gb": host.ram_avail_gb,
        "memory_limit_gb": host.memory_limit_gb,
        "pids_limit": host.pids_limit,
        "pids_current": host.pids_current,
        "cgroup_cpu_max": host.cgroup_cpu_max,
        "cgroup_cpuset_effective": host.cgroup_cpuset_effective,
        "cgroup_cpu_stat_before": host.cgroup_cpu_stat_before,
        "cgroup_cpu_stat_after": cpu_stat_after,
        "logical_cpu_count": host.logical_cpu_count,
        "numa_binding": host.numa_binding,
        "numa_nodes": [asdict(node) for node in host.numa_nodes],
        "gpus": [asdict(gpu) for gpu in host.gpus],
    }


def emit_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_benchmark() -> dict[str, Any]:
    started = time.time()
    started_monotonic = time.monotonic()
    host = host_profile()
    gpus = list(host.gpus)
    encoders, decoders = ffmpeg_inventory()
    video_fixtures = {
        "h264": load_video_fixture(FIXTURE_VIDEO_PATH, expected_sha256=FIXTURE_VIDEO_SHA256),
        "hevc": load_video_fixture(
            FIXTURE_VIDEO_HEVC_PATH, expected_sha256=FIXTURE_VIDEO_HEVC_SHA256
        ),
    }
    video_h264 = video_fixtures["h264"]
    video_hevc = video_fixtures["hevc"]
    deadline_monotonic = started_monotonic + OVERALL_TIMEOUT_SEC

    with make_work_dir() as tmp:
        work_dir = Path(tmp)
        staged_paths = {
            key: stage_video_source(fixture.path, work_dir)
            for key, fixture in video_fixtures.items()
        }
        video_probe = {
            key: probe_source(path, "v:0") for key, path in staged_paths.items()
        }
        audio_paths = stage_audio_sources(FIXTURE_PATH, work_dir, host)
        audio_probe = probe_source(audio_paths[0], "a:0")
        scenario_results: list[ScenarioResult] = []
        for spec in SCENARIOS:
            video = (
                video_fixture_for(spec, video_fixtures)
                if spec.media_type == "video"
                else video_h264
            )
            if time.monotonic() >= deadline_monotonic:
                scenario_results.append(
                    skipped_result(spec, "overall_timeout", video=video)
                )
                continue
            supported, reason = scenario_supported(spec, encoders, decoders, gpus)
            if not supported:
                logger.info("skipping %s: %s", spec.name, reason)
                scenario_results.append(skipped_result(spec, reason, video=video))
                continue
            if spec.media_type == "audio":
                inputs = audio_paths
            else:
                inputs = [staged_paths[spec.source_codec]]
            scenario_results.append(
                run_scenario(
                    spec, inputs, host, gpus, deadline_monotonic, video=video,
                )
            )

    finished = time.time()
    video_source = {
        "path": "Tears of Steel excerpt",
        "title": video_h264.title,
        "license": "Creative Commons Attribution 3.0",
        "license_url": video_h264.license_url,
        "source_url": f"{CDN_FIXTURE_BASE}/source.mp4",
        "hevc_source_url": f"{CDN_FIXTURE_BASE}/source-hevc.mp4",
        "upstream_url": "http://ftp.halifax.rwth-aachen.de/blender/demo/movies/ToS/tears_of_steel_1080p.mov",
        "sha256": video_h264.sha256,
        "hevc_sha256": video_hevc.sha256,
        "resolution": f"{video_h264.width}x{video_h264.height}",
        "fps": video_h264.fps,
        "duration_sec": video_h264.duration_sec,
        "codec": video_h264.codec,
        "hevc_codec": video_hevc.codec,
        "probe": video_probe["h264"],
        "hevc_probe": video_probe["hevc"],
    }
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "started_at_unix": round(started, 3),
        "finished_at_unix": round(finished, 3),
        "duration_sec": round(finished - started, 3),
        "host": host_to_dict(host, cgroup_cpu_stat()),
        "ffmpeg": {
            "version": ffmpeg_version(),
            "binary": ffmpeg_bin(),
            "buildconf": ffmpeg_buildconf(),
            "encoders_available": sorted(encoders),
            "decoders_available": sorted(decoders),
            "audio_gpu_acceleration": False,
            "audio_gpu_reason": "No FFmpeg hardware encoder exists for Vorbis or FLAC",
        },
        "sources": {
            "video": video_source,
            "audio": {
                "path": "CC0 Entre dos Aguas music sample",
                "license": "CC0-1.0",
                "source_url": f"{CDN_FIXTURE_BASE}/source.flac",
                "sha256": FIXTURE_AUDIO_SHA256,
                "sample_rate_hz": AUDIO_SAMPLE_RATE,
                "channels": AUDIO_CHANNELS,
                "probe": audio_probe,
            },
        },
        # Preserve the v1 singular source object for downstream consumers.
        "source": video_source,
        "methodology": {
            "concurrency_model": "synchronized_parallel_ffmpeg_processes",
            "aggregate_clock": "time.monotonic_ns",
            "aggregate_formula": "successful_workers * media_duration / group_makespan",
            "threads_per_worker": 1,
            "target_repetition_sec": TARGET_REPETITION_SEC,
            "repetition_timeout_sec": REPETITION_TIMEOUT_SEC,
            "repetitions": REPETITIONS,
            "max_repetitions": MAX_REPETITIONS,
            "cv_threshold": CV_THRESHOLD,
            "worker_sweep": "physical_cores_then_adaptive_doubling",
            "duration_scaling": "per_worker_count_pilot_to_target_wall_time",
            "output_muxer": "null",
            "audio_profiles": [
                "Ogg Vorbis 24/96/160/320 kbps",
                "FLAC lossless compression level 5",
            ],
            "audio_gpu_acceleration": "not_available",
            "metrics": [
                "wall_time_sec",
                "processed_frames",
                "aggregate_fps",
                "processed_audio_seconds",
                "audio_seconds_per_sec",
            ],
            "pts_ffmpeg_reference": {
                "test_profile": "pts/ffmpeg",
                "workload": "vbench upload (libx264/libx265, crf 18, threads 1)",
                "comparable_field": "single_stream_fps",
                "intentional_differences": [
                    "Tears of Steel excerpt vs full vbench corpus",
                    "synchronized capacity sweep vs serial transcode",
                    "custom ffmpeg build vs PTS static build",
                    "audio and NVIDIA scenarios are benchmark extensions",
                ],
            },
        },
        "scenarios": [scenario_to_dict(result) for result in scenario_results],
    }


def main() -> int:
    parser = ArgumentParser(description="FFmpeg aggregate transcoding benchmark")
    parser.add_argument("--version", action="store_true", help="Print versions and exit")
    args = parser.parse_args()
    if args.version:
        print(f"benchmark-ffmpeg {BENCHMARK_VERSION} ffmpeg={ffmpeg_version()}")
        return 0
    if shutil.which(ffmpeg_bin()) is None:
        logger.error("ffmpeg not found in PATH")
        return 1
    try:
        payload = run_benchmark()
    except Exception as exc:  # pragma: no cover - surfaced in container logs
        logger.exception("benchmark failed: %s", exc)
        emit_json({"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION, "error": str(exc)})
        return 1
    emit_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
