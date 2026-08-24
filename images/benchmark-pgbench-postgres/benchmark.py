#!/usr/bin/env python3
"""pgbench driver for Spare Cores inspector.

Workloads:
  * ``pgbench_ro`` — cached CPU-heavy ``ro_cpu_*`` script (``-f`` + ``-D scale``).
    Fixed concurrency ``{1, V/2, V, 2·V}``; no geometric ladder / adaptive search.
    Headline score is TPM only (txn/min); per-second TPS is omitted.
  * ``pgbench_tpcb`` — built-in tpcb-like; geometric anchors + upward search.

Env (set by postgres_multi / postgres_dbaas):
  SC_WORKLOAD=pgbench_ro|pgbench_tpcb
  SC_CPU_SCALE — RO work multiplier (``-D scale=N``), default 1
  SC_SCALEFACTOR / SC_SCALEFACTORS — TPC-B pgbench ``-s`` values
  SC_PROFILE_VUS — concurrency list (RO: 1,V/2,V,2V; TPC-B: anchors)
  SC_PROFILE_SEARCH — TPC-B only (RO forces off)
  SC_PROFILE_IMPROVE_PCT / SC_PROFILE_MAX_CLIENTS / SC_PROFILE_HARD_MAX_CLIENTS
  SC_WARMUP_ONCE / SC_WARMUP_SECONDS / SC_SETTLE_SECONDS / SC_RUN_SECONDS
  SC_DB_* — connection; SC_CDN_* — dump cache
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from db_dataset_cache import (
    dataset_spec_for_pgbench,
    dataset_spec_for_pgbench_ro_cpu,
    prepare_database,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RO_SETUP_SQL = SCRIPT_DIR / "ro_cpu_setup.sql"
RO_TXN_SQL = SCRIPT_DIR / "ro_cpu_txn.sql"
RO_MAX_JOBS = 32

# Keep in sync with sc-inspector/inspector/benchmark_tiers.py (TPC-B only).
GEOMETRIC_CONCURRENCY_LADDER: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
)

_RE_TPS = re.compile(
    r"^tps\s*=\s*([0-9.]+)\s*\(without initial connection time\)", re.M
)
_RE_LAT_AVG = re.compile(r"^latency average\s*=\s*([0-9.]+)\s*ms", re.M)
_RE_LAT_STD = re.compile(r"^latency stddev\s*=\s*([0-9.]+)\s*ms", re.M)
_RE_TX = re.compile(r"^number of transactions actually processed:\s*([0-9]+)", re.M)
_RE_FAIL = re.compile(r"^number of failed transactions:\s*([0-9]+)", re.M)
_RE_CONN = re.compile(r"^initial connection time\s*=\s*([0-9.]+)\s*ms", re.M)

LATENCY_SAMPLE_RATE = 0.01


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def parse_csv_ints(name: str) -> list[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [max(1, int(x.strip())) for x in raw.split(",") if x.strip()]


def run(cmd: list[str], *, timeout: int, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out[-4000:]}")
    return out


def pg_env(password: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    return env


# ---------------------------------------------------------------------------
# Standalone mode: when SC_DB_HOST is unset, this same container starts its
# own local Postgres server (this image is FROM postgres:18, so the server
# binaries are already present) instead of connecting to a companion VM.
# A separate client VM was pure waste here — pgbench barely uses any CPU
# while it stresses the server (observed <1% CPU, ~10 MB RSS on an 8-vCPU
# benchmark), so running both roles on one instance is strictly better.
# ---------------------------------------------------------------------------

STANDALONE_PG_LOG = Path("/tmp/pg-server.log")


def local_mem_gib() -> float:
    """Total system memory in GiB, read directly (no psutil dependency)."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def max_connections_for_vcpus(vcpus: int) -> int:
    """Postgres max_connections floor; vcpus kept for API symmetry.

    Keep in sync with sc-inspector/inspector/benchmark_tiers.py:max_connections_for_vcpus.
    """
    _ = vcpus
    return GEOMETRIC_CONCURRENCY_LADDER[-1] + 50


def pg_guc_settings(
    mem_gib: float,
    durability: str = "durable",
    *,
    vcpus: int | None = None,
    min_max_connections: int | None = None,
) -> dict[str, str]:
    """pgtune-derived GUCs for the local server.

    Keep in sync with sc-inspector/inspector/postgres_multi.py:pg_guc_settings.
    """
    from pgtune_leopard import generate_for_host

    cpu = max(1, int(vcpus if vcpus is not None else os.cpu_count() or 4))
    result = generate_for_host(mem_gib=mem_gib, cpu_num=cpu)
    settings = dict(result.settings)
    settings["synchronous_commit"] = "off" if durability == "async" else "on"
    need = max_connections_for_vcpus(cpu)
    if min_max_connections is not None:
        need = max(need, int(min_max_connections))
    cur = int(str(settings.get("max_connections", "100")).split()[0])
    if need > cur:
        settings["max_connections"] = str(need)
    return settings


def pg_gucs(mem_gib: float, durability: str = "durable", *, vcpus: int | None = None) -> list[str]:
    args: list[str] = []
    for name, value in pg_guc_settings(mem_gib, durability, vcpus=vcpus).items():
        if name == "listen_addresses":
            continue  # set explicitly by the caller
        args.extend(["-c", f"{name}={value}"])
    return args


def start_local_postgres(
    mem_gib: float, vcpus: int, durability: str, password: str
) -> subprocess.Popen:
    """Launch a local Postgres server; docker-entrypoint.sh handles initdb/auth."""
    cmd = [
        "docker-entrypoint.sh",
        "postgres",
        "-c",
        "listen_addresses=*",
        *pg_gucs(mem_gib, durability, vcpus=vcpus),
    ]
    env = os.environ.copy()
    env["POSTGRES_PASSWORD"] = password
    env["POSTGRES_USER"] = "postgres"
    log = open(STANDALONE_PG_LOG, "wb")  # real file, not a PIPE: avoids buffer-fill deadlock
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_local_postgres_ready(password: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    last_err: str = "timed out"
    while time.monotonic() < deadline:
        try:
            run(
                ["psql", "-h", "127.0.0.1", "-p", "5432", "-U", "postgres", "-d", "postgres", "-tAc", "SELECT 1"],
                timeout=5,
                env=pg_env(password),
            )
            return
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1)
    raise TimeoutError(f"local postgres did not become ready: {last_err}")


def stop_local_postgres(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        proc.kill()


def parse_pgbench_summary(text: str) -> dict[str, Any]:
    """Parse pgbench stdout. Score is TPM only (no TPS field)."""
    out: dict[str, Any] = {}
    m = _RE_TPS.search(text)
    if m:
        tps = float(m.group(1))
        out["tpm"] = int(round(tps * 60))
        out["score"] = out["tpm"]
    m = _RE_LAT_AVG.search(text)
    if m:
        out["latency_avg_ms"] = round(float(m.group(1)), 4)
    m = _RE_LAT_STD.search(text)
    if m:
        out["latency_stddev_ms"] = round(float(m.group(1)), 4)
    m = _RE_TX.search(text)
    if m:
        out["tx_processed"] = int(m.group(1))
    m = _RE_FAIL.search(text)
    if m:
        out["tx_failed"] = int(m.group(1))
    m = _RE_CONN.search(text)
    if m:
        out["initial_connection_ms"] = round(float(m.group(1)), 3)
    return out


def percentiles_from_logs(work_dir: Path) -> dict[str, Any]:
    samples: list[float] = []
    for path in sorted(work_dir.glob("pgbench_log*")):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        samples.append(float(parts[2]) / 1000.0)
                    except ValueError:
                        continue
        except OSError:
            continue
    if not samples:
        return {}
    samples.sort()

    def pct(p: float) -> float:
        if len(samples) == 1:
            return samples[0]
        k = (len(samples) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(samples) - 1)
        if f == c:
            return samples[f]
        return samples[f] + (samples[c] - samples[f]) * (k - f)

    return {
        "p50": round(pct(50), 4),
        "p95": round(pct(95), 4),
        "p99": round(pct(99), 4),
        "avg": round(statistics.fmean(samples), 4),
        "samples": len(samples),
    }


def psql_file(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    sql_path: Path,
    timeout: int = 14400,
) -> None:
    run(
        [
            "psql",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            dbname,
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            str(sql_path),
        ],
        timeout=timeout,
        env=pg_env(password),
    )


def apply_ro_session_defaults(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
) -> None:
    """One-time GUCs so the txn script does not pay per-txn SET round-trips."""
    stmts = [
        f'ALTER DATABASE "{dbname}" SET jit = off',
        f"ALTER DATABASE \"{dbname}\" SET work_mem = '64MB'",
        f'ALTER DATABASE "{dbname}" SET max_parallel_workers_per_gather = 0',
    ]
    for stmt in stmts:
        run(
            [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-d",
                dbname,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                stmt,
            ],
            timeout=60,
            env=pg_env(password),
        )


def pgbench_init_tpcb(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    scale: int,
) -> None:
    run(
        [
            "pgbench",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-i",
            "-s",
            str(scale),
            dbname,
        ],
        timeout=14400,
        env=pg_env(password),
    )


def pgbench_init_ro_cpu(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
) -> None:
    if not RO_SETUP_SQL.is_file():
        raise RuntimeError(f"missing RO setup script: {RO_SETUP_SQL}")
    psql_file(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        sql_path=RO_SETUP_SQL,
    )
    apply_ro_session_defaults(
        host=host, port=port, user=user, password=password, dbname=dbname
    )


def pgbench_run(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    workload: str,
    clients: int,
    seconds: int,
    progress: bool,
    latency_log: bool,
    work_dir: Path,
    cpu_scale: int = 1,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if workload == "pgbench_ro":
        jobs = min(clients, RO_MAX_JOBS)
    else:
        jobs = clients
    args = [
        "pgbench",
        "-h",
        host,
        "-p",
        str(port),
        "-U",
        user,
        "-c",
        str(clients),
        "-j",
        str(jobs),
        "-T",
        str(seconds),
        "-M",
        "prepared",
        "-n",
    ]
    if workload == "pgbench_ro":
        if not RO_TXN_SQL.is_file():
            raise RuntimeError(f"missing RO txn script: {RO_TXN_SQL}")
        args.extend(["-D", f"scale={cpu_scale}", "-f", str(RO_TXN_SQL)])
    elif workload == "pgbench_tpcb":
        args.extend(["-b", "tpcb-like"])
    else:
        raise RuntimeError(f"unsupported workload {workload!r}")
    if progress:
        args.extend(["-P", "5", "--progress-timestamp"])
    if latency_log:
        args.extend(
            [
                "-l",
                f"--log-prefix={work_dir / 'pgbench_log'}",
                f"--sampling-rate={LATENCY_SAMPLE_RATE}",
            ]
        )
    args.append(dbname)
    t0 = time.time()
    try:
        out = run(args, timeout=seconds + 600, env=pg_env(password))
    except RuntimeError as exc:
        text = str(exc)
        parsed = parse_pgbench_summary(text)
        if not parsed.get("tpm"):
            raise
        out = text
    elapsed = round(time.time() - t0, 1)
    parsed = parse_pgbench_summary(out)
    parsed["run_seconds"] = elapsed
    parsed["jobs"] = jobs
    if latency_log:
        pct = percentiles_from_logs(work_dir)
        if pct:
            parsed["latency_ms"] = {
                "p50": pct["p50"],
                "p95": pct["p95"],
                "p99": pct["p99"],
                "avg": pct.get("avg", parsed.get("latency_avg_ms")),
                "samples": pct["samples"],
            }
            if "latency_avg_ms" in parsed and "avg" not in pct:
                parsed["latency_ms"]["avg"] = parsed["latency_avg_ms"]
    return parsed


def rung(x: float) -> int:
    target = max(1.0, float(x))
    return min(GEOMETRIC_CONCURRENCY_LADDER, key=lambda v: (abs(v - target), v))


def concurrency_profile_ro(vcpus: int) -> list[int]:
    v = max(1, int(vcpus))
    return sorted({1, max(1, v // 2), v, 2 * v})


def concurrency_anchors_tpcb(vcpus: int) -> list[int]:
    v = max(1, int(vcpus))
    return sorted({1, rung(v / 4), rung(v / 2), rung(v)})


def choose_concurrency_plan(
    *,
    anchors: list[int],
    search: bool,
    max_clients: int,
) -> list[int]:
    base = sorted({max(1, int(c)) for c in anchors if int(c) <= max_clients}) or [1]
    if not search:
        return base
    start = max(base)
    tail = [c for c in GEOMETRIC_CONCURRENCY_LADDER if start < c <= max_clients]
    return base + tail


def next_ladder_rung(current: int, cap: int) -> int | None:
    for candidate in GEOMETRIC_CONCURRENCY_LADDER:
        if candidate > current and candidate <= cap:
            return candidate
    return None


def plan_for_scale(
    *,
    workload: str,
    scale: int,
    host_anchors: list[int],
    search: bool,
    host_max_clients: int,
    db_vcpus: int,
) -> tuple[list[int], list[int], int]:
    if workload == "pgbench_ro":
        # Fixed profile; ignore search.
        anchors = [c for c in host_anchors if c <= host_max_clients] or [1]
        return anchors, list(anchors), host_max_clients

    max_clients = max(1, min(int(scale), int(host_max_clients)))
    anchor_v = max(1, min(int(db_vcpus), int(scale)))
    anchors = concurrency_anchors_tpcb(anchor_v)
    plan = choose_concurrency_plan(
        anchors=anchors, search=search, max_clients=max_clients
    )
    return anchors, plan, max_clients


def expand_scales(workload: str) -> list[int]:
    if workload == "pgbench_ro":
        # One fixed schema; cpu_scale is separate.
        return [0]
    many = parse_csv_ints("SC_SCALEFACTORS")
    if many:
        return many
    return [env_int("SC_SCALEFACTOR", 65)]


def show_synchronous_commit(host: str, port: int, user: str, password: str, dbname: str) -> str:
    out = run(
        [
            "psql",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            dbname,
            "-Atc",
            "SHOW synchronous_commit;",
        ],
        timeout=60,
        env=pg_env(password),
    )
    return out.strip()


def show_max_connections(host: str, port: int, user: str, password: str, dbname: str) -> int:
    out = run(
        [
            "psql",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            dbname,
            "-Atc",
            "SHOW max_connections;",
        ],
        timeout=60,
        env=pg_env(password),
    )
    return int(out.strip())


MAX_CONNECTIONS_CLIENT_RESERVE = 50


def ensure_db(host: str, port: int, user: str, password: str, admin_db: str, dbname: str) -> None:
    try:
        run(
            [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-d",
                admin_db,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                f'CREATE DATABASE "{dbname}";',
            ],
            timeout=120,
            env=pg_env(password),
        )
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower():
            raise


def run_size(
    *,
    workload: str,
    scale: int,
    cpu_scale: int,
    host: str,
    port: int,
    user: str,
    password: str,
    admin_db: str,
    dbname: str,
    plan: list[int],
    anchors: set[int],
    improve_pct: float,
    hard_max_clients: int,
    run_seconds: int,
    warmup_seconds: int,
    settle_seconds: int,
    warmup_once: bool,
    warmup_done: bool,
    search: bool,
) -> tuple[dict[str, Any], bool]:
    ensure_db(host, port, user, password, admin_db, dbname)
    if workload == "pgbench_ro":
        spec = dataset_spec_for_pgbench_ro_cpu()
        build = lambda: pgbench_init_ro_cpu(
            host=host, port=port, user=user, password=password, dbname=dbname
        )
    else:
        spec = dataset_spec_for_pgbench(scalefactor=scale)
        build = lambda: pgbench_init_tpcb(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            scale=scale,
        )
    dataset_meta = prepare_database(
        spec,
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        admin_db=admin_db,
        build=build,
    )
    if workload == "pgbench_ro":
        # Restores from CDN skip setup's ALTER DATABASE; re-apply cheaply.
        apply_ro_session_defaults(
            host=host, port=port, user=user, password=password, dbname=dbname
        )

    profile: list[dict[str, Any]] = []
    peak_tpm = 0.0
    stop_reason = ""
    plan_dyn = list(plan)
    with tempfile.TemporaryDirectory(prefix="pgbench-") as tmp:
        tmp_path = Path(tmp)
        i = 0
        while i < len(plan_dyn):
            clients = plan_dyn[i]
            is_anchor = clients in anchors
            need_warmup = not (warmup_once and warmup_done)
            w_secs = warmup_seconds if need_warmup else settle_seconds
            if w_secs > 0:
                pgbench_run(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                    workload=workload,
                    clients=clients,
                    seconds=w_secs,
                    progress=False,
                    latency_log=False,
                    work_dir=tmp_path / f"warm_{clients}",
                    cpu_scale=cpu_scale,
                )
                warmup_done = True

            measure = pgbench_run(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=dbname,
                workload=workload,
                clients=clients,
                seconds=run_seconds,
                progress=True,
                latency_log=True,
                work_dir=tmp_path / f"meas_{clients}",
                cpu_scale=cpu_scale,
            )
            tpm = float(measure.get("tpm") or 0)
            prev_peak = peak_tpm
            if tpm > peak_tpm:
                peak_tpm = tpm
            entry = {
                "concurrency": clients,
                "jobs": measure.get("jobs", clients),
                "anchor": is_anchor,
                "warmup_seconds": w_secs,
                **{k: v for k, v in measure.items() if k != "jobs"},
            }
            profile.append(entry)

            if workload == "pgbench_tpcb" and search:
                if (
                    not is_anchor
                    and prev_peak > 0
                    and tpm < prev_peak * (1.0 + improve_pct / 100.0)
                ):
                    stop_reason = (
                        f"tpm {tpm:.0f} did not improve peak {prev_peak:.0f} "
                        f"by >={improve_pct:g}%"
                    )
                    entry["stop_reason"] = stop_reason
                    break
                if (
                    i == len(plan_dyn) - 1
                    and not is_anchor
                    and prev_peak > 0
                    and tpm >= prev_peak * (1.0 + improve_pct / 100.0)
                    and clients < hard_max_clients
                ):
                    nxt = next_ladder_rung(clients, hard_max_clients)
                    if nxt is not None:
                        plan_dyn.append(nxt)
            i += 1

    best = max(profile, key=lambda r: float(r.get("tpm") or 0)) if profile else {}
    final_peak = float(best.get("tpm") or 0)
    for entry in profile:
        tpm = float(entry.get("tpm") or 0)
        entry["tpm_vs_final_peak_pct"] = (
            round(100.0 * tpm / final_peak, 2) if final_peak > 0 else 100.0
        )
    size_out: dict[str, Any] = {
        "dataset": dataset_meta,
        "profile": profile,
        "profile_vus": sorted(anchors),
        "concurrency_plan": plan_dyn,
        "profile_max_clients": max(plan) if plan else 1,
        "peak_concurrency": best.get("concurrency"),
        "score": best.get("tpm") or best.get("score") or 0,
        "latency_ms": best.get("latency_ms"),
        "latency_avg_ms": best.get("latency_avg_ms"),
        "latency_stddev_ms": best.get("latency_stddev_ms"),
        "stop_reason": stop_reason,
    }
    if workload == "pgbench_ro":
        size_out["cpu_scale"] = cpu_scale
    else:
        size_out["scalefactor"] = scale
    return size_out, warmup_done


def main() -> int:
    workload = os.environ.get("SC_WORKLOAD", "pgbench_ro").strip().lower()
    if workload not in {"pgbench_ro", "pgbench_tpcb"}:
        raise RuntimeError(f"unsupported SC_WORKLOAD={workload!r}")

    host = os.environ.get("SC_DB_HOST", "").strip()
    port = env_int("SC_DB_PORT", 5432)
    user = os.environ.get("SC_DB_USER", "postgres")
    password = os.environ.get("SC_DB_PASSWORD", "postgres")
    admin_db = os.environ.get("SC_DB_NAME", "postgres")
    dbname = os.environ.get("SC_PGBENCH_DB", "pgbench")

    # Standalone: no companion VM configured, run Postgres locally instead.
    pg_proc = None
    if not host:
        durability = os.environ.get("SC_DURABILITY", "durable")
        vcpus = env_int("SC_DB_VCPUS", os.cpu_count() or 2)
        mem_gib = local_mem_gib()
        pg_proc = start_local_postgres(mem_gib, vcpus, durability, password)
        wait_local_postgres_ready(password)
        host = "127.0.0.1"
        os.environ.setdefault("SC_DB_MEM_GIB", str(mem_gib))
        os.environ.setdefault("SC_TOPOLOGY", "single_vm")

    run_seconds = env_int("SC_RUN_SECONDS", 300)
    warmup_seconds = env_int("SC_WARMUP_SECONDS", 120)
    settle_seconds = env_int("SC_SETTLE_SECONDS", 60)
    warmup_once = env_bool("SC_WARMUP_ONCE", True)
    improve_pct = env_float("SC_PROFILE_IMPROVE_PCT", 5.0)
    search = env_bool("SC_PROFILE_SEARCH", workload == "pgbench_tpcb")
    if workload == "pgbench_ro":
        search = False
    db_vcpus = env_int("SC_DB_VCPUS", os.cpu_count() or 2)
    cpu_scale = env_int("SC_CPU_SCALE", 1)
    if workload == "pgbench_ro":
        host_anchors = parse_csv_ints("SC_PROFILE_VUS") or concurrency_profile_ro(
            db_vcpus
        )
    else:
        host_anchors = parse_csv_ints("SC_PROFILE_VUS") or concurrency_anchors_tpcb(
            db_vcpus
        )
    host_max_clients = env_int("SC_PROFILE_MAX_CLIENTS", max(host_anchors))
    host_hard_max_clients = env_int(
        "SC_PROFILE_HARD_MAX_CLIENTS",
        max(host_anchors) if workload == "pgbench_ro" else GEOMETRIC_CONCURRENCY_LADDER[-1],
    )
    scales = expand_scales(workload)

    sync_commit = show_synchronous_commit(host, port, user, password, admin_db)
    max_connections = show_max_connections(host, port, user, password, admin_db)
    conn_cap = max(1, max_connections - MAX_CONNECTIONS_CLIENT_RESERVE)
    host_max_clients = min(host_max_clients, conn_cap)
    host_hard_max_clients = min(host_hard_max_clients, conn_cap)
    host_anchors = [c for c in host_anchors if c <= conn_cap] or [1]

    sizes: list[dict[str, Any]] = []
    warmup_done = False
    for scale in scales:
        anchors, plan, scale_max = plan_for_scale(
            workload=workload,
            scale=scale if workload != "pgbench_ro" else 1,
            host_anchors=host_anchors,
            search=search,
            host_max_clients=host_max_clients,
            db_vcpus=db_vcpus,
        )
        hard_max = (
            host_hard_max_clients
            if workload != "pgbench_tpcb"
            else min(host_hard_max_clients, scale)
        )
        if workload == "pgbench_tpcb" and plan and max(plan) > scale:
            raise RuntimeError(
                f"TPC-B plan violates pgbench docs (-s >= -c): "
                f"scale={scale} max_clients={max(plan)}"
            )
        size_result, warmup_done = run_size(
            workload=workload,
            scale=scale,
            cpu_scale=cpu_scale,
            host=host,
            port=port,
            user=user,
            password=password,
            admin_db=admin_db,
            dbname=dbname,
            plan=plan,
            anchors=set(anchors),
            improve_pct=improve_pct,
            hard_max_clients=hard_max,
            run_seconds=run_seconds,
            warmup_seconds=warmup_seconds,
            settle_seconds=settle_seconds,
            warmup_once=warmup_once,
            warmup_done=warmup_done,
            search=search,
        )
        size_result["clients_capped_at_scale"] = workload == "pgbench_tpcb"
        size_result["profile_max_clients"] = scale_max
        sizes.append(size_result)

    best_size = max(sizes, key=lambda s: float(s.get("score") or 0)) if sizes else {}
    summary: dict[str, Any] = {
        "benchmark": "pgbench_postgres",
        "workload": workload,
        "topology": os.environ.get("SC_TOPOLOGY", "multi_vm"),
        "durability": os.environ.get("SC_DURABILITY", "durable"),
        "synchronous_commit": sync_commit,
        "max_connections": max_connections,
        "max_connections_client_cap": conn_cap,
        "run_seconds": run_seconds,
        "warmup_seconds": warmup_seconds,
        "settle_seconds": settle_seconds,
        "warmup_once": warmup_once,
        "improve_pct": improve_pct if workload == "pgbench_tpcb" else None,
        "profile_search": search,
        "profile_vus": host_anchors,
        "profile_max_clients": host_max_clients,
        "profile_hard_max_clients": host_hard_max_clients,
        "db_vcpus": db_vcpus,
        "client_vcpus": env_int("SC_CLIENT_VCPUS", os.cpu_count() or 2),
        "db_mem_gib": env_float("SC_DB_MEM_GIB", 0.0) or None,
        "sizes": sizes,
        "peak_concurrency": best_size.get("peak_concurrency"),
        "score": best_size.get("score") or 0,
        "score_unit": "tpm",
        "latency_ms": best_size.get("latency_ms"),
        "latency_avg_ms": best_size.get("latency_avg_ms"),
        "latency_stddev_ms": best_size.get("latency_stddev_ms"),
    }
    if workload == "pgbench_ro":
        summary["cpu_scale"] = cpu_scale
        summary["peak_cpu_scale"] = best_size.get("cpu_scale")
    else:
        summary["scalefactors"] = scales
        summary["peak_scalefactor"] = best_size.get("scalefactor")
        summary["scalefactor"] = best_size.get("scalefactor")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if pg_proc is not None:
        stop_local_postgres(pg_proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
