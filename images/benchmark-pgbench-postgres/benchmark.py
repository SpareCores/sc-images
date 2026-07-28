#!/usr/bin/env python3
"""pgbench RO (-S) and TPC-B (tpcb-like) driver for Spare Cores inspector.

Env (set by postgres_multi / postgres_dbaas):
  SC_WORKLOAD=pgbench_ro|pgbench_tpcb
  SC_SCALEFACTOR / SC_SCALEFACTORS — one or more pgbench -s values
  SC_PROFILE_VUS — host anchor concurrencies (comma list); TPC-B may shrink
  SC_PROFILE_SEARCH=1 — walk geometric ladder upward while TPM improves ≥5%
  SC_PROFILE_IMPROVE_PCT — default 5
  SC_PROFILE_MAX_CLIENTS — host search cap (inclusive); TPC-B also caps at -s
  SC_PROFILE_HARD_MAX_CLIENTS — adaptive tail hard cap (inclusive)
  SC_WARMUP_ONCE=1 — full warmup only on the first timed rung; then settle
  SC_WARMUP_SECONDS / SC_SETTLE_SECONDS / SC_RUN_SECONDS
  SC_DB_* — connection
  SC_CDN_* — dump cache (prepare_database)

TPC-B (tpcb-like): pgbench docs require -s >= max -c. We keep fixed GiB size
rungs from the inspector and never run more clients than the scale factor.
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

from db_dataset_cache import dataset_spec_for_pgbench, prepare_database

# Keep in sync with sc-inspector/inspector/benchmark_tiers.py
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


def parse_pgbench_summary(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = _RE_TPS.search(text)
    if m:
        tps = float(m.group(1))
        out["tps"] = round(tps, 2)
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


def pgbench_init(
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
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
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
        args.append("-S")
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
        if not parsed.get("tps"):
            raise
        out = text
    elapsed = round(time.time() - t0, 1)
    parsed = parse_pgbench_summary(out)
    parsed["run_seconds"] = elapsed
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
            # Prefer sampled avg when present for the headline latency_ms.avg
            if "latency_avg_ms" in parsed and "avg" not in pct:
                parsed["latency_ms"]["avg"] = parsed["latency_avg_ms"]
    return parsed


def rung(x: float) -> int:
    """Snap a concurrency target onto ``GEOMETRIC_CONCURRENCY_LADDER``."""
    target = max(1.0, float(x))
    return min(GEOMETRIC_CONCURRENCY_LADDER, key=lambda v: (abs(v - target), v))


def concurrency_anchors(vcpus: int) -> list[int]:
    v = max(1, int(vcpus))
    return sorted({1, rung(v / 4), rung(v / 2), rung(v)})


def choose_concurrency_plan(
    *,
    anchors: list[int],
    search: bool,
    max_clients: int,
) -> list[int]:
    """Anchors first (all measured), then optional upward search candidates."""
    # Drop anchors above the client cap (TPC-B: max_clients <= scale).
    base = sorted({max(1, int(c)) for c in anchors if int(c) <= max_clients}) or [1]
    if not search:
        return base
    start = max(base)
    tail = [c for c in GEOMETRIC_CONCURRENCY_LADDER if start < c <= max_clients]
    # Return anchors + tail; early-stop happens while iterating.
    return base + tail


def next_ladder_rung(current: int, cap: int) -> int | None:
    """Next ladder rung after ``current`` up to ``cap``."""
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
    """Return ``(anchors, plan, max_clients)`` for one scale.

    For TPC-B, enforce pgbench's ``-s >= -c`` by capping clients at ``scale``
    and recomputing anchors against ``min(V, scale)`` so small rungs still get
    four always-measured points instead of early-stopping the whole ladder.
    """
    if workload != "pgbench_tpcb":
        plan = choose_concurrency_plan(
            anchors=host_anchors, search=search, max_clients=host_max_clients
        )
        return host_anchors, plan, host_max_clients

    # Keep in sync with sc-inspector/inspector/benchmark_tiers.py
    # pgbench_tpcb_client_cap / pgbench_tpcb_anchor_vcpus.
    max_clients = max(1, min(int(scale), int(host_max_clients)))
    anchor_v = max(1, min(int(db_vcpus), int(scale)))
    anchors = concurrency_anchors(anchor_v)
    plan = choose_concurrency_plan(
        anchors=anchors, search=search, max_clients=max_clients
    )
    return anchors, plan, max_clients


def expand_scales(workload: str) -> list[int]:
    many = parse_csv_ints("SC_SCALEFACTORS")
    if many:
        return many
    one = env_int("SC_SCALEFACTOR", 65 if workload == "pgbench_ro" else 65)
    return [one]


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
) -> tuple[dict[str, Any], bool]:
    """Load one scale and run the concurrency plan. Returns (size_result, warmup_done)."""
    ensure_db(host, port, user, password, admin_db, dbname)
    spec = dataset_spec_for_pgbench(scalefactor=scale)
    dataset_meta = prepare_database(
        spec,
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        admin_db=admin_db,
        build=lambda: pgbench_init(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            scale=scale,
        ),
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
            # After anchors, stop when a search rung fails to improve peak by IMPROVE_PCT.
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
            )
            tpm = float(measure.get("tpm") or 0)
            prev_peak = peak_tpm
            if tpm > peak_tpm:
                peak_tpm = tpm
            entry = {
                "concurrency": clients,
                "jobs": clients,
                "anchor": is_anchor,
                "warmup_seconds": w_secs,
                **{k: v for k, v in measure.items()},
            }
            profile.append(entry)

            # Early-stop only on search (non-anchor) rungs: require ≥ IMPROVE_PCT
            # gain vs the previous peak to continue.
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
            # If we hit the planned cap and it still improves, extend by one rung
            # (bounded by hard_max_clients) and keep searching.
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
    return (
        {
            "scalefactor": scale,
            "dataset": dataset_meta,
            "profile": profile,
            "profile_vus": sorted(anchors),
            "concurrency_plan": plan_dyn,
            "profile_max_clients": max(plan) if plan else 1,
            "peak_concurrency": best.get("concurrency"),
            "score": best.get("tpm") or best.get("score") or 0,
            "tps": best.get("tps"),
            "latency_ms": best.get("latency_ms"),
            "latency_avg_ms": best.get("latency_avg_ms"),
            "latency_stddev_ms": best.get("latency_stddev_ms"),
            "stop_reason": stop_reason,
        },
        warmup_done,
    )


def main() -> int:
    workload = os.environ.get("SC_WORKLOAD", "pgbench_ro").strip().lower()
    if workload not in {"pgbench_ro", "pgbench_tpcb"}:
        raise RuntimeError(f"unsupported SC_WORKLOAD={workload!r}")

    host = os.environ["SC_DB_HOST"]
    port = env_int("SC_DB_PORT", 5432)
    user = os.environ.get("SC_DB_USER", "postgres")
    password = os.environ.get("SC_DB_PASSWORD", "postgres")
    admin_db = os.environ.get("SC_DB_NAME", "postgres")
    dbname = os.environ.get("SC_PGBENCH_DB", "pgbench")

    run_seconds = env_int("SC_RUN_SECONDS", 300)
    warmup_seconds = env_int("SC_WARMUP_SECONDS", 120)
    settle_seconds = env_int("SC_SETTLE_SECONDS", 60)
    warmup_once = env_bool("SC_WARMUP_ONCE", True)
    improve_pct = env_float("SC_PROFILE_IMPROVE_PCT", 5.0)
    search = env_bool("SC_PROFILE_SEARCH", True)
    db_vcpus = env_int("SC_DB_VCPUS", os.cpu_count() or 2)
    host_anchors = parse_csv_ints("SC_PROFILE_VUS") or [1, max(1, db_vcpus)]
    host_max_clients = env_int("SC_PROFILE_MAX_CLIENTS", max(host_anchors) * 4)
    host_hard_max_clients = env_int(
        "SC_PROFILE_HARD_MAX_CLIENTS", GEOMETRIC_CONCURRENCY_LADDER[-1]
    )
    scales = expand_scales(workload)

    sync_commit = show_synchronous_commit(host, port, user, password, admin_db)

    sizes: list[dict[str, Any]] = []
    warmup_done = False
    for scale in scales:
        anchors, plan, scale_max = plan_for_scale(
            workload=workload,
            scale=scale,
            host_anchors=host_anchors,
            search=search,
            host_max_clients=host_max_clients,
            db_vcpus=db_vcpus,
        )
        hard_max = host_hard_max_clients if workload != "pgbench_tpcb" else min(
            host_hard_max_clients, scale
        )
        if workload == "pgbench_tpcb" and plan and max(plan) > scale:
            raise RuntimeError(
                f"TPC-B plan violates pgbench docs (-s >= -c): "
                f"scale={scale} max_clients={max(plan)}"
            )
        size_result, warmup_done = run_size(
            workload=workload,
            scale=scale,
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
        )
        size_result["clients_capped_at_scale"] = workload == "pgbench_tpcb"
        size_result["profile_max_clients"] = scale_max
        sizes.append(size_result)

    # Headline score: best TPM across sizes (and their concurrency peaks).
    best_size = max(sizes, key=lambda s: float(s.get("score") or 0)) if sizes else {}
    summary: dict[str, Any] = {
        "benchmark": "pgbench_postgres",
        "workload": workload,
        "topology": os.environ.get("SC_TOPOLOGY", "multi_vm"),
        "durability": os.environ.get("SC_DURABILITY", "durable"),
        "synchronous_commit": sync_commit,
        "run_seconds": run_seconds,
        "warmup_seconds": warmup_seconds,
        "settle_seconds": settle_seconds,
        "warmup_once": warmup_once,
        "improve_pct": improve_pct,
        "profile_search": search,
        "profile_vus": host_anchors,
        "profile_max_clients": host_max_clients,
        "profile_hard_max_clients": host_hard_max_clients,
        "scalefactors": scales,
        "db_vcpus": db_vcpus,
        "client_vcpus": env_int("SC_CLIENT_VCPUS", os.cpu_count() or 2),
        "db_mem_gib": env_float("SC_DB_MEM_GIB", 0.0) or None,
        "sizes": sizes,
        "peak_scalefactor": best_size.get("scalefactor"),
        "peak_concurrency": best_size.get("peak_concurrency"),
        "score": best_size.get("score") or 0,
        "score_unit": "tpm",
        "tps": best_size.get("tps"),
        "latency_ms": best_size.get("latency_ms"),
        "latency_avg_ms": best_size.get("latency_avg_ms"),
        "latency_stddev_ms": best_size.get("latency_stddev_ms"),
        # Flat profile of the winning size for consumers that expect profile[].
        "profile": best_size.get("profile") or [],
        "concurrency_plan": best_size.get("concurrency_plan") or [],
        "scalefactor": best_size.get("scalefactor"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
