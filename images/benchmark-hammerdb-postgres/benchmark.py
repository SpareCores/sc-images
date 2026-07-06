#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

WH_SIZE_GIB = 0.095
BUFFER_FRAC = 0.25
WH_PER_VU_MIN = 5
TPCH_SCALE_FACTORS = (1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
TPCH_GIB_PER_SF = 1.0


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def mem_gib() -> float:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / 1024 / 1024
    return 4.0


def snap_tpch_scale(value: int) -> int:
    if value in TPCH_SCALE_FACTORS:
        return value
    allowed = [sf for sf in TPCH_SCALE_FACTORS if sf <= value]
    if allowed:
        return max(allowed)
    return min(TPCH_SCALE_FACTORS, key=lambda sf: abs(sf - value))


def tpch_scale_factor(cache_ratio: float, mem_gib: float) -> int:
    schema_gib = BUFFER_FRAC * mem_gib / max(cache_ratio, 0.05)
    target = schema_gib / TPCH_GIB_PER_SF
    allowed = [sf for sf in TPCH_SCALE_FACTORS if sf <= max(target, 1)]
    return max(allowed) if allowed else TPCH_SCALE_FACTORS[0]


def profile_points(vcpus: int) -> list[int]:
    if vcpus <= 2:
        return sorted({1, vcpus})
    if vcpus <= 8:
        return sorted({1, 2, 4, vcpus})
    if vcpus <= 32:
        return sorted({1, 4, 8, 16, min(vcpus, 24)})
    return sorted({1, 8, 16, 32, min(vcpus, 48)})


def run(args: list[str], *, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def pg_rtt_ms(host: str, port: int, user: str, password: str, dbname: str) -> float:
    """Round-trip latency on a warm connection (min of timed SELECT 1 samples)."""
    import psycopg

    warmup = env_int("SC_RTT_WARMUP", 3)
    samples_n = env_int("SC_RTT_SAMPLES", 20)

    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=10,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            for _ in range(warmup):
                cur.execute("SELECT 1")
                if cur.fetchone()[0] != 1:
                    raise RuntimeError("RTT warmup query returned unexpected result")

            samples: list[float] = []
            for _ in range(samples_n):
                t0 = time.perf_counter()
                cur.execute("SELECT 1")
                row = cur.fetchone()
                samples.append((time.perf_counter() - t0) * 1000)
                if row[0] != 1:
                    raise RuntimeError("RTT sample query returned unexpected result")

    if not samples:
        raise RuntimeError("no RTT samples collected")
    return round(min(samples), 3)


def hammerdb_cli(script: str, timeout: int = 7200) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = run(
            ["bash", "-lc", f"cd /home/hammerdb && ./hammerdbcli auto {path}"],
            timeout=timeout,
        )
        return proc.stdout + proc.stderr
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def buildschema_tpcc(host: str, port: int, warehouses: int, build_vus: int, user: str, password: str) -> None:
    script = f"""
dbset db pg
dbset bm TPC-C
vuset logtotemp 0
diset connection pg_host {host}
diset connection pg_port {port}
diset tpcc pg_superuser {user}
diset tpcc pg_superuserpass {password}
diset tpcc pg_defaultdbase postgres
diset tpcc pg_storedprocs true
diset tpcc pg_count_ware {warehouses}
diset tpcc pg_num_vu {build_vus}
buildschema
"""
    out = hammerdb_cli(script, timeout=14400)
    if "TPCC SCHEMA COMPLETE" not in out:
        raise RuntimeError(f"buildschema failed: {out[-2000:]}")


def buildschema_tpch(host: str, port: int, scale_factor: int, user: str, password: str) -> None:
    script = f"""
dbset db pg
dbset bm TPC-H
vuset logtotemp 0
diset connection pg_host {host}
diset connection pg_port {port}
diset tpch pg_superuser {user}
diset tpch pg_superuserpass {password}
diset tpch pg_defaultdbase postgres
diset tpch pg_scale_fact {scale_factor}
buildschema
"""
    out = hammerdb_cli(script, timeout=14400)
    if "TPCH SCHEMA COMPLETE" not in out and "SCHEMA COMPLETE" not in out:
        raise RuntimeError(f"buildschema failed: {out[-2000:]}")


def run_tpcc(host: str, port: int, run_vus: int, user: str, password: str) -> dict[str, int]:
    script = f"""
dbset db pg
dbset bm TPC-C
vuset logtotemp 1
vuset unique 1
diset connection pg_host {host}
diset connection pg_port {port}
diset tpcc pg_superuser {user}
diset tpcc pg_superuserpass {password}
diset tpcc pg_defaultdbase tpcc
diset tpcc pg_user tpcc
diset tpcc pg_pass tpcc
diset tpcc pg_storedprocs true
diset tpcc pg_driver timed
diset tpcc pg_rampup 1
diset tpcc pg_duration 2
loadscript
vuset vu {run_vus}
vucreate
vurun
vudestroy
"""
    out = hammerdb_cli(script, timeout=3600)
    match = re.search(r"TEST RESULT : System achieved (\d+) NOPM from (\d+) PostgreSQL TPM", out)
    if not match:
        raise RuntimeError(f"no TPROC-C result found: {out[-2000:]}")
    return {"score": int(match.group(1)), "tpm": int(match.group(2))}


def run_tpch(host: str, port: int, run_vus: int, user: str, password: str) -> dict[str, int]:
    script = f"""
dbset db pg
dbset bm TPC-H
vuset logtotemp 1
vuset unique 1
diset connection pg_host {host}
diset connection pg_port {port}
diset tpch pg_superuser {user}
diset tpch pg_superuserpass {password}
diset tpch pg_defaultdbase tpch
diset tpch pg_driver timed
diset tpch pg_rampup 1
diset tpch pg_duration 2
loadscript
vuset vu {run_vus}
vucreate
vurun
vudestroy
"""
    out = hammerdb_cli(script, timeout=3600)
    match = re.search(r"QphH@Size.*?(\d+)", out)
    if not match:
        raise RuntimeError(f"no TPROC-H score found: {out[-2000:]}")
    return {"score": int(match.group(1)), "tpm": 0}


def choose_concurrency_candidates(
    *,
    run_vus: int,
    warehouses: int,
    profiling_enabled: bool,
    profile_vcpus: int,
) -> list[int]:
    max_by_warehouses = max(1, warehouses // WH_PER_VU_MIN)
    if not profiling_enabled:
        return [max(1, min(run_vus, max_by_warehouses))]
    points = profile_points(profile_vcpus)
    points.append(run_vus)
    return sorted({max(1, min(v, max_by_warehouses)) for v in points})


def sizing_vcpus(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return fallback
    return max(1, int(value))


def main() -> int:
    db_host = os.environ["SC_DB_HOST"]
    db_port = env_int("SC_DB_PORT", 5432)
    cache_ratio = env_float("SC_CACHE_RATIO", 1.0)
    workload = os.environ.get("SC_WORKLOAD", "tpcc").strip().lower()
    if workload not in {"tpcc", "tpch"}:
        raise RuntimeError("SC_WORKLOAD must be tpcc or tpch")

    db_user = os.environ.get("SC_DB_USER", "postgres")
    db_pass = os.environ.get("SC_DB_PASSWORD", "postgres")
    db_name = os.environ.get("SC_DB_NAME", "postgres")

    mem = mem_gib()
    default_wh = max(10, int((BUFFER_FRAC * mem) / max(cache_ratio, 0.05) / WH_SIZE_GIB))
    db_vcpus = sizing_vcpus("SC_DB_VCPUS", os.cpu_count() or 2)
    client_vcpus = sizing_vcpus("SC_CLIENT_VCPUS", os.cpu_count() or 2)
    profiling_enabled = os.environ.get("SC_PROFILE", "0") == "1"

    if workload == "tpcc":
        warehouses = env_int("SC_WAREHOUSES", default_wh)
        scale_units = warehouses
        build_vus = min(
            env_int("SC_BUILD_VUS", min(16, db_vcpus, warehouses)),
            client_vcpus,
            warehouses,
        )
        run_vus = env_int("SC_RUN_VUS", min(db_vcpus, max(1, warehouses // WH_PER_VU_MIN)))
    else:
        wh_env = os.environ.get("SC_WAREHOUSES")
        scale_units = snap_tpch_scale(int(wh_env)) if wh_env else tpch_scale_factor(cache_ratio, mem)
        build_vus = min(env_int("SC_BUILD_VUS", min(16, db_vcpus)), client_vcpus)
        run_vus = env_int("SC_RUN_VUS", min(db_vcpus, max(1, scale_units // WH_PER_VU_MIN)))

    rtt_ms = pg_rtt_ms(db_host, db_port, db_user, db_pass, db_name)

    if workload == "tpcc":
        buildschema_tpcc(db_host, db_port, scale_units, build_vus, db_user, db_pass)
    else:
        buildschema_tpch(db_host, db_port, scale_units, db_user, db_pass)

    candidates = choose_concurrency_candidates(
        run_vus=run_vus,
        warehouses=scale_units,
        profiling_enabled=profiling_enabled,
        profile_vcpus=db_vcpus,
    )
    profile: list[dict[str, Any]] = []
    best = {"concurrency": candidates[0], "score": -1, "tpm": 0}
    for vus in candidates:
        result = run_tpcc(db_host, db_port, vus, db_user, db_pass) if workload == "tpcc" else run_tpch(db_host, db_port, vus, db_user, db_pass)
        entry = {"concurrency": vus, **result}
        profile.append(entry)
        if result["score"] > best["score"]:
            best = {"concurrency": vus, "score": result["score"], "tpm": result["tpm"]}

    final = run_tpcc(db_host, db_port, best["concurrency"], db_user, db_pass) if workload == "tpcc" else run_tpch(db_host, db_port, best["concurrency"], db_user, db_pass)

    summary: dict[str, Any] = {
        "benchmark": "hammerdb_postgres",
        "workload": workload,
        "topology": "multi_vm",
        "cache_ratio": cache_ratio,
        "client_rtt_ms": rtt_ms,
        "peak_concurrency": best["concurrency"],
        "score": final["score"],
        "score_unit": "NOPM" if workload == "tpcc" else "QphH",
        "profile": profile,
    }
    if workload == "tpcc":
        summary["warehouses"] = scale_units
    else:
        summary["scale_factor"] = scale_units

    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
