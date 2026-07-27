#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_dataset_cache import dataset_spec_for_benchbase, prepare_database

WH_SIZE_GIB = 0.095
BUFFER_FRAC = 0.25
UNITS_PER_VU_MIN = 5
RESULTS_DIR = Path("/tmp/benchbase-results")
BENCHBASE_JAR = Path("/benchbase/profiles/postgres/benchbase.jar")
BENCHBASE_JAVA = Path("/opt/java/openjdk/bin/java")

WORKLOADS: dict[str, dict[str, Any]] = {
    "tpcc": {
        "bench": "tpcc",
        "txn_types": ("NewOrder", "Payment", "OrderStatus", "Delivery", "StockLevel"),
        "weights": "45,43,4,4,4",
        "extra": {},
    },
    "wikipedia": {
        "bench": "wikipedia",
        "txn_types": (
            "AddWatchList",
            "RemoveWatchList",
            "UpdatePage",
            "GetPageAnonymous",
            "GetPageAuthenticated",
        ),
        "weights": "1,1,7,90,1",
        "extra": {},
    },
    "ycsb": {
        "bench": "ycsb",
        "txn_types": ("ReadRecord", "UpdateRecord"),
        "weights": "50,50",
        "extra": {},
    },
}


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


def provision_context() -> dict[str, Any]:
    if not os.environ.get("SC_PROVISION_VENDOR_ID"):
        return {}
    ctx: dict[str, Any] = {
        "topology": os.environ.get("SC_TOPOLOGY", "dbaas"),
        "cache_tier": os.environ.get("SC_CACHE_TIER", ""),
        "vendor_id": os.environ["SC_PROVISION_VENDOR_ID"],
        "native_id": os.environ.get("SC_PROVISION_NATIVE_ID", ""),
        "engine_version": os.environ.get("SC_PROVISION_ENGINE_VERSION", ""),
        "ha_mode": os.environ.get("SC_PROVISION_HA_MODE", ""),
        "sku_id": os.environ.get("SC_PROVISION_SKU_ID", ""),
        "cpu_count": float(os.environ.get("SC_PROVISION_CPU_COUNT", "0") or 0),
        "memory_gib": float(os.environ.get("SC_PROVISION_MEMORY_GIB", "0") or 0),
        "storage_gib": int(os.environ.get("SC_PROVISION_STORAGE_GIB", "0") or 0),
        "storage_edition": os.environ.get("SC_PROVISION_STORAGE_EDITION", ""),
        "client_instance": os.environ.get("SC_PROVISION_CLIENT_INSTANCE", ""),
        "region": os.environ.get("SC_PROVISION_REGION", ""),
        "zone": os.environ.get("SC_PROVISION_ZONE", ""),
        "db_fqdn": os.environ.get("SC_DB_HOST", ""),
        "network_mode": os.environ.get("SC_PROVISION_NETWORK_MODE", ""),
    }
    storage_type = os.environ.get("SC_PROVISION_STORAGE_TYPE", "").strip()
    if storage_type:
        ctx["storage_type"] = storage_type
    iops_tier = os.environ.get("SC_PROVISION_IOPS_TIER", "").strip()
    if iops_tier:
        ctx["iops_tier"] = iops_tier
    disk_iops = os.environ.get("SC_PROVISION_DISK_IOPS", "").strip()
    if disk_iops:
        ctx["disk_iops"] = int(disk_iops)
    disk_throughput = os.environ.get("SC_PROVISION_DISK_THROUGHPUT", "").strip()
    if disk_throughput:
        ctx["disk_throughput_mb_s"] = int(disk_throughput)
    raw = os.environ.get("SC_PROVISION_SYNC_COMMIT_SETTABLE", "").strip().lower()
    if raw in ("true", "false"):
        ctx["sync_commit_session_settable"] = raw == "true"
    return ctx


def host_context() -> dict[str, Any]:
    """Merge multi-VM host disk metadata when SC_MULTI_VM_* env vars are set."""
    if os.environ.get("SC_TOPOLOGY", "multi_vm") != "multi_vm":
        return {}
    ctx: dict[str, Any] = {"topology": "multi_vm"}
    raw_gib = os.environ.get("SC_PROVISIONED_DISK_GIB", "").strip()
    if raw_gib:
        ctx["storage_gib"] = int(raw_gib)
    disk_type = os.environ.get("SC_MULTI_VM_DB_DISK_TYPE", "").strip()
    if disk_type:
        ctx["storage_type"] = disk_type
    disk_iops = os.environ.get("SC_MULTI_VM_DB_DISK_IOPS", "").strip()
    if disk_iops:
        ctx["disk_iops"] = int(disk_iops)
    disk_throughput = os.environ.get("SC_MULTI_VM_DB_DISK_THROUGHPUT", "").strip()
    if disk_throughput:
        ctx["disk_throughput_mb_s"] = int(disk_throughput)
    return ctx


def sizing_context() -> dict[str, Any]:
    """Client/DB sizing knobs forwarded from inspector via SC_* env vars."""
    ctx: dict[str, Any] = {}
    for key, env_name, cast in (
        ("db_vcpus", "SC_DB_VCPUS", int),
        ("client_vcpus", "SC_CLIENT_VCPUS", int),
        ("db_mem_gib", "SC_DB_MEM_GIB", float),
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            ctx[key] = cast(raw)
    profile = os.environ.get("SC_PROFILE_VUS", "").strip()
    if profile:
        ctx["profile_vus"] = [int(p) for p in profile.split(",") if p.strip()]
    pg_image = os.environ.get("SC_PG_IMAGE", "").strip()
    if pg_image:
        ctx["pg_image"] = pg_image
    return ctx


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


def profile_points(vcpus: int) -> list[int]:
    """Legacy local ladder when SC_PROFILE_VUS is not set (older inspector builds)."""
    vcpus = max(1, int(vcpus))
    rungs = [1]
    if vcpus >= 2:
        rungs.append(max(1, vcpus // 2))
        rungs.append(vcpus)
    return sorted(set(rungs))


def parse_profile_vus() -> list[int] | None:
    raw = os.environ.get("SC_PROFILE_VUS", "").strip()
    if not raw:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def timed_run_seconds() -> int:
    return env_int("SC_RUN_SECONDS", 300)


def timed_warmup_seconds() -> int:
    return env_int("SC_WARMUP_SECONDS", 120)


def timed_settle_seconds() -> int:
    return env_int("SC_SETTLE_SECONDS", 60)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def choose_concurrency_candidates(
    *,
    run_vus: int,
    scalefactor: int,
    profiling_enabled: bool,
    profile_vcpus: int,
) -> tuple[list[int], set[int]]:
    """Return (ordered plan, anchor set).

    Anchors from ``SC_PROFILE_VUS`` (or legacy 1/n/2/n) are always measured.
    When ``SC_PROFILE_SEARCH=1``, append geometric rungs above max(anchor) up to
    ``SC_PROFILE_MAX_CLIENTS``; the main loop early-stops the search tail.
    """
    del scalefactor  # wikipedia no longer ties SF to terminals
    profile_vus = parse_profile_vus()
    if profile_vus is not None:
        anchors = sorted({max(1, int(v)) for v in profile_vus})
    elif not profiling_enabled:
        anchors = [max(1, run_vus)]
    else:
        anchors = sorted(set(profile_points(profile_vcpus) + [run_vus]))

    plan = list(anchors)
    if env_bool("SC_PROFILE_SEARCH", False) and anchors:
        max_clients = env_int("SC_PROFILE_MAX_CLIENTS", max(anchors) * 4)
        start = max(anchors)
        plan.extend(
            c for c in GEOMETRIC_CONCURRENCY_LADDER if start < c <= max_clients
        )
    return plan, set(anchors)


def benchbase_latency_ms(summary: dict[str, Any]) -> dict[str, float] | None:
    """Transaction latency (ms) from BenchBase ``Latency Distribution`` (microseconds)."""
    dist = summary.get("Latency Distribution")
    if not isinstance(dist, dict):
        return None

    def us_to_ms(key: str) -> float | None:
        if key not in dist:
            return None
        return round(float(dist[key]) / 1000.0, 3)

    latency = {
        "p50": us_to_ms("Median Latency (microseconds)"),
        "p95": us_to_ms("95th Percentile Latency (microseconds)"),
        "p99": us_to_ms("99th Percentile Latency (microseconds)"),
        "avg": us_to_ms("Average Latency (microseconds)"),
        "min": us_to_ms("Minimum Latency (microseconds)"),
        "max": us_to_ms("Maximum Latency (microseconds)"),
    }
    if all(v is None for v in latency.values()):
        return None
    return {k: v for k, v in latency.items() if v is not None}


def run(args: list[str], *, timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def db_sslmode() -> str:
    return os.environ.get("SC_DB_SSLMODE", "prefer").strip() or "prefer"


def pg_connect_kwargs() -> dict[str, str]:
    mode = db_sslmode()
    if mode == "disable":
        return {}
    return {"sslmode": mode}


def show_synchronous_commit(host: str, port: int, user: str, password: str, dbname: str) -> str:
    """Return SHOW synchronous_commit for a benchmark DB session."""
    import psycopg

    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=10,
        autocommit=True,
        **pg_connect_kwargs(),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW synchronous_commit")
            return str(cur.fetchone()[0])


def synchronous_commit_verify(
    host: str,
    port: int,
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture SHOW synchronous_commit for each benchmark-relevant DB session."""
    verified: list[dict[str, str]] = []
    benchmark_session: dict[str, str] | None = None
    for sess in sessions:
        value = show_synchronous_commit(
            host,
            port,
            sess["user"],
            sess["password"],
            sess["database"],
        )
        entry = {
            "user": sess["user"],
            "database": sess["database"],
            "synchronous_commit": value,
        }
        verified.append(entry)
        if sess.get("benchmark"):
            benchmark_session = entry
    out: dict[str, Any] = {"sessions": verified}
    if benchmark_session is not None:
        out["benchmark_session"] = benchmark_session
    return out


def postgres_repro_context(
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
) -> dict[str, Any]:
    """Snapshot live Postgres GUCs and related knobs for run reproduction."""
    import psycopg

    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=10,
        autocommit=True,
        **pg_connect_kwargs(),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = str(cur.fetchone()[0])
            cur.execute("SHOW server_version")
            server_version = str(cur.fetchone()[0])
            cur.execute("SHOW server_version_num")
            server_version_num = int(cur.fetchone()[0])
            cur.execute("SELECT pg_is_in_recovery()")
            in_recovery = bool(cur.fetchone()[0])
            cur.execute(
                """
                SELECT name, setting, unit, current_setting(name) AS pretty,
                       category, short_desc, context, vartype,
                       source, pending_restart
                FROM pg_settings
                ORDER BY name
                """
            )
            settings: dict[str, str] = {}
            nondefault: dict[str, dict[str, Any]] = {}
            for (
                name,
                setting,
                unit,
                pretty,
                category,
                short_desc,
                context,
                vartype,
                source,
                pending_restart,
            ) in cur.fetchall():
                # Prefer SHOW-style values (e.g. 4GB) over raw unit counts.
                settings[name] = pretty
                if source and source != "default":
                    entry: dict[str, Any] = {
                        "setting": pretty,
                        "source": source,
                        "context": context,
                        "vartype": vartype,
                        "category": category,
                    }
                    if unit:
                        entry["unit"] = unit
                        entry["setting_raw"] = setting
                    if short_desc:
                        entry["short_desc"] = short_desc
                    if pending_restart:
                        entry["pending_restart"] = True
                    nondefault[name] = entry
            cur.execute(
                "SELECT extname, extversion FROM pg_extension ORDER BY extname"
            )
            extensions = [
                {"name": name, "version": ver} for name, ver in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT
                    COALESCE(r.rolname, 'All'),
                    COALESCE(d.datname, 'All'),
                    s.setconfig
                FROM pg_db_role_setting s
                LEFT JOIN pg_roles r ON r.oid = s.setrole
                LEFT JOIN pg_database d ON d.oid = s.setdatabase
                ORDER BY 1, 2
                """
            )
            role_settings = [
                {
                    "role": role,
                    "database": database,
                    "config": list(config or []),
                }
                for role, database, config in cur.fetchall()
            ]

    out: dict[str, Any] = {
        "version": version,
        "server_version": server_version,
        "server_version_num": server_version_num,
        "in_recovery": in_recovery,
        "settings": settings,
        "nondefault_settings": nondefault,
        "extensions": extensions,
        "role_settings": role_settings,
    }
    raw_requested = os.environ.get("SC_PG_GUCS_REQUESTED", "").strip()
    if raw_requested:
        try:
            requested = json.loads(raw_requested)
            if isinstance(requested, dict):
                out["requested_gucs"] = requested
        except json.JSONDecodeError:
            out["requested_gucs_raw"] = raw_requested
    return out


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
        **pg_connect_kwargs(),
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


def default_scalefactor(workload: str, cache_ratio: float) -> int:
    mem = mem_gib()
    schema_gib = BUFFER_FRAC * mem / cache_ratio
    if workload == "wikipedia":
        return max(10, min(200, int(mem / 2.5)))
    if workload == "ycsb":
        return max(100, min(500_000, int(schema_gib * 1024)))
    return max(10, int(schema_gib / WH_SIZE_GIB))


def config_path(bench: str) -> Path:
    return Path(f"/tmp/{bench}_config.xml")


def jdbc_sslmode() -> str:
    return os.environ.get("SC_DB_SSLMODE", "disable").strip() or "disable"


def write_config(
    *,
    bench: str,
    host: str,
    port: int,
    user: str,
    password: str,
    scalefactor: int,
    terminals: int,
    run_seconds: int,
    txn_types: tuple[str, ...],
    weights: str,
    extra: dict[str, str],
    warmup_seconds: int = 0,
) -> Path:
    path = config_path(bench)
    root = ET.Element("parameters")
    fields = {
        "type": "POSTGRES",
        "driver": "org.postgresql.Driver",
        "url": (
            f"jdbc:postgresql://{host}:{port}/benchbase"
            f"?sslmode={jdbc_sslmode()}&ApplicationName={bench}&reWriteBatchedInserts=true"
        ),
        "username": user,
        "password": password,
        "reconnectOnConnectionFailure": "true",
        "isolation": "TRANSACTION_READ_COMMITTED",
        "batchsize": "128",
        "scalefactor": str(scalefactor),
        "terminals": str(terminals),
    }
    for key, value in fields.items():
        elem = ET.SubElement(root, key)
        elem.text = value
    for key, value in extra.items():
        elem = ET.SubElement(root, key)
        elem.text = value

    works = ET.SubElement(root, "works")
    work = ET.SubElement(works, "work")
    for key, value in (
        ("warmup", str(max(0, int(warmup_seconds)))),
        ("time", str(run_seconds)),
        ("rate", "unlimited"),
        ("weights", weights),
    ):
        elem = ET.SubElement(work, key)
        elem.text = value

    tx_types = ET.SubElement(root, "transactiontypes")
    for name in txn_types:
        tx_type = ET.SubElement(tx_types, "transactiontype")
        tx_name = ET.SubElement(tx_type, "name")
        tx_name.text = name

    xml_text = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="    ")
    path.write_text(xml_text, encoding="utf-8")
    return path


def java_bin() -> str:
    if BENCHBASE_JAVA.is_file():
        return str(BENCHBASE_JAVA)
    found = shutil.which("java")
    if found:
        return found
    raise RuntimeError("java not found (expected /opt/java/openjdk/bin/java in BenchBase image)")


def benchbase_cmd(bench: str, config: Path, extra_args: list[str], timeout: int) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    proc = run(
        [
            java_bin(),
            "-jar",
            str(BENCHBASE_JAR),
            "-b",
            bench,
            "-c",
            str(config),
            "-d",
            str(RESULTS_DIR),
            *extra_args,
        ],
        timeout=timeout,
    )
    return proc.stdout + proc.stderr


def latest_summary_json(bench: str) -> dict[str, Any]:
    files = sorted(RESULTS_DIR.glob(f"{bench}_*.summary.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError("No BenchBase summary JSON produced")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def create_and_load(
    *,
    spec: dict[str, Any],
    host: str,
    port: int,
    user: str,
    password: str,
    scalefactor: int,
) -> None:
    bench = spec["bench"]
    config = write_config(
        bench=bench,
        host=host,
        port=port,
        user=user,
        password=password,
        scalefactor=scalefactor,
        terminals=1,
        run_seconds=10,
        txn_types=spec["txn_types"],
        weights=spec["weights"],
        extra=spec.get("extra", {}),
    )
    out = benchbase_cmd(bench, config, ["--create=true", "--load=true", "--execute=false"], timeout=14400)
    if "Exception" in out and "Finished" not in out and "Data loaded" not in out:
        raise RuntimeError(f"BenchBase load failed: {out[-2000:]}")


def run_once(
    *,
    spec: dict[str, Any],
    host: str,
    port: int,
    user: str,
    password: str,
    scalefactor: int,
    terminals: int,
    run_seconds: int,
    warmup_seconds: int,
) -> dict[str, Any]:
    bench = spec["bench"]
    config = write_config(
        bench=bench,
        host=host,
        port=port,
        user=user,
        password=password,
        scalefactor=scalefactor,
        terminals=terminals,
        run_seconds=run_seconds,
        warmup_seconds=warmup_seconds,
        txn_types=spec["txn_types"],
        weights=spec["weights"],
        extra=spec.get("extra", {}),
    )
    out = benchbase_cmd(
        bench,
        config,
        ["--create=false", "--load=false", "--execute=true"],
        timeout=run_seconds + warmup_seconds + 900,
    )
    if "Unexpected error" in out and "Throughput" not in out:
        raise RuntimeError(f"BenchBase execute failed: {out[-2000:]}")
    summary = latest_summary_json(bench)
    tps = float(summary.get("Throughput (requests/second)", 0))
    result: dict[str, Any] = {
        "score": int(round(tps * 60)),
        "tps": round(tps, 2),
    }
    latency_ms = benchbase_latency_ms(summary)
    if latency_ms:
        result["latency_ms"] = latency_ms
    return result


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
    workload_spec = WORKLOADS.get(workload)
    if workload_spec is None:
        supported = ", ".join(sorted(WORKLOADS))
        raise RuntimeError(f"BenchBase wrapper supports SC_WORKLOAD in {{{supported}}}, got {workload!r}")

    db_user = os.environ.get("SC_DB_USER", "postgres")
    db_pass = os.environ.get("SC_DB_PASSWORD", "postgres")
    db_name = os.environ.get("SC_DB_NAME", "postgres")

    scalefactor = env_int("SC_SCALEFACTOR", default_scalefactor(workload, cache_ratio))
    db_vcpus = sizing_vcpus("SC_DB_VCPUS", os.cpu_count() or 2)
    run_vus = env_int("SC_RUN_VUS", min(db_vcpus, max(1, scalefactor // UNITS_PER_VU_MIN)))
    profiling_enabled = os.environ.get("SC_PROFILE", "0") == "1"
    run_seconds = timed_run_seconds()
    warmup_seconds = timed_warmup_seconds()

    bench_db = "benchbase"
    rtt_ms = pg_rtt_ms(db_host, db_port, db_user, db_pass, db_name)
    dataset_spec = dataset_spec_for_benchbase(workload=workload, scalefactor=scalefactor)
    dataset_meta = prepare_database(
        dataset_spec,
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        dbname=bench_db,
        admin_db=db_name,
        build=lambda: create_and_load(
            spec=workload_spec,
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            scalefactor=scalefactor,
        ),
    )

    sync_verify = synchronous_commit_verify(
        db_host,
        db_port,
        [{"user": db_user, "password": db_pass, "database": db_name, "benchmark": True}],
    )
    try:
        postgres_repro = postgres_repro_context(
            db_host, db_port, db_user, db_pass, db_name
        )
    except Exception as exc:
        postgres_repro = {"error": str(exc)}

    candidates, anchors = choose_concurrency_candidates(
        run_vus=run_vus,
        scalefactor=scalefactor,
        profiling_enabled=profiling_enabled,
        profile_vcpus=db_vcpus,
    )
    warmup_once = env_bool("SC_WARMUP_ONCE", True)
    settle_seconds = timed_settle_seconds()
    improve_pct = env_float("SC_PROFILE_IMPROVE_PCT", 5.0)
    profile: list[dict[str, Any]] = []
    best = {"concurrency": candidates[0], "score": -1}
    peak_score = 0
    warmup_done = False
    stop_reason = ""
    for terminals in candidates:
        is_anchor = terminals in anchors
        w_secs = warmup_seconds
        if warmup_once and warmup_done:
            w_secs = settle_seconds
        result = run_once(
            spec=workload_spec,
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            scalefactor=scalefactor,
            terminals=terminals,
            run_seconds=run_seconds,
            warmup_seconds=w_secs,
        )
        warmup_done = True
        score = int(result.get("score") or 0)
        prev_peak = peak_score
        if score > peak_score:
            peak_score = score
        entry = {
            "concurrency": terminals,
            "anchor": is_anchor,
            "warmup_seconds": w_secs,
            **result,
        }
        if peak_score > 0:
            entry["tpm_vs_peak_pct"] = round(100.0 * score / peak_score, 2)
        profile.append(entry)
        if score > best["score"]:
            best = {"concurrency": terminals, "score": score}
        if (
            not is_anchor
            and prev_peak > 0
            and score < prev_peak * (1.0 + improve_pct / 100.0)
        ):
            stop_reason = (
                f"tpm {score} did not improve peak {prev_peak} by >={improve_pct:g}%"
            )
            entry["stop_reason"] = stop_reason
            break

    best_entry = next(p for p in profile if p["concurrency"] == best["concurrency"])

    summary: dict[str, Any] = {
        "benchmark": "benchbase_postgres",
        "workload": workload,
        "topology": os.environ.get("SC_TOPOLOGY", "multi_vm"),
        "cache_ratio": cache_ratio,
        "durability": os.environ.get("SC_DURABILITY", "durable"),
        "scalefactor": scalefactor,
        "run_seconds": run_seconds,
        "warmup_seconds": warmup_seconds,
        "settle_seconds": settle_seconds,
        "warmup_once": warmup_once,
        "improve_pct": improve_pct,
        "profile_search": env_bool("SC_PROFILE_SEARCH", False),
        "client_rtt_ms": rtt_ms,
        "peak_concurrency": best["concurrency"],
        "score": best_entry["score"],
        "score_unit": "tpm",
        "tps": best_entry.get("tps"),
        "profile": profile,
        "stop_reason": stop_reason,
    }
    summary.update(provision_context())
    summary.update(host_context())
    summary.update(sizing_context())
    summary["dataset"] = dataset_meta
    summary["synchronous_commit"] = sync_verify
    summary["postgres"] = postgres_repro
    if best_entry.get("latency_ms"):
        summary["latency_ms"] = best_entry["latency_ms"]
    if workload == "tpcc":
        summary["warehouses"] = scalefactor

    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
