#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

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


def profile_points(vcpus: int) -> list[int]:
    """Legacy local ladder when SC_PROFILE_VUS is not set (older inspector builds)."""
    if vcpus <= 2:
        return sorted({1, vcpus})
    if vcpus <= 8:
        return sorted({1, 2, 4, vcpus})
    if vcpus <= 32:
        return sorted({1, 4, 8, 16, vcpus})
    return sorted({1, 8, 16, 32, vcpus})


def parse_profile_vus() -> list[int] | None:
    raw = os.environ.get("SC_PROFILE_VUS", "").strip()
    if not raw:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


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
) -> Path:
    path = config_path(bench)
    root = ET.Element("parameters")
    fields = {
        "type": "POSTGRES",
        "driver": "org.postgresql.Driver",
        "url": (
            f"jdbc:postgresql://{host}:{port}/benchbase"
            f"?sslmode=disable&ApplicationName={bench}&reWriteBatchedInserts=true"
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
    for key, value in (("time", str(run_seconds)), ("rate", "unlimited"), ("weights", weights)):
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
        txn_types=spec["txn_types"],
        weights=spec["weights"],
        extra=spec.get("extra", {}),
    )
    out = benchbase_cmd(
        bench,
        config,
        ["--create=false", "--load=false", "--execute=true"],
        timeout=run_seconds + 900,
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


def choose_concurrency_candidates(
    *,
    run_vus: int,
    scalefactor: int,
    profiling_enabled: bool,
    profile_vcpus: int,
) -> list[int]:
    max_by_scale = max(1, scalefactor // UNITS_PER_VU_MIN)
    if not profiling_enabled:
        return [max(1, min(run_vus, max_by_scale))]
    profile_vus = parse_profile_vus()
    if profile_vus is not None:
        points = list(profile_vus)
    else:
        points = list(profile_points(profile_vcpus))
        points.append(run_vus)
    return sorted({max(1, min(v, max_by_scale)) for v in points})


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
    spec = WORKLOADS.get(workload)
    if spec is None:
        supported = ", ".join(sorted(WORKLOADS))
        raise RuntimeError(f"BenchBase wrapper supports SC_WORKLOAD in {{{supported}}}, got {workload!r}")

    db_user = os.environ.get("SC_DB_USER", "postgres")
    db_pass = os.environ.get("SC_DB_PASSWORD", "postgres")
    db_name = os.environ.get("SC_DB_NAME", "postgres")

    scalefactor = env_int("SC_SCALEFACTOR", default_scalefactor(workload, cache_ratio))
    db_vcpus = sizing_vcpus("SC_DB_VCPUS", os.cpu_count() or 2)
    run_vus = env_int("SC_RUN_VUS", min(db_vcpus, max(1, scalefactor // UNITS_PER_VU_MIN)))
    profiling_enabled = os.environ.get("SC_PROFILE", "0") == "1"

    # Exposed for compatibility with HammerDB wrappers even though BenchBase build phase does not use VUs.
    _ = env_int("SC_BUILD_VUS", min(16, db_vcpus))

    rtt_ms = pg_rtt_ms(db_host, db_port, db_user, db_pass, db_name)
    create_and_load(
        spec=spec,
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        scalefactor=scalefactor,
    )

    candidates = choose_concurrency_candidates(
        run_vus=run_vus,
        scalefactor=scalefactor,
        profiling_enabled=profiling_enabled,
        profile_vcpus=db_vcpus,
    )
    profile: list[dict[str, Any]] = []
    best = {"concurrency": candidates[0], "score": -1}
    for terminals in candidates:
        result = run_once(
            spec=spec,
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            scalefactor=scalefactor,
            terminals=terminals,
            run_seconds=90,
        )
        entry = {"concurrency": terminals, **result}
        profile.append(entry)
        if result["score"] > best["score"]:
            best = {"concurrency": terminals, "score": result["score"]}

    final = run_once(
        spec=spec,
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        scalefactor=scalefactor,
        terminals=best["concurrency"],
        run_seconds=120,
    )

    summary: dict[str, Any] = {
        "benchmark": "benchbase_postgres",
        "workload": workload,
        "topology": "multi_vm",
        "cache_ratio": cache_ratio,
        "durability": os.environ.get("SC_DURABILITY", "durable"),
        "scalefactor": scalefactor,
        "client_rtt_ms": rtt_ms,
        "peak_concurrency": best["concurrency"],
        "score": final["score"],
        "score_unit": "tpm",
        "profile": profile,
    }
    if final.get("latency_ms"):
        summary["latency_ms"] = final["latency_ms"]
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
