#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

WH_SIZE_GIB = 0.095
BUFFER_FRAC = 0.25
WH_PER_VU_MIN = 5
RESULTS_DIR = Path("/tmp/benchbase-results")
CONFIG_PATH = Path("/tmp/tpcc_config.xml")


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
    if vcpus <= 2:
        return sorted({1, vcpus})
    if vcpus <= 8:
        return sorted({1, 2, 4, vcpus})
    if vcpus <= 32:
        return sorted({1, 4, 8, 16, min(vcpus, 24)})
    return sorted({1, 8, 16, 32, min(vcpus, 48)})


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


def write_tpcc_config(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    warehouses: int,
    terminals: int,
    run_seconds: int,
) -> None:
    root = ET.Element("parameters")
    fields = {
        "type": "POSTGRES",
        "driver": "org.postgresql.Driver",
        "url": f"jdbc:postgresql://{host}:{port}/benchbase?sslmode=disable",
        "username": user,
        "password": password,
        "isolation": "TRANSACTION_READ_COMMITTED",
        "batchsize": "128",
        "scalefactor": str(warehouses),
        "terminals": str(terminals),
    }
    for key, value in fields.items():
        elem = ET.SubElement(root, key)
        elem.text = value

    works = ET.SubElement(root, "works")
    work = ET.SubElement(works, "work")
    for key, value in (("time", str(run_seconds)), ("rate", "unlimited"), ("weights", "45,43,4,4,4")):
        elem = ET.SubElement(work, key)
        elem.text = value

    tx_types = ET.SubElement(root, "transactiontypes")
    for name in ("NewOrder", "Payment", "OrderStatus", "Delivery", "StockLevel"):
        tx_type = ET.SubElement(tx_types, "transactiontype")
        tx_name = ET.SubElement(tx_type, "name")
        tx_name.text = name

    xml_text = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="    ")
    CONFIG_PATH.write_text(xml_text, encoding="utf-8")


def benchbase_cmd(extra_args: list[str], timeout: int) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    proc = run(
        [
            "java",
            "-jar",
            "/benchbase/benchbase.jar",
            "-b",
            "tpcc",
            "-c",
            str(CONFIG_PATH),
            "-d",
            str(RESULTS_DIR),
            *extra_args,
        ],
        timeout=timeout,
    )
    return proc.stdout + proc.stderr


def latest_summary_json() -> dict[str, Any]:
    files = sorted(RESULTS_DIR.glob("tpcc_*.summary.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError("No BenchBase summary JSON produced")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def create_and_load(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    warehouses: int,
) -> None:
    write_tpcc_config(
        host=host,
        port=port,
        user=user,
        password=password,
        warehouses=warehouses,
        terminals=1,
        run_seconds=10,
    )
    out = benchbase_cmd(["--create=true", "--load=true", "--execute=false"], timeout=14400)
    if "Exception" in out and "Finished" not in out and "Data loaded" not in out:
        raise RuntimeError(f"BenchBase load failed: {out[-2000:]}")


def run_once(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    warehouses: int,
    terminals: int,
    run_seconds: int,
) -> dict[str, Any]:
    write_tpcc_config(
        host=host,
        port=port,
        user=user,
        password=password,
        warehouses=warehouses,
        terminals=terminals,
        run_seconds=run_seconds,
    )
    out = benchbase_cmd(["--create=false", "--load=false", "--execute=true"], timeout=run_seconds + 900)
    if "Unexpected error" in out and "Throughput" not in out:
        raise RuntimeError(f"BenchBase execute failed: {out[-2000:]}")
    summary = latest_summary_json()
    tps = float(summary.get("Throughput (requests/second)", 0))
    return {
        "score": int(round(tps * 60)),
        "tps": round(tps, 2),
    }


def choose_concurrency_candidates(
    *,
    run_vus: int,
    warehouses: int,
    profiling_enabled: bool,
) -> list[int]:
    max_by_warehouses = max(1, warehouses // WH_PER_VU_MIN)
    if not profiling_enabled:
        return [max(1, min(run_vus, max_by_warehouses))]
    points = profile_points(os.cpu_count() or 2)
    points.append(run_vus)
    return sorted({max(1, min(v, max_by_warehouses)) for v in points})


def main() -> int:
    db_host = os.environ["SC_DB_HOST"]
    db_port = env_int("SC_DB_PORT", 5432)
    cache_ratio = env_float("SC_CACHE_RATIO", 1.0)
    workload = os.environ.get("SC_WORKLOAD", "tpcc").strip().lower()
    if workload != "tpcc":
        raise RuntimeError("BenchBase wrapper currently supports SC_WORKLOAD=tpcc")

    db_user = os.environ.get("SC_DB_USER", "postgres")
    db_pass = os.environ.get("SC_DB_PASSWORD", "postgres")
    db_name = os.environ.get("SC_DB_NAME", "postgres")

    default_wh = max(10, int((BUFFER_FRAC * mem_gib()) / max(cache_ratio, 0.05) / WH_SIZE_GIB))
    warehouses = env_int("SC_WAREHOUSES", default_wh)
    run_vus = env_int("SC_RUN_VUS", min(os.cpu_count() or 2, max(1, warehouses // WH_PER_VU_MIN)))
    profiling_enabled = os.environ.get("SC_PROFILE", "0") == "1"

    # Exposed for compatibility with HammerDB wrappers even though BenchBase build phase does not use VUs.
    _ = env_int("SC_BUILD_VUS", min(16, os.cpu_count() or 2))

    rtt_ms = pg_rtt_ms(db_host, db_port, db_user, db_pass, db_name)
    create_and_load(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        warehouses=warehouses,
    )

    candidates = choose_concurrency_candidates(
        run_vus=run_vus,
        warehouses=warehouses,
        profiling_enabled=profiling_enabled,
    )
    profile: list[dict[str, Any]] = []
    best = {"concurrency": candidates[0], "score": -1}
    for terminals in candidates:
        result = run_once(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            warehouses=warehouses,
            terminals=terminals,
            run_seconds=90,
        )
        entry = {"concurrency": terminals, **result}
        profile.append(entry)
        if result["score"] > best["score"]:
            best = {"concurrency": terminals, "score": result["score"]}

    final = run_once(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        warehouses=warehouses,
        terminals=best["concurrency"],
        run_seconds=120,
    )

    summary = {
        "benchmark": "benchbase_postgres",
        "workload": workload,
        "topology": "multi_vm",
        "cache_ratio": cache_ratio,
        "warehouses": warehouses,
        "client_rtt_ms": rtt_ms,
        "peak_concurrency": best["concurrency"],
        "score": final["score"],
        "score_unit": "tpm",
        "profile": profile,
    }

    output = Path("/output")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
