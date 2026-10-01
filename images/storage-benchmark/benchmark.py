#!/usr/bin/env python3
"""Raw block device benchmark with fio.

Discovers the real block devices of the host, groups identical ones, benchmarks one
representative per group (single mode) and all members of multi-device groups
concurrently (group mode), then prints a single JSON document on stdout.

Needs a privileged container with the host PID namespace (`--privileged --pid=host`):
host mounts are read from /proc/1/mountinfo and unmounts run via nsenter.
"""

import bisect
import glob
import itertools
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA_VERSION = 1
KiB, MiB, GiB = 1024, 1024**2, 1024**3


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


RUNTIME_S = env_int("SC_STORAGE_RUNTIME_S", 10)
RAMP_S = env_int("SC_STORAGE_RAMP_S", 2)
MIN_RUNTIME_S = env_int("SC_STORAGE_MIN_RUNTIME_S", 5)
FILL_CAP_S = env_int("SC_STORAGE_FILL_CAP_S", 90)
WS_TARGET_BYTES = env_int("SC_STORAGE_WS_TARGET_BYTES", 128 * GiB)
BUDGET_S = env_int("SC_STORAGE_BUDGET_S", 2400)
UMOUNT = os.environ.get("SC_STORAGE_UMOUNT", "1") != "0"

LOG_AVG_MSEC = 500
INFLIGHT_CAP_BYTES = 256 * MiB
MAX_JOBS_PER_DEVICE = 4
QD_RANDOM = 256
QD_SEQ = 64
FILL_BS = 1 * MiB
# per-test fio startup/teardown overhead used for the time budget estimate
TEST_OVERHEAD_S = 2
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99]
BLOCK_SIZES = [512, 4 * KiB, 16 * KiB, 64 * KiB, 256 * KiB, 1 * MiB, 4 * MiB, 16 * MiB]
# dropped first when the estimated runtime exceeds the budget
LOW_PRIORITY_TESTS = {"randread_512", "randwrite_512", "randread_16m", "randwrite_16m", "randrw70_64k", "randrw70_1m"}

PROTECTED_MOUNTS = {"/", "/boot", "/boot/efi", "/usr", "/var", "/home", "/root", "/etc", "/opt", "/srv", "/tmp"}
PROTECTED_PREFIXES = ("/snap/",)
DOCKER_ROOT = "/var/lib/docker"
VIRTUAL_NAME_RE = re.compile(r"^(zram|ram|loop|nbd|md|dm-|sr|fd)")
# vendor quirks: identical devices reported with per-device model names
MODEL_NORMALIZERS = [(re.compile(r"^nvme_card\d+$"), "nvme_card")]

NPROC = len(os.sched_getaffinity(0))


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def host_cmd(args: list[str]) -> list[str]:
    return ["nsenter", "-t", "1", "-m", "--", *args]


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def bs_label(bs: int) -> str:
    for unit, suffix in ((MiB, "m"), (KiB, "k")):
        if bs >= unit and bs % unit == 0:
            return f"{bs // unit}{suffix}"
    return str(bs)


# ---------------------------------------------------------------------------
# host state
# ---------------------------------------------------------------------------


def unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def host_mounts() -> dict[str, list[dict]]:
    """maj:min -> mounts, from the host (PID 1) mount namespace."""
    mounts = defaultdict(list)
    with open("/proc/1/mountinfo") as f:
        for line in f:
            pre, _, post = line.partition(" - ")
            pre_fields, post_fields = pre.split(), post.split()
            mounts[pre_fields[2]].append(
                {
                    "mountpoint": unescape(pre_fields[4]),
                    "options": pre_fields[5],
                    "fstype": post_fields[0],
                    "source": unescape(post_fields[1]) if len(post_fields) > 1 else None,
                }
            )
    return mounts


def host_swaps() -> set[str]:
    """maj:min of block devices used as swap (swap files live on mounted filesystems anyway)."""
    swaps = set()
    with open("/proc/swaps") as f:
        next(f, None)
        for line in f:
            try:
                st = os.stat(unescape(line.split()[0]))
            except (OSError, IndexError):
                continue
            if stat.S_ISBLK(st.st_mode):
                swaps.add(f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}")
    return swaps


def sysfs(kname: str, rel: str) -> str | None:
    try:
        with open(f"/sys/class/block/{kname}/{rel}") as f:
            return f.read().strip()
    except OSError:
        return None


def holders(kname: str) -> list[str]:
    try:
        return os.listdir(f"/sys/class/block/{kname}/holders")
    except OSError:
        return []


def host_info() -> dict:
    mem_bytes = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_bytes = int(line.split()[1]) * KiB
                break
    return {"kernel": os.uname().release, "arch": os.uname().machine, "nproc": NPROC, "mem_bytes": mem_bytes}


def fio_version() -> str:
    return run(["fio", "--version"]).stdout.strip()


def lsblk_version() -> str:
    return run(["lsblk", "-V"]).stdout.strip().split()[-1]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def descendants(node: dict):
    for child in node.get("children") or []:
        yield child
        yield from descendants(child)


def clean_str(value):
    if isinstance(value, str):
        value = value.strip()
    return value or None


def to_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def exclusion_reason(dev: dict) -> str | None:
    if dev.get("type") != "disk":
        return f"type={dev.get('type')}"
    if VIRTUAL_NAME_RE.match(dev["name"]):
        return "virtual device"
    if dev.get("rm"):
        return "removable"
    if not dev.get("size"):
        return "zero size"
    if dev.get("subsystems") in (None, "", "block"):
        return f"virtual device (subsystems={dev.get('subsystems')})"
    if (dev.get("zoned") or "none") != "none":
        return f"zoned={dev.get('zoned')}"
    return None


def inventory_entry(dev: dict) -> dict:
    return {
        "path": dev.get("path"),
        "model": clean_str(dev.get("model")),
        "vendor": clean_str(dev.get("vendor")),
        "serial": clean_str(dev.get("serial")),
        "rev": clean_str(dev.get("rev")),
        "size": dev.get("size"),
        "subsystems": dev.get("subsystems"),
        "tran": clean_str(dev.get("tran")),
        "rota": dev.get("rota"),
        "ro": dev.get("ro"),
        "log_sec": to_int(dev.get("log-sec")),
        "phy_sec": to_int(dev.get("phy-sec")),
        "min_io": to_int(dev.get("min-io")),
        "opt_io": to_int(dev.get("opt-io")),
        "mq": to_int(dev.get("mq")),
        "rq_size": to_int(dev.get("rq-size")),
        "sched": clean_str(dev.get("sched")),
        "disc_max": to_int(dev.get("disc-max")),
        "write_cache": sysfs(dev["kname"], "queue/write_cache"),
        "max_sectors_kb": to_int(sysfs(dev["kname"], "queue/max_sectors_kb")),
    }


def device_usage(dev: dict, mounts: dict, swaps: set) -> tuple[list[dict], bool, list[str]]:
    nodes = [dev, *descendants(dev)]
    dev_mounts = [m for n in nodes for m in mounts.get(n["maj:min"], [])]
    swap = any(n["maj:min"] in swaps for n in nodes)
    held_by = sorted({h for n in nodes if n.get("type") in ("disk", "part") for h in holders(n["kname"])})
    return dev_mounts, swap, held_by


def is_protected(mountpoint: str) -> bool:
    if mountpoint in PROTECTED_MOUNTS or mountpoint.startswith(PROTECTED_PREFIXES):
        return True
    return DOCKER_ROOT == mountpoint or DOCKER_ROOT.startswith(mountpoint.rstrip("/") + "/")


def release_mounts(dev_mounts: list[dict]) -> tuple[bool, list[str], str | None]:
    """Unmount deepest-first without -l/-f; the kernel's EBUSY is the in-use check. Roll back on failure."""
    done = []
    for m in sorted(dev_mounts, key=lambda m: m["mountpoint"].count("/"), reverse=True):
        proc = run(host_cmd(["umount", m["mountpoint"]]), timeout=60)
        if proc.returncode != 0:
            for d in reversed(done):
                run(host_cmd(["mount", "-t", d["fstype"], "-o", d["options"], d["source"], d["mountpoint"]]), timeout=60)
            return False, [], f"umount {m['mountpoint']} failed: {proc.stderr.strip()}"
        log(f"unmounted {m['mountpoint']} ({m['source']})")
        done.append(m)
    return True, [m["mountpoint"] for m in done], None


def decide_mode(dev: dict, mounts: dict, swaps: set) -> dict:
    dev_mounts, swap, held_by = device_usage(dev, mounts, swaps)
    usage = {"mountpoints": sorted({m["mountpoint"] for m in dev_mounts}), "swap": swap, "holders": held_by}
    if dev.get("ro"):
        return usage | {"mode": "read_only", "mode_reason": "device is read-only", "unmounted": []}
    if swap:
        return usage | {"mode": "read_only", "mode_reason": "used as swap", "unmounted": []}
    if held_by:
        return usage | {"mode": "read_only", "mode_reason": f"has holders: {', '.join(held_by)}", "unmounted": []}
    if not dev_mounts:
        return usage | {"mode": "read_write", "mode_reason": "not in use", "unmounted": []}
    protected = sorted({m["mountpoint"] for m in dev_mounts if is_protected(m["mountpoint"])})
    if protected:
        return usage | {"mode": "read_only", "mode_reason": f"protected mount: {', '.join(protected)}", "unmounted": []}
    if not UMOUNT:
        return usage | {"mode": "read_only", "mode_reason": "mounted, unmounting disabled", "unmounted": []}
    ok, unmounted, error = release_mounts(dev_mounts)
    if not ok:
        return usage | {"mode": "read_only", "mode_reason": error, "unmounted": []}
    return usage | {"mode": "read_write", "mode_reason": "unmounted", "unmounted": unmounted}


def normalize_model(model: str | None) -> str | None:
    if model is None:
        return None
    for pattern, replacement in MODEL_NORMALIZERS:
        if pattern.match(model):
            return replacement
    return model


def discover() -> tuple[dict, dict, dict, list]:
    lsblk = json.loads(run(["lsblk", "-J", "-b", "-O"], timeout=60).stdout)
    mounts, swaps = host_mounts(), host_swaps()
    inventory, excluded, raw = {}, {}, {}
    for dev in lsblk.get("blockdevices", []):
        reason = exclusion_reason(dev)
        if reason:
            excluded[dev["name"]] = {"reason": reason}
            continue
        raw[dev["name"]] = dev
        inventory[dev["name"]] = inventory_entry(dev) | decide_mode(dev, mounts, swaps)

    by_key = defaultdict(list)
    for name, inv in inventory.items():
        by_key[(normalize_model(inv["model"]), inv["size"], inv["subsystems"])].append(name)
    groups = {}
    for idx, (key, members) in enumerate(sorted(by_key.items(), key=lambda kv: min(kv[1]))):
        members = sorted(members)
        representative = min(members, key=lambda n: (inventory[n]["mode"] != "read_write", n))
        gid = f"g{idx}"
        groups[gid] = {
            "key": {"model": key[0], "size": key[1], "subsystems": key[2]},
            "members": members,
            "representative": representative,
            "mode": inventory[representative]["mode"],
        }
        for name in members:
            inventory[name]["group"] = gid
    return inventory, excluded, groups, raw


def still_in_use(names: list[str], raw: dict) -> str | None:
    mounts, swaps = host_mounts(), host_swaps()
    for name in names:
        dev_mounts, swap, held_by = device_usage(raw[name], mounts, swaps)
        if dev_mounts or swap or held_by:
            return f"{name} is in use again"
    return None


# ---------------------------------------------------------------------------
# test plan
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    rw: str  # fio rw: read, write, randread, randwrite, randrw
    bs: int
    qd: int  # nominal outstanding I/Os per device
    rwmixread: int | None = None
    fill: bool = False

    @property
    def writes(self) -> bool:
        return self.rw in ("write", "randwrite", "randrw")

    @property
    def sequential(self) -> bool:
        return self.rw in ("read", "write")

    @property
    def kind(self) -> str:
        if self.rw == "randrw":
            return f"randrw{self.rwmixread}"
        return {"read": "seqread", "write": "seqwrite"}.get(self.rw, self.rw)

    @property
    def base_name(self) -> str:
        return f"{self.kind}_{bs_label(self.bs)}"


@dataclass
class PlanItem:
    group: str
    scope: str  # single or group
    devices: list[str]
    spec: Spec
    numjobs: int
    iodepth: int

    @property
    def name(self) -> str:
        return f"{self.spec.base_name}_qd{self.numjobs * self.iodepth}"


def layout(spec: Spec, n_devices: int) -> tuple[int, int]:
    if spec.qd == 1 or spec.fill:
        numjobs = 1
    else:
        numjobs = max(1, min(MAX_JOBS_PER_DEVICE, NPROC // n_devices))
    iodepth = max(1, spec.qd // numjobs)
    iodepth = max(1, min(iodepth, INFLIGHT_CAP_BYTES // (spec.bs * numjobs * n_devices)))
    return numjobs, iodepth


def single_specs(inv: dict, skipped: list, gid: str) -> list[Spec]:
    log_sec = inv["log_sec"] or 512
    sizes = []
    for bs in BLOCK_SIZES:
        if bs < log_sec:
            skipped.append({"group": gid, "scope": "single", "test": f"*_{bs_label(bs)}", "reason": f"bs < log_sec ({log_sec})"})
        else:
            sizes.append(bs)
    writable = inv["mode"] == "read_write"
    specs = []
    if writable:
        specs.append(Spec("write", FILL_BS, QD_SEQ, fill=True))
    specs.append(Spec("randread", 4 * KiB, 1))
    specs += [Spec("randread", bs, QD_RANDOM) for bs in sizes]
    specs.append(Spec("read", 1 * MiB, QD_SEQ))
    if writable:
        specs += [Spec("randrw", bs, QD_RANDOM, rwmixread=70) for bs in (4 * KiB, 64 * KiB, 1 * MiB)]
        specs.append(Spec("randrw", 4 * KiB, 1, rwmixread=70))
        specs.append(Spec("randwrite", 4 * KiB, 1))
        specs += [Spec("randwrite", bs, QD_RANDOM) for bs in sizes]
    else:
        skipped.append({"group": gid, "scope": "single", "test": "write tests", "reason": f"read_only: {inv['mode_reason']}"})
    return specs


def build_plan(inventory: dict, groups: dict, skipped: list) -> list[PlanItem]:
    plan = []
    for gid, group in groups.items():
        rep = group["representative"]
        for spec in single_specs(inventory[rep], skipped, gid):
            plan.append(PlanItem(gid, "single", [rep], spec, *layout(spec, 1)))
        members = group["members"]
        if len(members) < 2:
            continue
        writable = [n for n in members if inventory[n]["mode"] == "read_write"]
        group_specs = []
        if len(writable) >= 2:
            group_specs.append((writable, Spec("write", FILL_BS, QD_SEQ, fill=True)))
        group_specs.append((members, Spec("randread", 4 * KiB, QD_RANDOM)))
        group_specs.append((members, Spec("read", 1 * MiB, QD_SEQ)))
        if len(writable) >= 2:
            group_specs.append((writable, Spec("randrw", 4 * KiB, QD_RANDOM, rwmixread=70)))
            group_specs.append((writable, Spec("randwrite", 4 * KiB, QD_RANDOM)))
        else:
            skipped.append({"group": gid, "scope": "group", "test": "write tests", "reason": "fewer than 2 writable members"})
        for devices, spec in group_specs:
            plan.append(PlanItem(gid, "group", devices, spec, *layout(spec, len(devices))))
    return plan


def estimate_s(plan: list[PlanItem], runtime: int, ramp: int) -> int:
    return sum((FILL_CAP_S if item.spec.fill else runtime + ramp) + TEST_OVERHEAD_S for item in plan)


def fit_budget(plan: list[PlanItem], skipped: list) -> tuple[list[PlanItem], int]:
    runtime = RUNTIME_S
    while estimate_s(plan, runtime, RAMP_S) > BUDGET_S and runtime > MIN_RUNTIME_S:
        runtime -= 1
    if estimate_s(plan, runtime, RAMP_S) > BUDGET_S:
        kept = []
        for item in plan:
            if item.spec.base_name in LOW_PRIORITY_TESTS:
                skipped.append({"group": item.group, "scope": item.scope, "test": item.name, "reason": "time budget"})
            else:
                kept.append(item)
        plan = kept
    return plan, runtime


# ---------------------------------------------------------------------------
# fio execution and result merging
# ---------------------------------------------------------------------------


def job_file(item: PlanItem, ws: dict[str, int], paths: dict[str, str], logdir: str, runtime: int) -> str:
    spec = item.spec
    lines = [
        "[global]",
        "ioengine=libaio",
        "direct=1",
        "invalidate=1",
        "norandommap=1",
        "randrepeat=0",
        "random_generator=tausworthe64",
        "lat_percentiles=1",
        "clat_percentiles=0",
        f"percentile_list={':'.join(str(p) for p in PERCENTILES)}",
        f"rw={spec.rw}",
        f"bs={spec.bs}",
        f"numjobs={item.numjobs}",
        f"iodepth={item.iodepth}",
        f"iodepth_batch_submit={item.iodepth}",
        f"iodepth_batch_complete_max={item.iodepth}",
        f"log_avg_msec={LOG_AVG_MSEC}",
        "allow_mounted_write=0",
        "exitall_on_error=1",
    ]
    if spec.rwmixread is not None:
        lines.append(f"rwmixread={spec.rwmixread}")
    if spec.fill:
        lines.append(f"runtime={FILL_CAP_S}")
    else:
        lines += ["time_based=1", f"runtime={runtime}", f"ramp_time={RAMP_S}"]
    for dev in item.devices:
        lines += ["", f"[{dev}]", f"filename={paths[dev]}", f"write_iops_log={logdir}/{dev}", f"write_bw_log={logdir}/{dev}"]
        if spec.sequential and item.numjobs > 1:
            stripe = align_down(ws[dev] // item.numjobs, spec.bs)
            lines += [f"size={stripe}", f"offset_increment={stripe}"]
        else:
            lines.append(f"size={align_down(ws[dev], spec.bs)}")
    return "\n".join(lines) + "\n"


def percentile_of(sorted_values: list[float], p: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def merge_latency(lats: list[dict]) -> dict | None:
    lats = [lat for lat in lats if lat.get("N")]
    if not lats:
        return None
    n = sum(lat["N"] for lat in lats)
    mean = sum(lat["N"] * lat["mean"] for lat in lats) / n
    second_moment = sum(lat["N"] * (lat["stddev"] ** 2 + lat["mean"] ** 2) for lat in lats) / n
    bins = defaultdict(int)
    for lat in lats:
        for value, count in (lat.get("bins") or {}).items():
            bins[int(value)] += count
    percentiles = {}
    if bins:
        values = sorted(bins)
        cumulative = list(itertools.accumulate(bins[v] for v in values))
        total = cumulative[-1]
        for p in PERCENTILES:
            idx = min(len(values) - 1, bisect.bisect_left(cumulative, p / 100 * total))
            percentiles[str(p)] = values[idx]
    return {
        "min": min(lat["min"] for lat in lats),
        "max": max(lat["max"] for lat in lats),
        "mean": mean,
        "stddev": max(0.0, second_moment - mean**2) ** 0.5,
        "samples": n,
        "percentiles": percentiles,
    }


def interval_stats(logfiles: list[str], ddir: int, scale: float) -> dict | None:
    """Aggregate per-window samples across jobs; only windows where every job reported are kept."""
    buckets = defaultdict(lambda: [0.0, 0])
    for path in logfiles:
        with open(path) as f:
            for line in f:
                parts = line.split(",")
                if len(parts) < 3 or int(parts[2]) != ddir:
                    continue
                bucket = buckets[round(int(parts[0]) / LOG_AVG_MSEC)]
                bucket[0] += float(parts[1]) * scale
                bucket[1] += 1
    complete = sorted(total for total, count in buckets.values() if count == len(logfiles))
    values = complete or sorted(total for total, _ in buckets.values())
    if not values:
        return None
    return {
        "min": values[0],
        "p5": percentile_of(values, 5),
        "p50": percentile_of(values, 50),
        "p95": percentile_of(values, 95),
        "max": values[-1],
        "mean": sum(values) / len(values),
        "samples": len(values),
    }


def merge_direction(jobs: list[dict], ddir: str, logdir: str, devices: list[str]) -> dict | None:
    parts = [job[ddir] for job in jobs if job.get(ddir, {}).get("total_ios")]
    if not parts:
        return None
    code = {"read": 0, "write": 1}[ddir]
    iops_logs = [p for dev in devices for p in glob.glob(f"{logdir}/{dev}_iops.*.log")]
    bw_logs = [p for dev in devices for p in glob.glob(f"{logdir}/{dev}_bw.*.log")]
    return {
        "iops": sum(p["iops"] for p in parts),
        "bw_bytes": sum(p["bw_bytes"] for p in parts),
        "io_bytes": sum(p["io_bytes"] for p in parts),
        "total_ios": sum(p["total_ios"] for p in parts),
        "runtime_ms": max(p["runtime"] for p in parts),
        "iops_interval": interval_stats(iops_logs, code, 1.0),
        "bw_bytes_interval": interval_stats(bw_logs, code, float(KiB)),
        "lat_ns": merge_latency([p["lat_ns"] for p in parts]),
    }


def parse_fio_output(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    return json.loads(text[text.index("{"):])


def run_item(item: PlanItem, ws: dict[str, int], paths: dict[str, str], runtime: int) -> dict:
    spec = item.spec
    record = {
        "group": item.group,
        "scope": item.scope,
        "devices": item.devices,
        "test": item.name,
        "rw": spec.rw,
        "bs": spec.bs,
        "rwmixread": spec.rwmixread,
        "numjobs": item.numjobs,
        "iodepth": item.iodepth,
        "fill": spec.fill,
        "runtime_s": FILL_CAP_S if spec.fill else runtime,
        "ramp_time_s": 0 if spec.fill else RAMP_S,
        "working_set_bytes": {dev: ws[dev] for dev in item.devices},
        "error": None,
    }
    with tempfile.TemporaryDirectory(prefix="fio-") as tmp:
        jobpath, outpath = f"{tmp}/job.fio", f"{tmp}/out.json"
        with open(jobpath, "w") as f:
            f.write(job_file(item, ws, paths, tmp, runtime))
        cmd = ["fio", "--eta=never", "--output-format=json+", f"--output={outpath}"]
        if not spec.writes:
            cmd.append("--readonly")
        cmd.append(jobpath)
        timeout = (FILL_CAP_S if spec.fill else runtime + RAMP_S) + 120
        started = time.monotonic()
        try:
            proc = run(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            return record | {"error": f"fio timed out after {timeout}s"}
        record["elapsed_s"] = round(time.monotonic() - started, 1)
        if proc.returncode != 0:
            return record | {"error": f"fio exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-2000:]}"}
        try:
            data = parse_fio_output(outpath)
        except (OSError, ValueError) as e:
            return record | {"error": f"cannot parse fio output: {e}"}

        jobs = data.get("jobs", [])
        errors = sorted({job["error"] for job in jobs if job.get("error")})
        if errors:
            record["error"] = f"fio job errors: {errors}"
        for ddir in ("read", "write"):
            record[ddir] = merge_direction(jobs, ddir, tmp, item.devices)
        if item.scope == "group":
            record["per_device"] = {
                dev: {
                    ddir: merge_direction([j for j in jobs if j["jobname"] == dev], ddir, tmp, [dev])
                    for ddir in ("read", "write")
                }
                for dev in item.devices
            }
        if spec.fill:
            record["written_bytes"] = {
                dev: sum(j["write"]["io_bytes"] for j in jobs if j["jobname"] == dev) for dev in item.devices
            }
        record["cpu"] = {
            "usr": sum(job.get("usr_cpu", 0) for job in jobs),
            "sys": sum(job.get("sys_cpu", 0) for job in jobs),
        }
        record["disk_util"] = {
            du["name"]: {k: du.get(k) for k in ("util", "read_ios", "write_ios", "in_queue")}
            for du in data.get("disk_util", [])
        }
    return record


def execute(plan: list[PlanItem], inventory: dict, groups: dict, raw: dict, runtime: int, skipped: list) -> list[dict]:
    paths = {name: inv["path"] for name, inv in inventory.items()}
    # working set per device: whole device until a fill defines the written region
    ws = {name: align_down(inv["size"], FILL_BS) for name, inv in inventory.items()}
    failed_fill = set()
    results = []
    for idx, item in enumerate(plan, 1):
        spec = item.spec
        # an unfilled working set would make reads hit unwritten blocks
        if (item.scope == "single" or spec.writes) and failed_fill.intersection(item.devices):
            skipped.append({"group": item.group, "scope": item.scope, "test": item.name, "reason": "fill failed"})
            continue
        if spec.writes:
            reason = still_in_use(item.devices, raw)
            if reason:
                skipped.append({"group": item.group, "scope": item.scope, "test": item.name, "reason": reason})
                continue
        if spec.fill:
            target = WS_TARGET_BYTES
            if item.scope == "group":
                target = ws[groups[item.group]["representative"]]
            for dev in item.devices:
                ws[dev] = align_down(min(inventory[dev]["size"], target), FILL_BS)
        log(f"[{idx}/{len(plan)}] {item.scope} {item.name} on {','.join(item.devices)}")
        record = run_item(item, ws, paths, runtime)
        if spec.fill:
            written = record.get("written_bytes") or {}
            for dev in item.devices:
                filled = align_down(written.get(dev, 0), FILL_BS)
                if record["error"] or filled == 0:
                    failed_fill.add(dev)
                else:
                    ws[dev] = filled
        if record["error"]:
            log(f"  error: {record['error']}")
        else:
            summary = []
            for ddir in ("read", "write"):
                if record.get(ddir):
                    d = record[ddir]
                    p99 = (d["lat_ns"] or {}).get("percentiles", {}).get("99")
                    summary.append(f"{ddir} {d['iops']:.0f} IOPS {d['bw_bytes'] / MiB:.1f} MiB/s p99 {p99 / 1000 if p99 else 0:.0f}us")
            log("  " + "; ".join(summary))
        results.append(record)
    return results


def main() -> int:
    if "--version" in sys.argv[1:]:
        print(fio_version())
        return 0
    started_at = now_iso()
    output = {
        "schema_version": SCHEMA_VERSION,
        "fio_version": fio_version(),
        "lsblk_version": lsblk_version(),
        "started_at": started_at,
        "host": host_info(),
    }
    inventory, excluded, groups, raw = discover()
    skipped: list[dict] = []
    plan = build_plan(inventory, groups, skipped)
    plan, runtime = fit_budget(plan, skipped)
    output["config"] = {
        "ioengine": "libaio",
        "ramp_time_s": RAMP_S,
        "runtime_s": runtime,
        "fill_cap_s": FILL_CAP_S,
        "ws_target_bytes": WS_TARGET_BYTES,
        "budget_s": BUDGET_S,
        "log_avg_msec": LOG_AVG_MSEC,
        "inflight_cap_bytes": INFLIGHT_CAP_BYTES,
        "percentiles": PERCENTILES,
        "estimated_s": estimate_s(plan, runtime, RAMP_S),
    }
    output["inventory"] = inventory
    output["excluded"] = excluded
    output["groups"] = groups
    log(f"devices: {', '.join(f'{n} ({i['mode']})' for n, i in inventory.items())}; excluded: {', '.join(excluded) or '-'}")
    log(f"plan: {len(plan)} tests, runtime {runtime}s, estimated {output['config']['estimated_s']}s")
    output["benchmarks"] = execute(plan, inventory, groups, raw, runtime, skipped)
    output["skipped"] = skipped
    output["finished_at"] = now_iso()
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
