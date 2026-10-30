# Usage

## Disk Safety

This benchmark decides for each disk whether it may write to it. Disks it may
write to (`read_write`) are filled and overwritten, and discarded first if they
support it. Their previous data is lost. Run it only on ephemeral servers,
never on production or similar systems.

[`benchmark.py`](../benchmark.py) checks each disk in the following order:

1. A disk that is read-only, used as swap, or held by another device is
   `read_only`. Holders include LVM (Logical Volume Manager) volumes and RAID
   (Redundant Array of Independent Disks) arrays built with `md`.
2. A disk with no mounted filesystem is `read_write`.
3. A disk with a protected mount is `read_only`. The protected mounts are `/`,
   `/boot`, `/boot/efi`, `/usr`, `/var`, `/home`, `/root`, `/etc`, `/opt`,
   `/srv`, `/tmp`, anything under `/snap/`, and any mount at or above
   `/var/lib/docker`.
4. With `SC_STORAGE_UMOUNT=0`, any other mounted disk is `read_only`.
5. Otherwise, this benchmark unmounts the disk's filesystems, deepest first,
   without forcing the unmount. If any unmount fails, for example because a
   filesystem is busy, it remounts what it already unmounted and marks the
   disk `read_only`. Otherwise, the disk is `read_write`.

This benchmark never remounts a filesystem after unmounting it, and the write
tests overwrite its data.

Before every write test, this benchmark checks the disks again and skips the
test if a disk is mounted, used as swap, or held again. The `fio` tool also runs with
`allow_mounted_write=0`, and read tests run with `fio --readonly`.

The mount checks read the host's mount table from `/proc/1/mountinfo` and
unmount through `nsenter` in the host's mount namespace. Both need
`--pid=host`. This benchmark does not check if it runs in the host's PID
(Process Identifier) namespace.

Note that `SC_STORAGE_UMOUNT=0` does not protect the disks that step 2 marks
`read_write`: this benchmark always writes to them. There is no setting for a
read-only run.

## Production Setup

Production runs the container with the following settings:

- `--privileged`, `--pid=host`, and host networking
- a timeout of 1 hour
- after the benchmarks that have a set priority, because this benchmark may
  unmount and overwrite free disks
- Resource Tracker settings, so the tracker that wraps this benchmark
  [streams](https://sparecores.com/article/resource-tracker-rs#streaming-to-sentinel)
  the run's resource usage to the
  [Spare Cores Sentinel](https://sentinel.sparecores.com/) platform

A plain `docker run` with `--privileged --pid=host` runs the same tests, but
without host networking and without the Resource Tracker settings.

Production runs this benchmark on ephemeral servers, on every server type with
bundled storage: the
[`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
file in `sc-crawler` skips only server types without it. The image does not
check whether it runs on an ephemeral server, and does not tell bundled disks
apart from other disks.

## Run Duration

A run took about 21 minutes on GCP (Google Cloud Platform)
`c4a-standard-16-lssd`, with four local NVMe (Non-Volatile Memory Express)
drives, and about 22 minutes on OVHcloud `i1-90`, with two NVMe drives. See
[`gcp/c4a-standard-16-lssd`](https://github.com/SpareCores/sc-inspector-data/tree/main/data/gcp/c4a-standard-16-lssd/0/storage)
and
[`ovh/i1-90`](https://github.com/SpareCores/sc-inspector-data/tree/main/data/ovh/i1-90/0/storage)
for the run outputs.

The following settings make up most of the time:

- each read test: 10s measured plus a 2s ramp
- each write or mixed test: 30s measured plus a 2s ramp
- the sequential fill: up to 300s in total, split evenly between the
  groups that have writable disks
- the random-write precondition: 120s per group with writable disks

Before running, this benchmark estimates the total test time, counting 2 seconds
of `fio` overhead per test. If the estimate exceeds `SC_STORAGE_BUDGET_S`
(2400s), it takes the following steps until the plan fits:

1. Shorten read tests, down to `SC_STORAGE_MIN_RUNTIME_S` (5s).
2. Shorten write tests, down to `SC_STORAGE_MIN_WRITE_RUNTIME_S` (15s).
3. Drop the low-priority tests: `randread_512`, `randwrite_512`,
   `randread_16m`, `randwrite_16m`, `randrw70_64k`, and `randrw70_1m`.

Dropped tests are listed in `skipped` with the reason `time budget`. The
estimate leaves out the discard and the health snapshots, so a run can take
longer than the estimate. The `config` field records the runtimes actually used
and the estimate.

## Output

This benchmark prints one JSON document to stdout. It always exits with code 0
once the run completes; a failed test records its error in its own `error`
field. Progress goes to stderr, one timestamped line per step, with per-test
IOPS (Input/Output Operations Per Second), MiB/s, and 99th-percentile latency
in microseconds.

### Top-Level Fields

The JSON document has the following top-level fields:

- `schema_version` (`1`), and the versions of `fio`, `lsblk`, `smartctl` (drive
  health queries), and `nvme-cli` (NVMe health logs)
- `started_at` and `finished_at`, as UTC (Coordinated Universal Time)
  timestamps
- `host`: kernel, architecture, CPU count (`nproc`), and memory (`mem_bytes`)
- `config`: the timing and budget settings used, including `runtime_s` and
  `write_runtime_s` after budget fitting, and `estimated_s`
- `inventory`, `excluded`, and `groups`: the disks found and how they were
  grouped
- `benchmarks` and `preconditioning`: the test records
- `health`: the disk health snapshots
- `skipped`: the tests that did not run

### Disks and Groups

Each `inventory` entry describes one disk, with `lsblk` and `sysfs` fields such
as `model`, `serial`, `size` (bytes), `tran` (transport), `rota` (rotational),
`log_sec` and `phy_sec` (sector sizes), `write_cache`, and `disc_max` (discard
support). It also records how the disk was used and the decision made for it,
in the following fields:

- `mountpoints`, `swap`, and `holders`
- `mode` (`read_write` or `read_only`) and `mode_reason`
- `unmounted`: the filesystems this benchmark unmounted
- `group`: the ID of the disk's group

The `excluded` field lists the devices that are not tested, each with a reason.
The following devices are excluded:

- devices that are not disks
- virtual devices, such as `loop`, `zram`, `md`, or `dm-`
- removable devices
- empty devices
- zoned devices

Disks with the same model, size, and subsystem form a group. Each `groups`
entry has its `key`, `members`, `representative` (the first writable member by
name, or the first member if none is writable), and `mode`.

### Test Records

Each `benchmarks` and `preconditioning` entry is one `fio` run, with the
following fields:

- `test`: `<kind>_<block size>_qd<queue depth>`. The kinds are `randread`,
  `seqread`, `seqwrite`, `randwrite`, `randrw70` (70% reads), `fill`, and
  `precondition_randwrite`. The queue depth is the effective number of I/O
  (Input/Output) requests in flight per disk, `numjobs` times `iodepth`, which
  can be lower than the nominal queue depth. See
  [`fio` Settings](./workloads.md#fio-settings).
- `group`, `scope` (`single` or `group`), and `devices`
- `rw`, `bs` (bytes), `rwmixread`, `numjobs`, and `iodepth`
- `runtime_s`, `ramp_time_s`, `elapsed_s`, and `working_set_bytes`
- `error`: `null`, or the `fio` error
- `read` and `write`: the results per direction, or `null` if the test did not
  use that direction
- `cpu`: the summed user and system CPU percentages of the `fio` jobs
- `disk_util`: the kernel's disk utilization statistics

Group tests also have `per_device`, with `read` and `write` results for each
disk. Fill records also have `written_bytes`, `purge` (the discard result), and
`filled_fraction`: the share of each disk the fill wrote, to four decimal
places.

Each `read` and `write` result has the following fields:

- `iops`, `bw_bytes` (bytes per second), `io_bytes`, `total_ios`, and
  `runtime_ms`
- `lat_ns`: end-to-end latency per I/O in nanoseconds, with `min`, `max`,
  `mean`, `stddev`, `samples`, and `percentiles` at 1, 5, 10, 25, 50, 75, 90,
  95, 99, 99.9, and 99.99, keyed as strings such as `"99"`
- `iops_interval` and `bw_bytes_interval`: statistics over 500 ms windows,
  with `min`, `p5`, `p50`, `p95`, `max`, `mean`, `tail_mean`, and `samples`

The `tail_mean` field is the mean of the second half of the run. It leaves out
the initial burst and includes the time a drive may still spend draining its
write buffer. Multi-job and multi-disk results are merged: IOPS and bandwidth
are summed, and latency percentiles come from the merged `fio` latency
histograms. Preconditioning records also have `iops_series` and
`bw_bytes_series`, averaged over steps of 5 seconds (`series_step_ms`).

### Health Snapshots

The `health` field has one entry per disk, with the following fields:

- `before` and `after`: snapshots of `smartctl -x` and, for NVMe drives,
  `nvme smart-log`, each with `taken_at`
- `nvme_id_ctrl` and `nvme_id_ns`: the NVMe controller and namespace
  identification, before the run
- `delta`: for NVMe drives, the change in counters over the run, such as
  `bytes_read`, `bytes_written`, `media_errors`, and `controller_busy_time`,
  plus `percent_used_after`, `avail_spare_after`, and `critical_warning_after`

Health data is best effort. Virtual disks often refuse these queries; a failed
query is recorded as `{"error": "..."}` and does not fail the run.

### Skipped Tests

Each `skipped` entry has `group`, `scope`, `test`, and `reason`, with one of
the following reasons:

- `bs < log_sec (<n>)`: the block size is below the disk's logical sector size
- `read_only: <mode_reason>`: write tests on a disk this benchmark may not
  write to
- `fewer than 2 writable members`: group write tests in a group with fewer
  than two writable disks
- `time budget`: dropped to fit the time budget
- `fill failed`: every later test on a disk whose fill failed
- `<disk> is in use again`: a write test on a disk that was mounted, used as
  swap, or held again

## Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `SC_STORAGE_RUNTIME_S` | Measured seconds per read test. | `10` |
| `SC_STORAGE_WRITE_RUNTIME_S` | Measured seconds per write or mixed test. | `30` |
| `SC_STORAGE_RAMP_S` | Ramp seconds per test, not counted in the results; not used for the fill and the random-write precondition. | `2` |
| `SC_STORAGE_MIN_RUNTIME_S` | Shortest read test when fitting the time budget. | `5` |
| `SC_STORAGE_MIN_WRITE_RUNTIME_S` | Shortest write test when fitting the time budget. | `15` |
| `SC_STORAGE_FILL_BUDGET_S` | Total sequential-fill seconds, split evenly between fill steps. | `300` |
| `SC_STORAGE_WS_TARGET_BYTES` | Bytes to fill per disk; `0` fills the whole disk. | `0` |
| `SC_STORAGE_RAND_PRECOND_S` | Random-write precondition seconds. | `120` |
| `SC_STORAGE_BUDGET_S` | Target for the estimated total test time in seconds. | `2400` |
| `SC_STORAGE_UMOUNT` | `0` keeps every mounted disk read-only instead of unmounting it. | `1` |
| `SC_STORAGE_PURGE` | `0` skips the discard before the fill. | `1` |

An empty value uses the default. The `SC_STORAGE_UMOUNT` and `SC_STORAGE_PURGE`
settings are off only when set to exactly `0`.
