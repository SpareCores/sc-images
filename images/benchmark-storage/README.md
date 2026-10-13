# benchmark-storage

This benchmark measures the raw block device performance of a cloud server's
bundled disks with [`fio`](https://fio.readthedocs.io/en/latest/). It finds the
host's disks, groups identical ones, and runs random, sequential, and mixed
read and write tests on one disk per group. Groups of two or more disks also
run a few tests on all their disks at once. All I/O (Input/Output) goes
directly to the raw device, bypassing the filesystem and the page cache.

This benchmark reports IOPS (Input/Output Operations Per Second), bandwidth,
and latency percentiles for each test, at block sizes from 512&nbsp;B to
16&nbsp;MiB. It also records a disk inventory and before-and-after health
snapshots. It overwrites every disk it can claim, so run it only on ephemeral
servers.

## Purpose

Our [Navigator](https://sparecores.com/servers) project publishes empirical
performance measurements for over 5,000 cloud server types, such as raw CPU
speed, memory bandwidth, and LLM (Large Language Model) inference speed. This
benchmark adds storage performance: Navigator will publish bandwidth and IOPS
at different block sizes, both per disk and for each group of identical disks.

This benchmark focuses on a server type's bundled storage: the disk or disks
provisioned with the server type by default, regardless of whether HDDs (Hard
Disk Drives), SSDs, or network-attached storage back them. Results reflect the
storage as provisioned, including any provider-imposed IOPS or bandwidth limit.
Network storage attached on demand is out of scope, because its performance is
not a property of the server type. See
[Disk Scope](./docs/limitations.md#disk-scope).

SSDs absorb short bursts of writes in their own caches and spare area. To make
results comparable across vendors, this benchmark preconditions each writable
disk before measuring it, and runs write tests for longer than read tests. See
[Preconditioning](./docs/workloads.md#preconditioning) for the steps.

## Limitations

These results describe raw block devices under synthetic `fio` load, not
filesystem or application performance. The main limitations are:

- **Destructive:** This benchmark discards and overwrites every disk that is
  not in use, and unmounts and overwrites disks whose mounts are not protected.
  It never remounts them. See [Disk Safety](./docs/usage.md#disk-safety).
- **Raw devices only:** Tests bypass the filesystem and the page cache, so
  results exclude filesystem overhead and caching. See
  [Raw Block I/O](./docs/limitations.md#raw-block-io).
- **Bundled storage is not detected:** The image tests every real block device
  it finds, including any volume attached on demand. See
  [Disk Scope](./docs/limitations.md#disk-scope).
- **Partial preconditioning:** The sequential fill is capped at 300&nbsp;s in
  total, which filled 26-33% of each NVMe
  ([Non-Volatile Memory Express](https://nvmexpress.org/)) drive on the first
  production servers. See [Partial Fill](./docs/limitations.md#partial-fill).
- **Disks in use are read only:** The boot disk and other disks in use get
  read tests only, on whatever data they already hold. See
  [Read-Only Disks](./docs/limitations.md#read-only-disks).
- **No single score:** The output has one record per test, and no headline
  metric is defined yet. See [Results](#results).

## Usage

Run the image via Docker on an ephemeral server that you can afford to wipe:

```bash
docker run --rm --privileged --pid=host ghcr.io/sparecores/benchmark-storage:main > storage.json
```

**Warning:** Every filesystem this benchmark unmounts is destroyed, together
with the data on every unmounted disk. Never run it on production or similar
systems. Set `SC_STORAGE_UMOUNT=0` to leave mounted disks alone; disks that are
not mounted are still overwritten. See
[Disk Safety](./docs/usage.md#disk-safety) for which disks this benchmark
writes to.

Both flags are required. `--privileged` gives the container raw access to the
host's disks. `--pid=host` lets it read the host's mount table from
`/proc/1/mountinfo` and unmount filesystems in the host's mount namespace, which
the disk safety checks depend on. The image does not check if it runs in the
host's PID namespace. See
[Production Setup](./docs/usage.md#production-setup) for how production runs
differ.

A run took about 21-22 minutes on the first production servers. This benchmark
shortens or drops tests when its estimate exceeds 2400&nbsp;s. See
[Run Duration](./docs/usage.md#run-duration) for how the time is spent.

### Settings

Common settings are listed here; see the
[full environment-variable reference](./docs/usage.md#environment-variables)
for all options and defaults:

- `SC_STORAGE_UMOUNT=0` keeps every mounted disk read-only instead of
  unmounting it.
- `SC_STORAGE_PURGE=0` skips the discard (`blkdiscard`) before the fill.
- `SC_STORAGE_RUNTIME_S` and `SC_STORAGE_WRITE_RUNTIME_S` set the measured
  seconds per read test (default `10`) and per write test (default `30`).
- `SC_STORAGE_FILL_BUDGET_S` sets the total sequential-fill time (default
  `300`), and `SC_STORAGE_BUDGET_S` the target for the estimated test time
  (default `2400`).

### Results

This benchmark prints one JSON document to stdout when the run ends, and a
timestamped progress log with per-test IOPS, MiB/s, and 99th-percentile latency
to stderr. Interpret the JSON document's main fields as follows:

- `inventory` lists each disk with its model, size, and transport. `mode` is
  `read_write` or `read_only`, and `mode_reason` says why.
- `groups` lists the groups of identical disks, each with its members and the
  `representative` that runs the single-disk tests.
- Each `benchmarks[]` entry is one test, named
  `<kind>_<block size>_qd<queue depth>`, for example `randread_4k_qd256`.
  `scope` is `single` or `group`.
- `read` and `write` hold `iops`, `bw_bytes` (bytes per second), and `lat_ns`,
  whose `percentiles` are latencies in nanoseconds. `iops_interval` and
  `bw_bytes_interval` summarize 500&nbsp;ms windows, including `tail_mean`, the
  mean of the second half of the run.
- `preconditioning` holds the fill and random-write steps. `filled_fraction`
  shows how much of each disk the fill wrote.
- `skipped` lists every test that did not run, with the reason.

See [Output](./docs/usage.md#output) for every field and its unit.

## Workloads

Each group of identical disks runs the following steps, in this order:

- [Preconditioning](./docs/workloads.md#preconditioning) discards and fills
  each writable disk sequentially, and later writes it randomly before the
  random-write tests.
- [Single-disk tests](./docs/workloads.md#single-disk-tests) run random reads
  and writes at eight block sizes, sequential reads and writes, and 70/30 mixed
  I/O on one representative disk.
- [Group tests](./docs/workloads.md#group-tests) run random and sequential
  reads on all disks of a group at once, and writes when at least two disks are
  writable.

For more information on test order and `fio` settings, see
[Workloads](./docs/workloads.md).

## Design History

The first version measured disks without preconditioning them. Partial fills
and short write runs measured the drives' write-buffer burst rather than their
sustained performance. The current version discards and fills each writable
disk, runs write tests for longer than read tests (30&nbsp;s compared with
10&nbsp;s), and records best-effort health snapshots, so that results are
comparable across vendors.

The fill is still capped by time. On the first production servers, it wrote
about a quarter to a third of each NVMe drive. See
[Partial Fill](./docs/limitations.md#partial-fill).

See the [changelog](./CHANGELOG.md) for the changes to this benchmark.

## References

See the full list of [references](./docs/references.md) used in this
documentation.
