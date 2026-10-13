# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### Raw Block I/O

This benchmark measures raw block devices with direct I/O (Input/Output). It
does *not* measure filesystem performance, page-cache effects, or how a given
application or database uses the disk. Results show what the disk can do under
synthetic `fio` load, not what an application will see through a filesystem.

### A Single Score

The output has one record per test and does *not* combine them into a single
number. Navigator will publish bandwidth and IOPS (Input/Output Operations Per
Second) at different block sizes, per disk and for all disks of a server
together. The rest of the output, including any headline metric, is not
decided yet.

### Other `fio` I/O Engines

All tests use the `libaio` I/O engine. The image includes `liburing`, but this
benchmark does *not* test the `io_uring` engine.

## Scope Decisions

### Disk Scope

This benchmark targets a server type's bundled storage: the disk or disks that
come with the server type by default. These can be locally attached SSDs or
HDDs (Hard Disk Drives), or a network drive behind the scenes. Network storage
attached on demand is out of scope, because its performance is not a property
of the server type.

The image does *not* detect which disks are bundled. It tests every real block
device that passes
[discovery](./workloads.md#disk-discovery-and-grouping), including any volume
attached on demand, and overwrites it if it can.

### Read-Only Disks

A disk that is in use (the boot disk, for example) is only read, never written.
Its read tests run over the whole disk, on whatever data it already holds,
without a fill, and it skips every write and mixed test. See
[Disk Safety](./usage.md#disk-safety) for how this benchmark decides.

## Design Constraints

### Partial Fill

The sequential fill runs for at most 300&nbsp;s in total
(`SC_STORAGE_FILL_BUDGET_S`), split between groups. A disk that takes longer to
write is only partly filled. On the first production runs, the fill wrote the
following share of each NVMe
([Non-Volatile Memory Express](https://nvmexpress.org/)) drive:

- GCP (Google Cloud Platform) `c4a-standard-16-lssd`: 26% of each of four
  drives
- OVHcloud `i1-90`: 33% of each of two drives

A disk that is not fully written keeps unmapped spare area, which flatters
random-write results. Check `filled_fraction` in the fill record before
comparing random-write results across disks. Setting
`SC_STORAGE_WS_TARGET_BYTES` limits the fill, and all later tests, to a smaller
region of each disk.

### Destructive Runs

This benchmark discards and overwrites every disk it may write to, and does not
remount the filesystems it unmounts. Run it only on ephemeral servers, never on
production or similar systems. `SC_STORAGE_UMOUNT=0` leaves mounted disks
alone, but disks that are not mounted are always overwritten. See
[Disk Safety](./usage.md#disk-safety).

### Time Budget

This benchmark targets an estimated 2400&nbsp;s of tests. On hosts with many
groups of disks, it shortens tests and then drops low-priority ones to fit. See
[Run Duration](./usage.md#run-duration) for the order.
