# Workloads

## Disk Discovery and Grouping

[`benchmark.py`](../benchmark.py) lists the host's block devices with
`lsblk`. It skips devices that are not disks, virtual devices (names starting
with `zram`, `ram`, `loop`, `nbd`, `md`, `dm-`, `sr`, or `fd`, or with no real
subsystem), and removable, empty, or zoned devices.

Disks with the same model, size, and subsystem form a group. One
representative per group, the first writable disk by name, runs the full set
of single-disk tests. Groups of two or more disks also run group tests on all
their disks at once, which measure the combined throughput. Each disk is
either written to (`read_write`) or only read (`read_only`), as described in
[Disk Safety](./usage.md#disk-safety).

The groups run one after the other, and so do the tests within a group. Each
group runs its steps in the following order:

1. Sequential fill of the writable disks (preconditioning)
2. Single-disk read tests
3. Group read tests
4. Sequential write tests, single-disk and then group
5. Random-write precondition of the writable disks
6. Single-disk random-write and mixed tests
7. Group random-write and mixed tests

Read tests run on the freshly filled disk. Random writes run last, because the
random-write precondition fragments the SSD's FTL (Flash Translation Layer)
mapping, and a sequential test that ran after it would inherit that state.

A group whose disks are all read-only runs only steps 2 and 3, on whatever data
the disks already hold.

## Preconditioning

SSDs absorb writes in their own DRAM (Dynamic Random-Access Memory) or SLC
(Single-Level Cell) caches and in unmapped spare area. Writes to a partly
written drive land on spare area and never trigger garbage collection, so
short tests on such a drive report burst speeds rather than sustained ones.
This benchmark prepares each writable disk in the following steps:

1. **Discard:** `blkdiscard` trims the whole disk, so every run starts from the
   same mapping state. Disks without discard support skip this step, and
   `SC_STORAGE_PURGE=0` turns it off.
2. **Sequential fill:** one `fio` job per disk writes it sequentially in
   1&nbsp;MiB blocks at a queue depth of 64, until the disk is full or the fill
   time runs out. The fill time is `SC_STORAGE_FILL_BUDGET_S` (300&nbsp;s),
   split evenly between the groups with writable disks.
3. **Random-write precondition:** after the read and sequential write tests,
   4&nbsp;KiB random writes at a queue depth of 256 run for
   `SC_STORAGE_RAND_PRECOND_S` (120&nbsp;s).

Both preconditioning steps are reported under `preconditioning`, not
`benchmarks`, with IOPS (Input/Output Operations Per Second) and bandwidth time
series in 5&nbsp;s steps.

After the fill, every later test on the disk uses only the region the fill
wrote. If the fill fails or writes nothing, every later test on that disk is
skipped, because reads would hit unmapped blocks. Read-only disks always use
the whole disk.

Write and mixed tests run for 30&nbsp;s, compared with 10&nbsp;s for read tests,
to get past the drive's DRAM or SLC write buffer.

## Single-Disk Tests

The representative of each group runs the following read tests:

- random reads of 4&nbsp;KiB at a queue depth of 1
- random reads at a queue depth of 256, at each block size: 512&nbsp;B,
  4&nbsp;KiB, 16&nbsp;KiB, 64&nbsp;KiB, 256&nbsp;KiB, 1&nbsp;MiB, 4&nbsp;MiB,
  and 16&nbsp;MiB
- sequential reads of 1&nbsp;MiB at a queue depth of 64

If the representative is writable, it also runs the following write tests:

- sequential writes of 1&nbsp;MiB at a queue depth of 64
- mixed random I/O (Input/Output) with 70% reads at a queue depth of 256, at
  4&nbsp;KiB, 64&nbsp;KiB, and 1&nbsp;MiB
- mixed random I/O with 70% reads, at 4 KiB and a queue depth of 1
- random writes of 4 KiB at a queue depth of 1
- random writes at a queue depth of 256, at each block size from 512&nbsp;B to
  16&nbsp;MiB

Block sizes below the disk's logical sector size are skipped. For example, a
disk with 4&nbsp;KiB sectors skips the 512&nbsp;B tests.

## Group Tests

A group of two or more disks runs the following read tests on all its disks at
once, including read-only ones:

- random reads of 4&nbsp;KiB at a queue depth of 256
- sequential reads of 1&nbsp;MiB at a queue depth of 64

If at least two disks of the group are writable, those disks also run
sequential writes of 1&nbsp;MiB, mixed random I/O with 70% reads at 4&nbsp;KiB,
and random writes of 4&nbsp;KiB, all at the same queue depths. Group results
add up all disks, and `per_device` holds each disk's share.

## `fio` Settings

All tests use the `libaio` I/O engine with `direct=1`, so I/O bypasses the page
cache. Each test runs for a fixed time (`time_based=1`) after a 2&nbsp;s ramp.
The sequential fill instead runs until the disk is full or its time runs out,
and neither preconditioning step has a ramp.

The queue depths above are the nominal number of I/Os in flight per disk.
[`benchmark.py`](../benchmark.py) splits them across jobs as follows:

- Tests with a queue depth of 1 and the fill use one job per disk. Other tests
  use up to four jobs per disk, and no more than the CPU count divided by the
  number of disks in the test.
- Each job's `iodepth` is the queue depth divided by the number of jobs.
- `iodepth` is then capped so that no test has more than 256&nbsp;MiB in flight
  across all its jobs and disks.

Because of this cap, large block sizes run at a lower queue depth. The test
name shows the effective queue depth. For example, on a 16&nbsp;vCPU server,
4&nbsp;MiB random reads run as `randread_4m_qd64` and 16&nbsp;MiB random reads
as `randread_16m_qd16`.

Sequential tests with more than one job split the disk into equal stripes, one
per job. Random tests use the `tausworthe64` generator without a random map
(`norandommap=1`) and without a repeatable seed (`randrepeat=0`).

`fio` measures the end-to-end latency of each I/O, and this benchmark reports
its percentiles from `fio`'s latency histograms. It also logs IOPS and
bandwidth every 500&nbsp;ms, which give the interval statistics in the output.
See [Output](./usage.md#output) for the fields.
