# Memory Bandwidth

## Running the Script

```shell
docker run --rm --privileged ghcr.io/sparecores/benchmark:main bw_mem.sh
```

In production, the median run takes \~4.0 minutes. See
[Run Duration](./usage.md#run-duration) for how it was measured, and
[Production Setup](./usage.md#production-setup) for how production runs differ
from a plain `docker run`.

## Workload

[`bw_mem.sh`](../bw_mem.sh) runs LMbench's `bw_mem`, from Ubuntu's `lmbench`
package, in three of its modes:

- `rd` reads data into the processor, computing the sum of an array.
- `wr` writes data to memory, assigning a constant value to each element.
- `rdwr` reads and then writes the same location, adding the current value to
  a running sum before assigning a new constant value.

Each mode accesses every fourth word of an array of integers. The script runs
each mode over the following 11 sizes, 33 runs in sequence:
`16k 256k 1M 2M 4M 8M 16M 32M 64M 256M 512M`.

Each run is `nice -n -20 bw_mem -P $(nproc) <size> <op>`. With `-P`, each of
the `nproc` processes allocates its own buffer of the given size, and the
reported bandwidth is the total across all processes.

The [`bw_mem` manual](https://github.com/intel/lmbench/blob/master/doc/bw_mem.8)
opens by describing a buffer twice the given size that is copied from one half
to the other. That description applies to the copy modes, which the image does
not run; `rd`, `wr`, and `rdwr` allocate one buffer of the given size.

## `bw_mem` Version

`bw_mem` comes from Ubuntu 24.04's `lmbench` package, whose version may differ
from the upstream LMbench source.

## Output

`bw_mem.sh` prints one line per successful run, `<op> <size> <bandwidth>`,
such as `rd 0.016384 147885.61`:

- `<op>` is `rd`, `wr`, or `rdwr`.
- `<size>` is the per-process buffer size in MB of 10^6 bytes, so `16k`
  prints as `0.016384`.
- `<bandwidth>` is the total MB/s across all processes.

A size that fails or prints unexpected output is skipped without an error, so
a run can have fewer than 33 lines.
