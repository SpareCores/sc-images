# nvbandwidth

This benchmark measures how fast data moves between the host CPU and its
NVIDIA GPUs, and between the GPUs themselves. It runs `nvbandwidth` v0.8, an
open-source NVIDIA tool that times memory copies to measure bandwidth and
memory access to measure latency. The copies run between the CPU and each
GPU, between GPUs, and within the memory of a single GPU. The image runs
every `nvbandwidth` test case in sequence, with the tool's default settings,
on all GPUs passed to the container with `--gpus all`.

It reports one bandwidth matrix per test case, in GB/s, or a latency matrix in
nanoseconds, plus the matrix sum. The image adds no harness and no score: the
output is the unchanged `nvbandwidth` report.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements, such as raw CPU speed, memory bandwidth,
and LLM (Large Language Model) inference speed, for more than 5,000 cloud
server types. This benchmark adds measured raw GPU performance: how fast
data moves between the CPU and the GPUs of a server type, and between its
GPUs.

The results depend on the following properties of the host:

- GPU model and count
- links between the CPU and the GPUs and between GPUs, such as PCIe
  (Peripheral Component Interconnect Express) or NVLink
- host CPU and memory
- NVIDIA driver version, recorded by each run

## Limitations

This benchmark's results compare data movement on one server, not GPU compute speed or
application performance. The main limitations are the following:

- **Copies Only:** Every test case copies memory or measures memory access
  latency. Compute throughput, GPU memory capacity, and real workloads are not
  measured.
- **Single Server:** `nvbandwidth` is built without its multinode mode, so
  only the CPU and GPUs of one host are measured, not links between servers or
  network throughput.
- **[No Peer Access](#results):** The 17 test cases between GPUs need P2P
  (Peer-to-Peer) access between at least two GPUs. On a server with one GPU,
  or with GPUs that cannot access each other, they report
  `"status": "Waived"` and no matrix.
- **NVIDIA Only:** `nvbandwidth` uses CUDA (Compute Unified Device
  Architecture), so it needs NVIDIA GPUs. The image does not check for them;
  production skips servers without NVIDIA GPUs.
- **[Fixed Settings](#settings):** The image passes no options, so the
  upstream defaults apply: a 512 MiB copy buffer, four untimed warmup copies
  before the timed copies, and the median of three iterations per measurement.
  The image adds no warmup or repetitions of its own. See also
  [Workload](#workload).
- **Host Driver:** The host provides the NVIDIA driver, so its version varies
  between runs. The image fixes only the `nvbandwidth` version and the CUDA
  Toolkit 12.6.1 it is built with.

## Usage

Run the image via Docker, with the host's GPUs passed to the container:

```bash
docker run --rm --gpus all ghcr.io/sparecores/nvbandwidth:main
```

The host needs an NVIDIA GPU, the NVIDIA driver, and the NVIDIA Container
Toolkit.

The default command prints the `nvbandwidth` report as plain text. Production
runs `nvbandwidth -j` instead, which prints JSON:

```bash
docker run --rm --gpus all ghcr.io/sparecores/nvbandwidth:main nvbandwidth -j
```

Arguments after the image name replace the default command, and still run
under `resource-tracker`, which samples resource usage during the run. Its
data is not published.

Production also runs the container privileged and with host networking, with a
30-minute timeout. Production runs with `nvbandwidth` v0.8 took between 55
seconds and \~20 minutes, with a median of \~3.5 minutes; the slowest
was on a server with eight GPUs.

### Settings

This benchmark has no settings of its own in environment variables. The image
sets only `TRACKER_QUIET=true`, which silences the output of
`resource-tracker`. To change a setting, pass `nvbandwidth` options after the
image name, as in the JSON example above. The most common options are the
following:

| Option | Meaning | Default |
| ------ | ------- | ------- |
| `-j` | Print JSON instead of plain text | Off |
| `-t <name or index>` | Run only the given test cases | All |
| `-p <prefix>` | Run only the test cases whose names start with the prefix | All |
| `-l` | List the test cases and exit | Off |
| `-b <MiB>` | Copy buffer size | 512 |
| `-i <count>` | Iterations per measurement | 3 |
| `-m` | Report the mean instead of the median of the iterations | Off |
| `-s` | Skip the data check after each copy | Off |
| `-d` | Turn off automatic CPU affinity | Off |
| `-v` | Verbose output | Off |

See the
[`nvbandwidth` v0.8 README](https://github.com/NVIDIA/nvbandwidth/blob/v0.8/README.md)
for all options and test cases.

### Results

With `-j`, `nvbandwidth` prints one JSON object to stdout, with the following
fields under its top-level `nvbandwidth` key:

- `CUDA Runtime Version` and `Driver Version` record the CUDA runtime and the
  host's NVIDIA driver, for example `13000` (CUDA 13.0) and `580.126.09`. The
  reported runtime version can differ from CUDA Toolkit 12.6.1, which the
  image is built with.
- `GPU Device list` names each GPU, with its index, name, bus address, and
  hostname.
- `version` is the `nvbandwidth` version (`v0.8`). `git_version` is empty,
  because the image builds `nvbandwidth` from a release archive.
- `testcases[]` has one entry per test case, with its `name` and `status`.

A test case that ran has `"status": "Passed"` and the following fields:

- `bandwidth_description` says what the matrix holds and in which unit, for
  example `memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)`.
- `bandwidth_matrix` is a two-dimensional array of strings, with rows and
  columns as the description says. Pairs that were not measured hold `N/A`.
- `sum` is the sum of the matrix cells, as a number.

The 17 test cases between GPUs need P2P access between at least two GPUs.
Without it, on a server with one GPU or with GPUs that cannot access each
other, they report `"status": "Waived"` and have no matrix. In production,
16 of 27 runs with `nvbandwidth` v0.8 on servers with two to four GPUs waived
all 17. An interrupted run can end with a test case still `Running`.

For example, on an `alicloud/ecs.gn5-c4g1.xlarge` with one Tesla P100 GPU,
`host_to_device_memcpy_ce` measured 11.06 GB/s, `host_device_latency_sm`
799 ns, and `device_local_copy` 268.75 GB/s, while all 17 test cases between
GPUs were waived
([production output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/alicloud/ecs.gn5-c4g1.xlarge/0/nvbandwidth/stdout)).

On a fatal error, such as a GPU that CUDA cannot initialize, the JSON has an
`error` field instead of `testcases`, and the process exits with code 1. A
failed data check prints `Invalid value when checking the pattern at ...`
before the JSON and also exits with code 1.

The Navigator stores the following metrics from each run, computed only from
test cases with `"status": "Passed"`, but does not show them on its website
yet. The table uses the following terms:

- CE (Copy Engine) and SM (Streaming Multiprocessor) name the copy method; see
  [Workload](#workload).
- P2P means direct copies between GPUs.
- "Averaged" means the `sum` divided by the number of cells that are not `N/A`.

| Metric | Meaning | Unit | Computed from |
| ------ | ------- | ---- | ------------- |
| `nvbandwidth:all:host_to_gpu` | Total rate when the CPU copies to all GPUs at once (CE) | GB/s | `host_to_all_memcpy_ce` `sum` |
| `nvbandwidth:all:gpu_to_host` | Total rate when all GPUs copy to the CPU at once (CE) | GB/s | `all_to_host_memcpy_ce` `sum` |
| `nvbandwidth:all:duplex` | Per-direction rate when copying both ways at once, the lower of the two directions (CE) | GB/s | Lower `sum` of `host_to_all_bidirectional_memcpy_ce` and `all_to_host_bidirectional_memcpy_ce` |
| `nvbandwidth:slot:host_to_gpu` | Average PCIe slot speed per GPU, CPU to GPU (CE) | GB/s | `host_to_device_memcpy_ce`, averaged |
| `nvbandwidth:slot:gpu_to_host` | Average PCIe slot speed per GPU, GPU to CPU (CE) | GB/s | `device_to_host_memcpy_ce`, averaged |
| `nvbandwidth:slot:latency` | Average CPU-GPU round-trip time (SM); lower is better | ns | `host_device_latency_sm`, averaged |
| `nvbandwidth:p2p:single` | Write speed between two GPUs with P2P copies (CE) | GB/s | `device_to_device_memcpy_write_ce`, averaged |
| `nvbandwidth:p2p:duplex` | Speed between two GPUs copying both ways at once (CE) | GB/s | `device_to_device_bidirectional_memcpy_write_ce`, averaged |
| `nvbandwidth:p2p:latency` | Latency between GPU pairs with P2P access (SM); lower is better | ns | `device_to_device_latency_sm`, averaged |
| `nvbandwidth:p2p:gather` | Rate into one GPU when all other GPUs write to it at once (CE) | GB/s | `all_to_one_write_ce`, averaged |
| `nvbandwidth:efficiency:sm_ce_ratio` | SM copy speed relative to CE copy speed, capped at 1 | Ratio | `host_to_device_memcpy_sm` `sum` divided by `host_to_device_memcpy_ce` `sum` |

Each stored result also records the following:

- driver version
- CUDA runtime version
- `nvbandwidth` version
- GPU count
- whether P2P is supported, that is, whether any `device_to_device_*`
  test case passed

## Workload

This benchmark runs one workload: the image's default command,
`nvbandwidth` with no options, which runs all 35 test cases in sequence. The
[`Dockerfile`](./Dockerfile) builds `nvbandwidth` v0.8 from NVIDIA's release
archive with CUDA Toolkit 12.6.1.

Each nvbandwidth test case uses one of the following copy methods, named by its
suffix:

-- **CE:** `_ce` test cases copy with the GPU's copy engines, through CUDA's
  `memcpy` functions.
-- **SM:** `_sm` test cases copy with a kernel that runs on the GPU's
  streaming multiprocessors, with 512 threads per block. The copy size is
  rounded down so that it divides evenly across the SMs.

The 16 CE and 16 SM test cases cover the following copy directions:

- `host_to_device_*` and `device_to_host_*`: between the CPU and each GPU, one
  GPU at a time.
- `host_to_all_*` and `all_to_host_*`: between the CPU and all GPUs at once.
- `*_bidirectional_*`: the same copies in both directions at once.
- `device_to_device_memcpy_read_*` and `device_to_device_memcpy_write_*`:
  between each pair of GPUs.
- `all_to_one_*` and `one_to_all_*`: from all GPUs to one, and from one GPU to
  all others, with a read and a write variant each.

Three more test cases complete the set: `host_device_latency_sm` and
`device_to_device_latency_sm` measure memory access latency between the CPU
and each GPU and between GPUs, and `device_local_copy` measures copies within
each GPU's own memory.

Each measurement times a fixed number of copies between start and stop
events, repeats three times, and reports the median. Four warmup copies run
before the timed copies, outside the timed interval. `nvbandwidth` queues all
copies before releasing them, so the time spent queuing them is not measured.
It checks the copied data after each copy and sets CPU affinity
automatically.

## Design History

This benchmark has no harness of its own: the image runs the upstream
`nvbandwidth` binary, so the test cases and their settings are NVIDIA's. The
image first shipped `nvbandwidth` v0.5 and moved to v0.8 in 2026. Production
records the version of each run, and many server types measured before the
update still have v0.5 results.

## References

See the full list of [references](./docs/references.md) used in this
documentation.
