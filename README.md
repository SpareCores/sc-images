# sc-images

Container images that inspect hardware and benchmark the performance of cloud
servers for [Spare Cores Navigator](https://docs.sparecores.com/overview). Its
[Inspector](https://github.com/SpareCores/sc-inspector) runs them on each server
type, and the Navigator publishes the results for over 5,000 cloud server
types.

Each folder under [`images/`](images) is published as
`ghcr.io/sparecores/<folder>:main`, built for amd64 and arm64 unless its
README says otherwise. Every image's GHCR package page shows this README. To
find the docs for an image, look it up in [Images](#images) and follow its
link.

## Images

The following tables list the images by their `CATEGORY`. Where an image has a
README, it covers what the image measures, how to run it, and how to read its
output.

### Benchmarks

| Image | Description |
| ----- | ----------- |
| [`benchmark`](images/benchmark) | General CPU, memory, and compression benchmarks: Geekbench 6 scores, OpenSSL speed throughput, LMbench bw_mem memory bandwidth, and compression ratio and throughput on a Silesia corpus text file. |
| [`benchmark-ffmpeg`](images/benchmark-ffmpeg) | FFmpeg benchmark of aggregate transcoding throughput: frames per second for H.264/H.265 video encoding and decoding on CPUs and NVIDIA GPUs, and audio seconds processed per second for audio encoding on CPUs. |
| [`benchmark-llm`](images/benchmark-llm) | Low-concurrency LLM inference speed benchmark with llama.cpp's llama-bench, measuring prompt processing and text generation across models and token counts. |
| [`benchmark-passmark`](images/benchmark-passmark) | PassMark PerformanceTest CPU and memory benchmark suite. |
| [`benchmark-pgbench-postgres`](images/benchmark-pgbench-postgres) | PostgreSQL database benchmark focusing on CPU and memory performance, measuring transactions per second and latency against a local or remote server. |
| [`benchmark-redis`](images/benchmark-redis) | Redis 7.4 SET throughput and latency benchmark with memtier_benchmark, one server and client pair per CPU. |
| [`benchmark-storage`](images/benchmark-storage) | Raw block device benchmark with fio, measuring the IOPS (Input/Output Operations Per Second), bandwidth, and latency of a server's bundled disks, per disk type and across identical disks at once. |
| [`benchmark-vllm-cpu`](images/benchmark-vllm-cpu) | High-concurrency LLM inference benchmark with vLLM and GuideLLM, targeting x86_64 CPUs with AVX-512 and Arm64 CPUs. |
| [`benchmark-vllm-cpu-avx2`](images/benchmark-vllm-cpu-avx2) | High-concurrency LLM inference benchmark with vLLM and GuideLLM, targeting x86_64 CPUs with AVX2 but no AVX-512. |
| [`benchmark-vllm-gpu`](images/benchmark-vllm-gpu) | High-concurrency LLM inference benchmark with vLLM and GuideLLM, using the largest valid tensor-parallel size per model across all available GPUs. |
| [`benchmark-web`](images/benchmark-web) | Static web server benchmark, serving files with binserve and measuring throughput and latency with wrk. |
| [`nvbandwidth`](images/nvbandwidth) | NVIDIA GPU memory bandwidth benchmark with nvbandwidth, measuring host-to-device and device-to-device copy speeds. |
| [`stress-ng`](images/stress-ng) | CPU, memory, and kernel-heavy workload benchmark with stress-ng, measuring bogo ops/s for 85 stressors on one and on all CPUs, and the multi-core scaling of the cpu stressor. |
| [`stress-ng-longrun`](images/stress-ng-longrun) | Pinned stress-ng version for the 24-hour long-running benchmark. |

### Hardware Discovery

| Image | Description |
| ----- | ----------- |
| [`dmidecode`](images/dmidecode) | Reads the server's DMI/SMBIOS tables with dmidecode to report hardware details such as memory modules and firmware. |
| [`hwinfo`](images/hwinfo) | Collects CPU, memory, and topology details with hwloc, likwid, and lshw. |
| [`virtualization`](images/virtualization) | Checks whether the server supports KVM hardware virtualization and reports the result as JSON. |

### Services

| Image | Description |
| ----- | ----------- |
| [`benchmark-postgres-server`](images/benchmark-postgres-server) | PostgreSQL 18 server running at high priority, used as the remote target of database benchmarks. |
| [`llama.cpp`](images/llama.cpp) | llama.cpp server (llama-server) with CUDA support, built from a pinned commit. |

### Base Images

| Image | Description |
| ----- | ----------- |
| [`resource-tracker`](images/resource-tracker) | The resource-tracker binary of Spare Cores Navigator, copied into the other images to record resource usage while benchmarks run. |
| [`vllm-cpu-base-avx2`](images/vllm-cpu-base-avx2) | vLLM CPU build for x86_64 hosts with AVX2 but no AVX-512, used as the base of benchmark-vllm-cpu-avx2. |

Some images share one manual, and their READMEs cover only what differs. The
[vLLM manual](vllm-common) in `vllm-common/`, which is not a published image,
covers the `benchmark-vllm-*` images. `benchmark-pgbench-postgres` covers
`benchmark-postgres-server`, and `stress-ng` covers `stress-ng-longrun`.

## Usage

Run an image via Docker:

```shell
docker run --rm ghcr.io/sparecores/<image>:main
```

**Note:** Some images, such as `stress-ng`, `dmidecode`, and `hwinfo`, have no
default command, so append the command (such as `-c 'stress-ng --cpu 1'`) to run after the image name. Others
need extra parameters, such as environment variables or Docker options. The
command and parameters production passes to each image are in sc-inspector's
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py).

An image's README, where it has one, lists its settings, its output, and any
extra Docker options it needs; otherwise, see its `Dockerfile`. Benchmark
images run their workload under
[resource-tracker](https://github.com/SpareCores/resource-tracker-rs), which
samples CPU, memory, disk, network, and GPU usage once per second.

Some images need the following extra Docker options:

- **GPUs:** `--gpus all` for `benchmark-vllm-gpu`, `nvbandwidth`, and
  `llama.cpp`, and for the GPU runs of `benchmark-llm` and `benchmark-ffmpeg`.
- **Shared memory:** `--shm-size=4g` for the `benchmark-vllm-*` images, which
  fail with Docker's default 64&nbsp;MiB `/dev/shm`.
- **Privileged mode:** `--privileged` for `dmidecode`, `hwinfo`, and `virtualization`,
  and for the images that raise their priority with `nice -n -20`:
  `benchmark`, `benchmark-passmark`, `benchmark-pgbench-postgres`,
  `benchmark-postgres-server`, `benchmark-redis`, and `benchmark-web`. Without
  it, the `nice` images run at normal priority.
- **Disk access:** `--privileged --pid=host` for `benchmark-storage`. It
  unmounts and overwrites every disk it may write to, so do not run it on a
  machine with data you need.

## Contributing

See the following pages for maintaining and extending this repo:

- [Adding an Image](docs/contributors.md): the folder layout, the metadata
  files that control each image's build, and how dependencies are ordered.
- [CI and Build Internals](docs/ci.md): how CI builds, caches, tracks, and
  publishes the images.

## License

This repo is licensed under the [Mozilla Public License 2.0](LICENSE).
