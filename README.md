# sc-images

Spare Cores container images inspect the hardware of cloud servers and
benchmark their performance for the
[Spare Cores Navigator](https://docs.sparecores.com/overview). The Navigator's
[Inspector](https://github.com/SpareCores/sc-inspector) runs them on cloud
servers, and the results are published in the Navigator, which lists over
5,000 cloud server types.

Each image is built for `amd64` and `arm64` unless its docs say otherwise. The
docs of each image can be found linked from the [image list](#images),
published at `https://docs.sparecores.com/images/<image>/`.

## Usage

Run an image via Docker:

```shell
docker run --rm ghcr.io/sparecores/<image>:main
```

**Note:** Some images, such as `benchmark`, `stress-ng`, `dmidecode`, and
`hwinfo`, have no default command, so append the command to run after the image
name: a script such as `openssl.sh` for `benchmark`, or
`-c 'stress-ng --cpu 1'` for `stress-ng`, whose entrypoint is `sh`.

Other images need extra parameters, such as environment variables or Docker
options. The command and parameters that production passes to each image are
in
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in the `sc-inspector` repository. An image's README, where it has one, lists its settings, its output, and any
extra Docker options it needs; otherwise, see its `Dockerfile`.

Benchmark
images run their workload under
[`resource-tracker`](https://github.com/SpareCores/resource-tracker-rs), which
samples CPU, memory, disk, network, and GPU usage once per second.

Some images need the following extra Docker options:

- **GPUs:** `--gpus all` for `benchmark-vllm-gpu`, `nvbandwidth`, and
  `llama.cpp`, and for the GPU runs of `benchmark-llm` and `benchmark-ffmpeg`.
- **Shared memory:** `--shm-size=4g` for the `benchmark-vllm-*` images, which
  fail with Docker's default 64 MiB `/dev/shm`.
- **Privileged mode:** `--privileged` for `dmidecode`, `hwinfo`, and
  `virtualization`, and for the images that raise their priority with
  `nice -n -20`:
  `benchmark`, `benchmark-passmark`, `benchmark-pgbench-postgres`,
  `benchmark-postgres-server`, `benchmark-redis`, and `benchmark-web`. Without
  it, the `nice` images run at normal priority.
- **Disk access:** `--privileged --pid=host` for `benchmark-storage`. It
  unmounts and overwrites every disk it may write to, so do not run it on a
  machine with data to keep.

## Images

The following tables list the images by their `CATEGORY`. Where an image has a
README, it covers what the image measures, how to run it, and how to read its
output.

### Benchmarks

| Image | Description | Docs |
| ----- | ----------- | ---- |
| [`benchmark`](images/benchmark) | General CPU, memory, and compression benchmarks: Geekbench 6 scores, OpenSSL speed throughput, LMbench bw_mem memory bandwidth, and compression ratio and throughput on a Silesia corpus text file. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark/) |
| [`benchmark-ffmpeg`](images/benchmark-ffmpeg) | FFmpeg benchmark of aggregate transcoding throughput: frames per second for H.264/H.265 video encoding and decoding on CPUs and NVIDIA GPUs, and audio seconds processed per second for audio encoding on CPUs. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-ffmpeg/) |
| [`benchmark-llm`](images/benchmark-llm) | Single-stream LLM inference speed benchmark with llama.cpp's llama-bench, measuring prompt processing and text generation in tokens per second for six models at several token counts, on CPUs and NVIDIA GPUs. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-llm/) |
| [`benchmark-passmark`](images/benchmark-passmark) | PassMark PerformanceTest benchmark of CPU and memory performance, reporting the CPU Mark and Memory Mark composite scores and the individual test results behind them. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-passmark/) |
| [`benchmark-pgbench-postgres`](images/benchmark-pgbench-postgres) | PostgreSQL database benchmark focusing on CPU and memory performance, measuring transactions per minute and latency against a local or remote server. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-pgbench-postgres/) |
| [`benchmark-redis`](images/benchmark-redis) | Redis 7.4 SET throughput and latency benchmark with memtier_benchmark, one server and client pair per CPU. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-redis/) |
| [`benchmark-storage`](images/benchmark-storage) | Raw block device benchmark with fio, measuring the IOPS (Input/Output Operations Per Second), bandwidth, and latency of a server's bundled disks, per disk type and across identical disks at once. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-storage/) |
| [`benchmark-vllm-cpu`](images/benchmark-vllm-cpu) | High-concurrency LLM inference benchmark with vLLM and GuideLLM on AMD64 CPUs with AVX-512 (Advanced Vector Extensions 512) and on Arm64 CPUs, measuring time to first token, time per output token, inter-token latency, end-to-end request latency, and token and request throughput, from one request at a time up to peak load. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-vllm-cpu/) |
| [`benchmark-vllm-cpu-avx2`](images/benchmark-vllm-cpu-avx2) | High-concurrency LLM inference benchmark with vLLM and GuideLLM on AMD64 CPUs with AVX2 (Advanced Vector Extensions 2) but without AVX-512 (Advanced Vector Extensions 512), measuring time to first token, time per output token, inter-token latency, end-to-end request latency, and token and request throughput, from one request at a time up to peak load. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-vllm-cpu-avx2/) |
| [`benchmark-vllm-gpu`](images/benchmark-vllm-gpu) | High-concurrency LLM inference benchmark with vLLM and GuideLLM on NVIDIA GPUs, measuring time to first token, time per output token, inter-token latency, end-to-end request latency, and token and request throughput, from one request at a time up to peak load. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-vllm-gpu/) |
| [`benchmark-web`](images/benchmark-web) | Static web server benchmark, serving files with binserve and measuring throughput and latency with wrk. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-web/) |
| [`nvbandwidth`](images/nvbandwidth) | NVIDIA GPU memory copy bandwidth and latency benchmark with nvbandwidth, measuring copies between the host CPU and each GPU, between GPUs, and within one GPU. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/nvbandwidth/) |
| [`stress-ng`](images/stress-ng) | CPU, memory, and kernel-heavy workload benchmark with stress-ng, measuring bogo ops/s for 85 stressors on one and on all CPUs, and the multi-core scaling of the cpu stressor. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/stress-ng/) |
| [`stress-ng-longrun`](images/stress-ng-longrun) | Pinned stress-ng version for the 24-hour long-running benchmark. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/stress-ng-longrun/) |

### Hardware Discovery

| Image | Description | Docs |
| ----- | ----------- | ---- |
| [`dmidecode`](images/dmidecode) | Reads the server's DMI (Desktop Management Interface) and SMBIOS (System Management BIOS) tables with dmidecode to report hardware details such as memory modules and firmware. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/dmidecode/) |
| [`hwinfo`](images/hwinfo) | Collects CPU, memory, and topology details with hwloc, likwid, and lshw. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/hwinfo/) |
| [`virtualization`](images/virtualization) | Checks whether the server supports KVM (Kernel-based Virtual Machine) hardware virtualization and reports the result as JSON. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/virtualization/) |

### Services

| Image | Description | Docs |
| ----- | ----------- | ---- |
| [`benchmark-postgres-server`](images/benchmark-postgres-server) | PostgreSQL 18 server running at high priority, used as the remote target of database benchmarks. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/benchmark-postgres-server/) |
| [`llama.cpp`](images/llama.cpp) | llama.cpp server (llama-server) with CUDA (Compute Unified Device Architecture) support, built from a pinned commit. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/llama.cpp/) |

### Base Images

| Image | Description | Docs |
| ----- | ----------- | ---- |
| [`resource-tracker`](images/resource-tracker) | The resource-tracker binary of Spare Cores Navigator, copied into the other images to record resource usage while benchmarks run. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/resource-tracker/) |
| [`vllm-cpu-base-avx2`](images/vllm-cpu-base-avx2) | vLLM CPU build for x86_64 hosts with AVX2 (Advanced Vector Extensions 2) but no AVX-512 (Advanced Vector Extensions 512), used as the base of benchmark-vllm-cpu-avx2. | [![Docs](https://raw.githubusercontent.com/SpareCores/sc-images/main/docs/img/docs.svg)](https://docs.sparecores.com/images/vllm-cpu-base-avx2/) |

The following docs cover more than one image. Where a covered image has its
own README, that README describes only what differs:

- The [vLLM benchmarks manual](./vllm-common) in `vllm-common/`, which is not
  a published image, covers the `benchmark-vllm-*` images.
- The `benchmark-pgbench-postgres` docs cover `benchmark-postgres-server`.
- The `stress-ng` docs cover `stress-ng-longrun`.

## Contributing

See the following pages for maintaining and extending the
[`sc-images`](https://github.com/SpareCores/sc-images) repository:

- [Adding an Image](./docs/contributors.md): the folder layout, the metadata
  files that control each image's build, and how dependencies are ordered.
- [CI and Build Internals](./docs/ci.md): how CI builds, caches, tracks, and
  publishes the images.

## License

The `sc-images` repository is licensed under the
[Mozilla Public License 2.0](./LICENSE).
