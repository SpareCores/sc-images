# sc-images

Container images that inspect hardware and benchmark the performance of cloud
servers. [sc-inspector](https://github.com/SpareCores/sc-inspector) runs them
on each server type, and [Navigator](https://sparecores.com/servers) publishes
the results for over 5,000 cloud server types.

Each folder under [`images/`](images) is published as
`ghcr.io/sparecores/<folder>:main`, built for amd64 and arm64 unless its
README says otherwise. Every image's GHCR package page shows this README. To
find the docs for an image, look it up in [Images](#images) and follow its
link to the image folder.

## Images

The following tables list the images by their `CATEGORY`. Where an image has a
README, it covers what the image measures, how to run it, and how to read its
output.

### Benchmarks

| Image | Description |
| ----- | ----------- |
| [`benchmark`](images/benchmark) | General CPU, memory, and compression benchmarks: Geekbench, OpenSSL speed, lmbench memory bandwidth (bw_mem), and compression of the Silesia corpus. |
| [`benchmark-ffmpeg`](images/benchmark-ffmpeg) | FFmpeg transcoding-capacity benchmark of audio and H.264/H.265 video encoding. |
| [`benchmark-llm`](images/benchmark-llm) | Low-concurrency LLM inference speed benchmark with llama.cpp's llama-bench, measuring prompt processing and text generation across models and token counts. |
| [`benchmark-passmark`](images/benchmark-passmark) | PassMark PerformanceTest CPU and memory benchmark suite. |
| [`benchmark-pgbench-postgres`](images/benchmark-pgbench-postgres) | PostgreSQL database benchmark focusing on CPU and memory performance, measuring transactions per second and latency against a local or remote server. |
| [`benchmark-redis`](images/benchmark-redis) | Redis 7.4 in-memory key-value store benchmark. |
| [`benchmark-storage`](images/benchmark-storage) | Raw block device benchmark with fio, testing each distinct disk type of the host and multi-disk groups concurrently. |
| [`benchmark-vllm-cpu`](images/benchmark-vllm-cpu) | High-concurrency LLM inference benchmark with vLLM and GuideLLM, targeted CPUs with AVX-512. |
| [`benchmark-vllm-cpu-avx2`](images/benchmark-vllm-cpu-avx2) | High-concurrency LLM inference benchmark with vLLM and GuideLLM, targeted CPUs with AVX2 but no AVX-512. |
| [`benchmark-vllm-gpu`](images/benchmark-vllm-gpu) | High-concurrency LLM inference benchmark with vLLM and GuideLLM, using tensor parallelism across all available GPUs. |
| [`benchmark-web`](images/benchmark-web) | Static web server benchmark, serving files with binserve and measuring throughput and latency with wrk. |
| [`nvbandwidth`](images/nvbandwidth) | NVIDIA GPU memory bandwidth benchmark with nvbandwidth, measuring host-to-device and device-to-device copy speeds. |
| [`stress-ng`](images/stress-ng) | CPU benchmark running stress-ng stressors and counting their operations per second. |
| [`stress-ng-longrun`](images/stress-ng-longrun) | Pinned stress-ng version for the 24-hour long-running benchmark. |

### Hardware Discovery

| Image | Description |
| ----- | ----------- |
| [`dmidecode`](images/dmidecode) | Reads the server's DMI/SMBIOS tables with dmidecode to report hardware details such as memory modules and firmware. |
| [`hwinfo`](images/hwinfo) | Collects CPU, memory, and topology details with lshw and hwloc, plus the measured CPU frequency. |
| [`virtualization`](images/virtualization) | Checks whether the server supports KVM hardware virtualization and reports the result as JSON. |

### Services

| Image | Description |
| ----- | ----------- |
| [`benchmark-postgres-server`](images/benchmark-postgres-server) | PostgreSQL 18 server running at high priority, used as the remote target of database benchmarks. |
| [`llama.cpp`](images/llama.cpp) | llama.cpp server (llama-server) with CUDA support, built from a pinned commit. |

### Base Images

| Image | Description |
| ----- | ----------- |
| [`resource-tracker`](images/resource-tracker) | Spare Cores resource-tracker binary, copied into the other images to record resource usage while benchmarks run. |
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

An image's README, where it has one, lists its settings, its output, and any
extra Docker options it needs; otherwise, see its `Dockerfile`. Benchmark
images run their workload under
[resource-tracker](https://github.com/SpareCores/resource-tracker-rs), which
samples CPU, memory, disk, network, and GPU usage once per second.

Some images need the following extra Docker options:

- **GPUs:** `--gpus all` for `benchmark-vllm-gpu`, `nvbandwidth`, and
  `llama.cpp`, and for the GPU runs of `benchmark-llm` and `benchmark-ffmpeg`.
- **Disk access:** `--privileged --pid=host` for `benchmark-storage`. It
  discards, overwrites, and unmounts the disks it tests, so do not run it on a
  machine with data you need.

Production runs set their Docker options per task in sc-inspector's
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py),
so they can differ from a plain `docker run`.

## Adding an Image

CI auto-discovers every folder under `images/` (any dir with a `Dockerfile` or `prepare.sh`) and builds it. Per-image behavior is declared with optional metadata files. The defaults cover simple single-stage images, so most folders need none of the following:

| File | Meaning | Default |
|------|---------|---------|
| `DEPENDS_ON` | image folder names that must be published **before** this one (one per line) | none |
| `PLATFORMS` | arches to build | `amd64 arm64` |
| `CONTEXT` | build context, repo-relative | `images/<folder>` |
| `DOCKERFILE` | Dockerfile path, repo-relative | `images/<folder>/Dockerfile` |
| `TARGET` | build target stage | none |
| `BUILD_ARGS` | `KEY=VALUE` lines; tokens `${ARCH}`, `${VLLM_VERSION}`, `${GUIDELLM_VERSION}`, `${RESOURCE_TRACKER_VERSION}` | none |
| `ZRAM` | enable compressed swap on the builder (`true`/`1`/`yes`, or a PERCENT e.g. `125`) | off |
| `SCCACHE` | enable sccache S3 compile cache for source builds (`true`/`1`/`yes`) | off |
| `prepare.sh` | pre-build hook (clone sources, patch Dockerfile, emit BuildKit secret + parallelism) | none |
| `DESCRIPTION` | one-sentence summary; published as the `org.opencontainers.image.description` label and index annotation, so it shows on the GHCR package page | none (CI warns) |
| `CATEGORY` | group name: `benchmark`, `hardware-discovery`, `service`, or `base`; published as the `com.sparecores.image.category` label and used to group the [Images](#images) tables | none (CI warns) |

[`resolve-build-plan.sh`](.github/scripts/resolve-build-plan.sh) topologically sorts folders by `DEPENDS_ON` into levels (0 = no deps). [`push.yml`](.github/workflows/push.yml) builds each level via the reusable [`build-level.yml`](.github/workflows/build-level.yml) and publishes it before the next, so a dependency is always available as a base image; [`read-image-config.sh`](.github/scripts/read-image-config.sh) resolves each folder's config at build time. No image names are hardcoded in the workflow.

Example: every `benchmark-*` image has `DEPENDS_ON: resource-tracker`; `benchmark-vllm-cpu-avx2` also depends on `vllm-cpu-base-avx2` (source-compiled via `prepare.sh`). `benchmark-vllm-gpu` uses Hub `vllm/vllm-openai` for amd64 and arm64. Add a new dependent image by dropping a folder in `images/` with a `DEPENDS_ON` file.

## CI and Build Internals

See [CI and Build Internals](docs/ci.md) for how CI builds, caches, tracks,
and publishes the images.

## License

This repo is licensed under the [Mozilla Public License 2.0](LICENSE).
