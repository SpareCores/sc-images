# sc-images

Spare Cores container images for hardware inspection and benchmarking workloads.

## resource-tracker

Benchmark images copy the static [resource-tracker](https://github.com/SpareCores/resource-tracker-rs) binary from [`ghcr.io/sparecores/resource-tracker`](images/resource-tracker). The release version is defined in [`images/resource-tracker/RESOURCE_TRACKER_VERSION`](images/resource-tracker/RESOURCE_TRACKER_VERSION).

CI publishes per-arch tags (`main-amd64`, `main-arm64`) during the build matrix; the per-level `merge` step then creates the `main` multi-arch manifest. Benchmark image builds use `RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main-<arch>` so they can resolve the binary without waiting for the merged tag.

Force-rebuild all resource-tracker consumer images without registry cache: set `FORCE_REBUILD_RESOURCE_TRACKER` to `"true"` in [`.github/workflows/push.yml`](.github/workflows/push.yml), or use **Actions → Build and publish Spare Cores images → Run workflow** (`force_rebuild_resource_tracker`).

### CI build tracking

Every image build job runs resource-tracker in **standalone mode** (no wrapped command, no `--pid`) while `docker buildx build` runs in parallel. `docker buildx` only drives buildkitd over the API, so compiles do not appear in a shell-wrapper process tree; without a tracked PID, resource-tracker collects system-wide CPU, memory, disk, and network (see [Usage.md](https://github.com/SpareCores/resource-tracker-rs/blob/main/resource-tracker-rs-book/src/Usage.md): *"Leave unset [pid] to collect system-wide metrics only"*). Metrics are sent to [Sentinel](https://sentinel.sparecores.net) when the repo secret **`SENTINEL_API_TOKEN`** is set (Settings → Secrets and variables → Actions → Secrets). Without the token, tracking runs locally only and Sentinel upload is skipped.

Job metadata (mirrors [sc-inspector](https://github.com/SpareCores/sc-inspector) benchmark runs):

| Variable | CI default |
|----------|------------|
| `TRACKER_PROJECT_NAME` | `sc-images` |
| `TRACKER_JOB_NAME` | matrix entry name, e.g. `vllm-gpu-base (arm64)` |
| `TRACKER_TASK_NAME` | image folder, e.g. `vllm-gpu-base` |
| `TRACKER_STAGE_NAME` | workflow name |
| `TRACKER_EXTERNAL_RUN_ID` | `github.run_id`-`github.run_attempt` |
| `TRACKER_CONTAINER_IMAGE` | `ghcr.io/sparecores/<folder>:main-<arch>` |
| `TRACKER_ORCHESTRATOR` | `github-actions` |
| `TRACKER_ENV` | `ci` |
| `TRACKER_QUIET` | `true` (no metric spam in build logs) |

Scripts: [`install-resource-tracker.sh`](.github/scripts/install-resource-tracker.sh), [`run-with-resource-tracker.sh`](.github/scripts/run-with-resource-tracker.sh), [`build-push-image.sh`](.github/scripts/build-push-image.sh).

### arm64 CI runner

Set repo variable **`ARM64_RUNNER`** (Settings → Secrets and variables → Actions → Variables):

| Value | `runs-on` | vLLM GPU compile parallelism |
|-------|-----------|------------------------------|
| `self-hosted` | `[self-hosted, sc-images-arm64]` | RAM-linear from 64 GiB baseline (effective RAM incl. zram budget) |
| `github` | `ubuntu-24.04-arm` | conservative: physical RAM only, 6 GiB/slot, max 2 compile jobs |

Default (variable unset): `self-hosted`.
Self-hosted: register with `./config.sh ... --labels sc-images-arm64` (`--no-default-labels` is OK).

Compile/runtime job counts are computed by [`.github/scripts/compute-build-parallelism.sh`](.github/scripts/compute-build-parallelism.sh). On self-hosted builders they scale with effective RAM (MemTotal + zram budget); on GitHub-hosted runners the script switches to a conservative physical-RAM profile so CUDA/C++ compiles do not OOM the agent. Dockerfile build-args stay at the fixed 64 GiB reference (`max_jobs=32`, `nvcc_threads=2`) so registry/GHA layer caches built on a large machine remain valid on smaller runners; actual `-j` values are injected through a BuildKit secret (excluded from layer cache keys).

Images with a `ZRAM` metadata file run [`.github/scripts/setup-zram.sh`](.github/scripts/setup-zram.sh) before the build (`PERCENT=125` by default → compressed swap ≈ 1.25× RAM). Currently enabled for the source-compiled vLLM bases (`vllm-gpu-base`, `vllm-cpu-base-avx2`). On **GitHub-hosted** Azure kernels, `zram` is in `linux-modules-extra-$(uname -r)` ([Launchpad #1762756](https://bugs.launchpad.net/ubuntu/bionic/+source/linux-azure/+bug/1762756)); the script installs that package when needed. If the extra-modules deb is missing from apt (kernel/image drift), it keeps the default `/swapfile`. Tune the default percent via workflow `env` `ZRAM_PERCENT`, or override per image with a numeric `ZRAM` value.

### sccache (vLLM source builds)

Images with a `SCCACHE` metadata file compile C++/CUDA through [sccache](https://github.com/mozilla/sccache) with an S3 backend. Infrastructure lives in [`sc-infra/prod/spare-cores/sc_images.py`](https://github.com/SpareCores/sc-infra/blob/main/prod/spare-cores/sc_images.py) (bucket in `us-west-2`, 90-day lifecycle).

Repo configuration (set by Pulumi after `pulumi up`):

| Name | Kind | Purpose |
|------|------|---------|
| `SCCACHE_BUCKET` | Actions variable | S3 bucket name |
| `SCCACHE_REGION` | Actions variable | Bucket region (`us-west-2`) |
| `SC_IMAGES_GHA_ROLE_ARN` | Actions secret | OIDC role for read/write on the bucket |

Each build uses prefix `{folder}/{arch}/` (e.g. `vllm-gpu-base/arm64/`) via `SCCACHE_S3_KEY_PREFIX`. Registry layer cache (`buildcache-<arch>` tags on GHCR) is unchanged; sccache caches compiler objects inside the vLLM source compile steps.

Coverage when enabled:

| Stage | Mechanism |
|-------|-----------|
| `rust-build` | `RUSTC_WRAPPER=/usr/bin/sccache` for cargo; `CC`/`CXX` absolute paths for build-script C compiles (openssl-sys, etc.) |
| `csrc-build` / `vllm-build` wheel | CMake compiler launchers via setup.py (`CMAKE_*_COMPILER_LAUNCHER=sccache`); `RUSTC_WRAPPER` unset so C++ is not double-wrapped |

`sccache` v0.15.0 is installed to `/usr/bin/sccache`. `SCCACHE_IGNORE_SERVER_IO_ERROR=1` falls back to local compiles if the cache server is unreachable.

Currently enabled for `vllm-gpu-base` and `vllm-cpu-base-avx2`.

## Build framework

CI auto-discovers every folder under `images/` (any dir with a `Dockerfile` or `prepare.sh`) and builds it. Per-image behaviour is declared with optional metadata files — defaults cover simple single-stage images, so most folders need none:

| File | Meaning | Default |
|------|---------|---------|
| `DEPENDS_ON` | image folder names that must be published **before** this one (one per line) | none |
| `PLATFORMS` | arches to build | `amd64 arm64` |
| `CONTEXT` | build context, repo-relative | `images/<folder>` |
| `DOCKERFILE` | Dockerfile path, repo-relative | `images/<folder>/Dockerfile` |
| `TARGET` | build target stage | none |
| `BUILD_ARGS` | `KEY=VALUE` lines; tokens `${ARCH}`, `${VLLM_VERSION}`, `${RESOURCE_TRACKER_VERSION}` | none |
| `ZRAM` | enable compressed swap on the builder (`true`/`1`/`yes`, or a PERCENT e.g. `125`) | off |
| `SCCACHE` | enable sccache S3 compile cache for source builds (`true`/`1`/`yes`) | off |
| `prepare.sh` | pre-build hook (clone sources, patch Dockerfile, emit BuildKit secret + parallelism) | none |

[`resolve-build-plan.sh`](.github/scripts/resolve-build-plan.sh) topologically sorts folders by `DEPENDS_ON` into levels (0 = no deps). [`push.yml`](.github/workflows/push.yml) builds each level via the reusable [`build-level.yml`](.github/workflows/build-level.yml) and publishes it before the next, so a dependency is always available as a base image; [`read-image-config.sh`](.github/scripts/read-image-config.sh) resolves each folder's config at build time. No image names are hardcoded in the workflow.

Example: every `benchmark-*` image has `DEPENDS_ON: resource-tracker`; `benchmark-vllm-gpu` also depends on `vllm-gpu-base` and `benchmark-vllm-cpu-avx2` on `vllm-cpu-base-avx2` (both source-compiled via `prepare.sh`). Add a new dependent image by dropping a folder in `images/` with a `DEPENDS_ON` file.

## Images

- [benchmark](images/benchmark) - Hardware benchmarks for memory bandwidth, compression, OpenSSL crypto operations, and Geekbench tests
- [benchmark-llm](images/benchmark-llm) - LLM inference speed benchmarks using `llama.cpp` with different models and token lengths for prompt processing and text generation
- [vllm-common](vllm-common) - Shared vLLM pin and harness (`benchmark.py`); not a published image
- [benchmark-vllm-gpu](images/benchmark-vllm-gpu) - vLLM GPU serving via `vllm serve` + [GuideLLM](https://github.com/vllm-project/guidellm) (amd64 + arm64, multi-GPU TP)
- [benchmark-vllm-cpu](images/benchmark-vllm-cpu) - vLLM CPU serving on Hub image (AVX-512 amd64 + arm64)
- [benchmark-vllm-cpu-avx2](images/benchmark-vllm-cpu-avx2) - vLLM CPU serving for AVX2-only amd64
- [vllm-cpu-base-avx2](images/vllm-cpu-base-avx2) - amd64 AVX2 vLLM base built from upstream `Dockerfile.cpu`
- [vllm-gpu-base](images/vllm-gpu-base) - arm64 CUDA vLLM `vllm-openai` image built from upstream source (for GH200-class GPUs)
- [benchmark-passmark](images/benchmark-passmark) - PassMark Performance Test benchmarking suite for CPU and memory performance
- [benchmark-redis](images/benchmark-redis) - Redis server performance benchmarks using `memtier_benchmark`
- [benchmark-web](images/benchmark-web) - Static web server performance benchmarks using `wrk` and `binserve`
- [dmidecode](images/dmidecode) - Hardware information reader using `dmidecode`
- [hwinfo](images/hwinfo) - Hardware information tools including `likwid` and `lshw`
- [nvbandwidth](images/nvbandwidth) - NVIDIA GPU memory bandwidth measurement tool
- [resource-tracker](images/resource-tracker) - resource-tracker binary distributed for use in benchmark images
- [stress-ng](images/stress-ng) - Stress testing CPU using the `div16` method on variable number of virtual CPU cores
- [stress-ng-longrun](images/stress-ng-longrun) - 24-hour stress-ng longrun benchmark
- [virtualization](images/virtualization) - Check if virtualization is supported on the host machine via KVM
