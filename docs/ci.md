# CI and Build Internals

## resource-tracker

Benchmark images copy the static
[resource-tracker](https://github.com/SpareCores/resource-tracker-rs) binary
from [`ghcr.io/sparecores/resource-tracker`](../images/resource-tracker). The
release version is defined in
[`images/resource-tracker/RESOURCE_TRACKER_VERSION`](../images/resource-tracker/RESOURCE_TRACKER_VERSION).

CI publishes per-arch tags (`main-amd64`, `main-arm64`) during the build matrix;
the per-level `merge` step then creates the `main` multi-arch manifest.
Benchmark image builds use
`RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main-<arch>` so
they can resolve the binary without waiting for the merged tag.

To rebuild resource-tracker and all its consumer images without registry
cache, use one of the following:

- For push builds, set the `FORCE_REBUILD_RESOURCE_TRACKER` env var in
  [`.github/workflows/push.yml`](../.github/workflows/push.yml) to `"true"`,
  and set it back to `"false"` once the consumers have been rebuilt.
- For a manual build, tick `force_rebuild_resource_tracker` under **Actions →
  Build and publish Spare Cores images → Run workflow**.

## CI Build Tracking

Every image build job runs `resource-tracker` in **standalone mode** (no
wrapped command, no `--pid`) while `docker buildx build` runs in parallel.
The `docker buildx` command only drives `buildkitd` over the API, so compiles
do not appear in a shell-wrapper process tree; without a tracked PID,
`resource-tracker` collects system-wide CPU, memory, disk, and network (see
[`Usage.md`](https://github.com/SpareCores/resource-tracker-rs/blob/main/resource-tracker-rs-book/src/Usage.md):
*"Leave unset [pid] to collect system-wide metrics only"*). Metrics are sent
to [Sentinel](https://sentinel.sparecores.com) when the repo secret
**`SENTINEL_API_TOKEN`** is set (Settings → Secrets and variables → Actions →
Secrets). Without the token, tracking runs locally only and Sentinel upload is
skipped.

Job metadata (mirrors [sc-inspector](https://github.com/SpareCores/sc-inspector)
benchmark runs):

| Variable | CI default |
|----------|------------|
| `TRACKER_PROJECT_NAME` | `sc-images` |
| `TRACKER_JOB_NAME` | matrix entry name, e.g. `benchmark-vllm-gpu (arm64)` |
| `TRACKER_TASK_NAME` | image folder, e.g. `benchmark-vllm-gpu` |
| `TRACKER_STAGE_NAME` | workflow name |
| `TRACKER_EXTERNAL_RUN_ID` | `github.run_id`-`github.run_attempt` |
| `TRACKER_CONTAINER_IMAGE` | `ghcr.io/sparecores/<folder>:main-<arch>` |
| `TRACKER_ORCHESTRATOR` | `github-actions` |
| `TRACKER_ENV` | `ci` |
| `TRACKER_QUIET` | `true` (no metric spam in build logs) |

Scripts:
[`install-resource-tracker.sh`](../.github/scripts/install-resource-tracker.sh),
[`run-with-resource-tracker.sh`](../.github/scripts/run-with-resource-tracker.sh),
[`build-push-image.sh`](../.github/scripts/build-push-image.sh).

## CI Runners

Push builds use GitHub-hosted runners (`ubuntu-latest` for amd64,
`ubuntu-24.04-arm` for arm64). The exception is
[`vllm-cpu-base-avx2`](../images/vllm-cpu-base-avx2), which always builds on
`[self-hosted, Linux, X64, workhorse]`
([`build-level.yml`](../.github/workflows/build-level.yml)).

Manual **Run workflow** checkboxes (default: off):

| Input | When enabled |
|-------|----------------|
| `self_hosted_x64` | amd64 → `[self-hosted, Linux, X64]` |
| `self_hosted_arm64` | arm64 → `[self-hosted, Linux, Arm64]` |

Images with a `ZRAM` metadata file run
[`.github/scripts/setup-zram.sh`](../.github/scripts/setup-zram.sh) before the
build (`PERCENT=125` by default → compressed swap ≈ 1.25× RAM). Currently
enabled for [`vllm-cpu-base-avx2`](../images/vllm-cpu-base-avx2)
(source-compiled CPU image) and [`benchmark-ffmpeg`](../images/benchmark-ffmpeg).
On **GitHub-hosted** Azure kernels, `zram` is in
`linux-modules-extra-$(uname -r)`
([Launchpad #1762756](https://bugs.launchpad.net/ubuntu/bionic/+source/linux-azure/+bug/1762756));
the script installs that package when needed. If the extra-modules deb is
missing from `apt` (kernel/image drift), it keeps the default `/swapfile`.
Tune the default percent via workflow `env` `ZRAM_PERCENT`, or override per
image with a numeric `ZRAM` value.

[`vllm-cpu-base-avx2`](../images/vllm-cpu-base-avx2) compile parallelism is
computed by
[`.github/scripts/compute-build-parallelism.sh`](../.github/scripts/compute-build-parallelism.sh)
(RAM-linear on self-hosted, conservative on GitHub-hosted). GPU images use the
multi-arch Hub [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai/tags)
base, so there is no in-repo CUDA source build.

## vLLM Source Builds

The following subsections apply only to source-built vLLM images, currently
[`vllm-cpu-base-avx2`](../images/vllm-cpu-base-avx2) on the self-hosted
`workhorse` runner.

### sccache

Images with a `SCCACHE` metadata file compile C++/CUDA through
[sccache](https://github.com/mozilla/sccache) with an S3 backend (bucket in
`us-west-2`, 90-day lifecycle).

Repo configuration (set by Pulumi after `pulumi up`):

| Name | Kind | Purpose |
|------|------|---------|
| `SCCACHE_BUCKET` | Actions variable | S3 bucket name |
| `SCCACHE_REGION` | Actions variable | Bucket region (`us-west-2`) |
| `SC_IMAGES_GHA_ROLE_ARN` | Actions secret | OIDC role for read/write on the bucket |

Each build uses prefix `{folder}/{arch}/` (e.g. `vllm-cpu-base-avx2/amd64/`)
via `SCCACHE_S3_KEY_PREFIX`. Docker layer cache is exported to **GHCR**
(`buildcache-<arch>`) and **GitHub Actions cache** (`type=gha`, scope
`{folder}-{arch}` or `{folder}-{arch}-v{VLLM_VERSION}` for source-built vLLM
CPU base). The `sccache` tool caches compiler objects inside the vLLM source
compile steps.

Coverage when enabled:

| Stage | Mechanism |
|-------|-----------|
| `rust-build` | `RUSTC_WRAPPER=/usr/bin/sccache` for `cargo`; `CC`/`CXX` absolute paths for build-script C compiles (`openssl-sys`, etc.) |
| `csrc-build` / `vllm-build` wheel | CMake compiler launchers via `setup.py` (`CMAKE_*_COMPILER_LAUNCHER=sccache`); `RUSTC_WRAPPER` unset so C++ is not double-wrapped |

Version 0.15.0 of `sccache` is installed to `/usr/bin/sccache`. The
`SCCACHE_IGNORE_SERVER_IO_ERROR=1` setting falls back to local compiles if the
cache server is unreachable.

Currently enabled for `vllm-cpu-base-avx2`.

### Sharing the Layer Cache Between GitHub- and Self-Hosted Runners

A slow image can be pre-warmed on a self-hosted runner and then finished
quickly on a GitHub-hosted one, but only if both produce identical BuildKit
cache keys. Two things break that by default, and both are handled for
source-built images:

- **Floating base tags and per-runner build args.** `ARG max_jobs` and
  `ENV CARGO_BUILD_JOBS` are baked to the 64 GiB reference values and
  `FROM ubuntu:22.04` is pinned by digest, so the Dockerfile is byte-identical
  everywhere
  ([`.github/scripts/tune-vllm-cpu-dockerfile.sh`](../.github/scripts/tune-vllm-cpu-dockerfile.sh)
  prints its `sha256`). Actual parallelism is injected at build time through
  the `vllm_parallelism` BuildKit secret, whose *content* is excluded from the
  cache key.
- **Build context ownership.** BuildKit hashes local contexts with a V1 tarsum
  covering name, mode, uid, gid, size, and content. It excludes `mtime` but not
  uid/gid. GitHub-hosted runners check out as `runner`, self-hosted ones as
  `ubuntu`, so identical sources would hash differently and every `COPY` (plus
  the entire graph below it) would miss.
  [`.github/scripts/normalize-context-ownership.sh`](../.github/scripts/normalize-context-ownership.sh)
  chowns the context to a canonical `0:0` before the build; `build-level.yml`
  reclaims workspace ownership on persistent runners before `actions/checkout`
  cleans it.

Changing any of the above invalidates the published `buildcache-<arch>`, so
re-warm once on a self-hosted runner before relying on it from GitHub-hosted
runners.

## Layer Compression

Published images use **zstd** layer compression (`oci-mediatypes=true`,
`force-compression=true`) so pulls on sc-inspector VMs decompress faster than
gzip (Docker's default). Requires Docker Engine ≥ 23.0 on pull clients.
Implemented in [`build-push-image.sh`](../.github/scripts/build-push-image.sh);
override with `BUILD_COMPRESSION=gzip` to revert to the legacy format.

CI push compression runs inside BuildKit's Go zstd encoder
([moby/buildkit#2345](https://github.com/moby/buildkit/issues/2345)); there is
no `zstd -T` equivalent exposed on `docker buildx build`. It uses limited
block-level concurrency (tied to `GOMAXPROCS` in the `buildkitd` container),
not `libzstd`'s multi-threaded mode.
