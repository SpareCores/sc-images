# Contributor Guide

## Build Files

This image has no `Dockerfile` of its own. CI builds the upstream vLLM CPU
`Dockerfile` from a vLLM source tree that the [`prepare.sh`](../prepare.sh)
hook clones and patches. The metadata files of the image folder set the
following:

| File | Value | Effect |
| --- | --- | --- |
| `CONTEXT` | `vllm-common/.vllm-src` | Build context: the cloned vLLM source tree |
| `DOCKERFILE` | `vllm-common/.vllm-src/docker/Dockerfile.cpu` | The patched upstream CPU `Dockerfile` |
| `TARGET` | `vllm-openai` | The upstream serving stage |
| `PLATFORMS` | `amd64` | Only AMD64 is built |
| `BUILD_ARGS` | `VLLM_CPU_AVX2=true` | Passed as a build argument; the upstream `Dockerfile` of vLLM 0.27.1 declares no such argument, so it has no effect |
| `ZRAM` | `true` | Compressed swap on the build runner; see [CI Builds](#ci-builds) |
| `SCCACHE` | `true` | Compiler cache in S3; see [Compiler Cache](#compiler-cache) |
| `CATEGORY` | `base` | Image category label |

The folder has no `DEPENDS_ON` file. CI finds the folder by its `prepare.sh`,
because it has no `Dockerfile`. Git ignores the cloned source tree and the
`.cache/` folder of the hook's output.

The [build framework](../../../docs/contributors.md) of the
[`sc-images` repository](https://github.com/SpareCores/sc-images) describes
the metadata files that all images share.

## Pre-Build Hook

CI runs `prepare.sh` before the build, because the file exists. The hook does
the following, in order:

1. Reads the vLLM version from
   [`vllm-common/VLLM_VERSION`](../../../vllm-common/VLLM_VERSION), currently
   `0.27.1`.
2. Sizes the compile parallelism (see
   [Compile Parallelism](#compile-parallelism)), and writes the runtime
   `MAX_JOBS`, `NVCC_THREADS`, and `CARGO_BUILD_JOBS` to
   `.cache/vllm-parallelism.env`, which the build mounts as the BuildKit
   secret `vllm_parallelism`.
3. Clones the vLLM tag `v<VLLM_VERSION>` with depth 1 into
   `vllm-common/.vllm-src`, or fetches and checks out the tag when the tree
   exists.
4. Patches `docker/Dockerfile.cpu` in that tree with
   [`tune-vllm-cpu-dockerfile.sh`](../../../.github/scripts/tune-vllm-cpu-dockerfile.sh);
   see [Dockerfile Patches](#dockerfile-patches).
5. Prints `dockerfile_sha256=<hash>` of the patched file.
6. Changes the owner of the source tree to `0:0` with `sudo -n chown`, so that
   the BuildKit context hash is the same on GitHub-hosted and self-hosted
   runners. `CONTEXT_CANONICAL_OWNER` overrides the owner.
7. Prints the `sccache`, install, and `FROM` lines of the patched file, in
   collapsed log groups.
8. Appends `max_jobs=32` and, when the compiler cache is on, its build
   arguments to `.cache/extra-build-args`.

Without passwordless `sudo`, step 6 prints
`normalize-context-ownership: WARNING could not chown ... layer cache will not be shared across runner types`
and the hook continues. When the tree is missing, step 6 exits with code 1.

## Dockerfile Patches

The `tune-vllm-cpu-dockerfile.sh` script makes the following changes to the
upstream `Dockerfile`, and fails the build when one of them cannot be
applied:

- Adds `.git` to the `.dockerignore` file of the build context.
- Sets `SETUPTOOLS_SCM_PRETEND_VERSION` to the vLLM version, and prints the
  resulting `ENV` line.
- Sets `ARG max_jobs` and `ENV CARGO_BUILD_JOBS` to 32, the values of a
  builder with 64 GiB of RAM (upstream `CARGO_BUILD_JOBS` is 4), so that the
  `Dockerfile` text and its layer cache keys stay the same on every runner.
- Makes the `RUN` steps of `setup.py bdist_wheel`, `build_rust.sh`, and
  `triton-cpu` read the runtime parallelism from the `vllm_parallelism`
  secret, whose content is not part of the layer cache key.
- Adds `git submodule update --init --recursive` after the `triton-cpu`
  checkout, because an empty `third_party/sleef` folder breaks the vLLM build.
- Pins both `ubuntu:22.04` stages, `base-common` and `rust-build`, to the
  digest
  `sha256:79676deb51ebb02885b0b9d33788e78a37cf1045ad79d1bb04c6a222c3556b3d`
  (resolved on 2026-08-21), and fails when an unpinned `ubuntu:22.04` `FROM`
  line remains.
- Adds the compiler cache to the `vllm-build` and `rust-build` stages; see
  [Compiler Cache](#compiler-cache).

## Compile Parallelism

The hook runs
[`compute-build-parallelism.sh`](../../../.github/scripts/compute-build-parallelism.sh)
with the `vllm-cpu` profile, which sizes the build as follows:

- `MAX_JOBS` = min((32 × `effective_GiB`) / 64, `nproc`,
  `MemTotal` / `VLLM_CPU_COMPILE_GIB_PER_SLOT`, 32)
- `CARGO_BUILD_JOBS` = min((32 × `effective_GiB`) / 64, `MAX_JOBS`, 32)
- BuildKit `max-parallelism` = min(`MAX_JOBS`, 16)
- `NVCC_THREADS` = 1

`VLLM_CPU_COMPILE_GIB_PER_SLOT` is the memory of one compile job in GiB
(default: 6). `MemTotal` is the physical RAM, and `effective_GiB` is
`MemTotal` plus 50% of the swap, rounded to the nearest GiB. The memory slots
(the third term) count physical RAM only, so the compressed swap raises the
first term but adds no slots. All divisions are integer divisions, and the
scaled terms and the memory slots are at least 1.

A `VLLM_CPU_COMPILE_GIB_PER_SLOT` of `0` or a value that is not an integer
makes the script fail, and the hook with it. A negative value sets `MAX_JOBS`
to 1.

The script prints the following three lines to stderr:

- the memory and CPU inputs, after `Parallelism (vllm-cpu):`
- the runtime values, after `runtime:`
- the values written to the `Dockerfile`, after `docker (cache-stable):`

On GitHub-hosted runners, the script caps `MAX_JOBS`, `CARGO_BUILD_JOBS`, and
the BuildKit parallelism at 2, but CI never builds this image there.

## CI Builds

CI builds this image on every push to `main`, in the first level of the build
plan, and then builds
[`benchmark-vllm-cpu-avx2`](../../benchmark-vllm-cpu-avx2/README.md) on top
of it. The build runs as follows:

- **Runner:** Always the self-hosted `[self-hosted, Linux, X64, workhorse]`
  runner. The workflow notes that the build is too heavy for GitHub-hosted
  runners, citing out-of-memory errors, disk pressure, and an unreliable
  compiler cache.
- **Compressed swap:** CI runs
  [`setup-zram.sh`](../../../.github/scripts/setup-zram.sh) before the build,
  with the default `ZRAM_PERCENT=125`: a `zram` swap of 125% of the RAM,
  compressed with `zstd`. The script exits with code 0 without `zram`, and
  keeps the existing swap, when the runner has no `apt-get` or the kernel has
  no `zram` module. It exits with code 1 when `zram` cannot start or no
  `zram` device shows in `swapon`.
- **Build command:** CI prints the full `docker buildx build` command after
  `build-push-image:`, and runs it under `run-with-resource-tracker.sh`.

### Tags

The build pushes `ghcr.io/sparecores/vllm-cpu-base-avx2:main-amd64`, and a
merge job then creates the `main` tag from it. The merge job runs even when
the build fails. When `main-amd64` is missing, it skips the `main` tag and
prints `Skipping <tag>: missing <ref>`. The package is public.

The image labels give the title, the description from
[`DESCRIPTION`](../DESCRIPTION), the documentation URL, and the category
`base`. The layers are pushed compressed with `zstd`, at level 7.

### Layer Cache

The build imports its layer cache from the following sources:

- `ghcr.io/sparecores/vllm-cpu-base-avx2:buildcache-amd64`
- `ghcr.io/sparecores/vllm-cpu-base-avx2:main-amd64`
- the GitHub Actions cache scope `vllm-cpu-base-avx2-amd64-v0.27.1`

It exports all layers (`mode=max`) to the `buildcache-amd64` tag and to the
same GitHub Actions cache scope. A new vLLM version starts a new scope.

The first successful build fills both caches. When the `buildcache-amd64` tag
does not exist yet, the build skips its import and still runs; see Docker's
[registry cache](https://docs.docker.com/build/cache/backends/registry/)
docs.

The force rebuild option of the workflow does not affect this image, because
it does not depend on `resource-tracker`: its cache stays on. The option
rebuilds `benchmark-vllm-cpu-avx2` without cache, because that image does
depend on `resource-tracker`. The option is off on pushes, and on by default
in a manual run.

## Compiler Cache

With `SCCACHE` set to `true`, CI runs the hook with `SCCACHE_ENABLED=true`,
and the build runs its compilers through `sccache` 0.15.0, with an S3 bucket
as its storage. The hook reads the following environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `SCCACHE_ENABLED` | `true` turns the compiler cache on | Unset; `true` in CI |
| `SCCACHE_BUCKET` | Name of the S3 bucket; the hook stops with `SCCACHE_BUCKET required when SCCACHE is enabled` when it is unset | Repository variable in CI |
| `SCCACHE_REGION` | Region of the bucket | `us-west-2` |
| `SCCACHE_PREFIX` | Key prefix of this image in the bucket | `vllm-cpu-base-avx2/amd64` |
| `SCCACHE_VERBOSE` | `1` turns on the trace logs | Unset |

From these, the hook writes the build arguments `USE_SCCACHE=1`,
`RUSTC_WRAPPER=/usr/local/bin/sc-rust-wrap`, `SCCACHE_BUCKET_NAME`,
`SCCACHE_REGION_NAME`, `SCCACHE_S3_NO_CREDENTIALS=0`,
`SCCACHE_S3_KEY_PREFIX`, and, when set, `SCCACHE_VERBOSE=1`. Without
`USE_SCCACHE=1`, the build skips the `sccache` install and its wrappers.

CI writes the AWS credentials of an assumed role to `.cache/aws-credentials`
and mounts them as an optional BuildKit secret. The
patched `Dockerfile` does the following with the cache:

- It downloads the `x86_64-unknown-linux-musl` build of `sccache` 0.15.0 from
  its GitHub releases, before the Rust and vLLM builds.
- In the `rust-build` stage, `cargo` runs each `rustc` call through the
  `/usr/local/bin/sc-rust-wrap` script
  ([`RUSTC_WRAPPER`](https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-reads)),
  which runs `sccache`. The script has its own name because a wrapper whose
  file name stem is `sccache` breaks the `openssl-sys` build. The stage also
  sets `CARGO_INCREMENTAL=0`, `CC=/usr/bin/gcc`, and `CXX=/usr/bin/g++`.
- The vLLM build compiles through `/usr/local/bin/sccache-wrapper`, which
  reruns a compile without `sccache` when `sccache` fails with one of the
  infrastructure errors that the wrapper lists, and prints
  `[sccache-wrapper] infrastructure failure rc=..., falling back`.

Outside the verbose mode, the build sets `SCCACHE_IGNORE_SERVER_IO_ERROR=1`,
so errors of the S3 storage do not fail the build, and the log shows only the
output of `sccache --show-stats` after each compile.

With `SCCACHE_VERBOSE=1`, the build sets `SCCACHE_IGNORE_SERVER_IO_ERROR=0`,
and the log also shows `[sccache-trace]` and `[sccache-debug]` lines and the
S3 errors. In this mode, the vLLM build still falls back to a plain compile on
the errors that `sccache-wrapper` lists, but the Rust build has no fallback.

## Local Build

The hook runs only on a host with `/proc/meminfo`, `nproc`, and `sha256sum`,
such as Linux; it stops on macOS. It also needs `git` to clone vLLM, and the
build needs Docker Buildx.

The hook writes `vllm-common/.vllm-src` and `.cache/` relative to the current
folder, so run it from the root of the `sc-images` repository. To clone and
patch the vLLM source:

```bash
bash images/vllm-cpu-base-avx2/prepare.sh
```

The hook leaves the source tree owned by root when passwordless `sudo` is
available. Outside CI, the build gets only the build arguments given with
`--build-arg`, and a local build needs none of those that CI passes: `VLLM_CPU_AVX2=true`
has no effect, `max_jobs=32` is already the default of the patched
`Dockerfile`, and without `SCCACHE_ENABLED=true`, the hook writes no compiler
cache arguments.

To build the image from the patched source, with the parallelism that the
hook computed for this machine:

```bash
docker buildx build --file vllm-common/.vllm-src/docker/Dockerfile.cpu --target vllm-openai --platform linux/amd64 --secret id=vllm_parallelism,src=.cache/vllm-parallelism.env --load -t vllm-cpu-base-avx2 vllm-common/.vllm-src
```

## Version Changes

This image takes its vLLM version from `vllm-common/VLLM_VERSION`, like the
other vLLM images; see the
[Version Changes](../../../vllm-common/docs/contributors.md#version-changes)
of the vLLM benchmarks. The hook clones the tag `v<VLLM_VERSION>` from the
[vLLM repository](https://github.com/vllm-project/vllm), and the build fails
when a patch no longer applies to the new `Dockerfile`.
