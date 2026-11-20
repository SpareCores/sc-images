# Contributor Guide

## Build Files

The image folder has the following files besides this guide and the main
page:

| File | Value | Effect |
| --- | --- | --- |
| `Dockerfile` | Two stages | Downloads the release and copies the binary into an empty image; see [Build Steps](#build-steps) |
| `RESOURCE_TRACKER_VERSION` | `0.1.17` | The release that CI builds; see [Version Changes](#version-changes) |
| `BUILD_ARGS` | `RESOURCE_TRACKER_VERSION=${RESOURCE_TRACKER_VERSION}` | CI replaces the token with the content of `RESOURCE_TRACKER_VERSION` |
| `CATEGORY` | `base` | Image category label |
| `DESCRIPTION` | One sentence | Image description on GHCR and in the root image list |

The folder has no `DEPENDS_ON`, `PLATFORMS`, `CONTEXT`, `ZRAM`, or `SCCACHE`
file, so CI builds it in the first level of the build plan, for the default
AMD64 and Arm64, from the image folder, without compressed swap or a compiler
cache. The
[build framework](../../../docs/contributors.md) of the
[`sc-images` repository](https://github.com/SpareCores/sc-images) describes
the metadata files that all images share.

## Build Steps

The [`Dockerfile`](../Dockerfile) builds the image in two stages:

1. The `fetch` stage starts from `alpine:3.21` and installs `curl` and `tar`.
2. It maps the output of `uname -m` to the architecture names of the
   releases: `x86_64` to `amd64`, and `aarch64` to `arm64`.
3. It downloads
   `resource-tracker-v<version>-linux-<arch>.tar.gz` from the
   [`resource-tracker-rs` releases](https://github.com/SpareCores/resource-tracker-rs/releases),
   extracts only the `resource-tracker` file, and installs it with mode
   `0755`. The `tar -xvzf` command lists the extracted file in the build log.
4. The final stage starts `FROM scratch` and copies in only the binary, as
   `/usr/local/bin/resource-tracker`. It sets no `ENTRYPOINT` or `CMD`.

The build fails when the release has no archive for the version or the
architecture: `curl -f` stops at the missing file. Any `uname -m` value other
than `x86_64` or `aarch64` goes into the file name unchanged.

## Version Changes

To change the version, update both of the following by hand, to the same
value:

- [`RESOURCE_TRACKER_VERSION`](../RESOURCE_TRACKER_VERSION), which CI reads
- the default of `ARG RESOURCE_TRACKER_VERSION` in the
  [`Dockerfile`](../Dockerfile), which a build without `--build-arg` uses

CI strips comments (`#...`) and whitespace from `RESOURCE_TRACKER_VERSION`
before it uses the value. When the file is empty or missing, CI still passes
`RESOURCE_TRACKER_VERSION=` to the build, which overrides the `Dockerfile`
default with an empty string, so the download fails.

The CI build tracking reads the same file to install `resource-tracker` on
the build runners, but strips only whitespace from it, so keep the file to the
bare version, with no comment. See
[CI Build Tracking](../../../docs/ci.md#ci-build-tracking).

## Local Build

To build the image for the architecture of the current machine, with the
version that the `Dockerfile` defaults to, run the following from the root of
the `sc-images` repository:

```bash
docker build -t resource-tracker images/resource-tracker
```

To build another release, pass its version without the leading `v`, such as
`0.1.17`:

```bash
docker build --build-arg RESOURCE_TRACKER_VERSION=0.1.17 -t resource-tracker images/resource-tracker
```

## CI Builds

CI builds this image natively on each architecture: AMD64 on `ubuntu-latest`
and Arm64 on `ubuntu-24.04-arm`, or on the self-hosted runners when the `self_hosted_x64` or `self_hosted_arm64`
option of a manual run is on. Later levels run only when no earlier level
failed, so a failed build of this image stops the build of every image that
depends on it.

### Consumers

The following images list `resource-tracker` in their `DEPENDS_ON` file,
grouped by the manual that covers them:

- **Own Manuals:** `benchmark`, `benchmark-ffmpeg`, `benchmark-llm`,
  `benchmark-passmark`, `benchmark-redis`, `benchmark-storage`,
  `benchmark-web`, `nvbandwidth`
- **vLLM:** `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`,
  `benchmark-vllm-gpu`
- **PostgreSQL:** `benchmark-pgbench-postgres`, `benchmark-postgres-server`
- **stress-ng:** `stress-ng`, `stress-ng-longrun`

Each of these images declares the following build argument in its
`Dockerfile`, and copies the binary from the image it names:

```dockerfile
ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
```

In CI, their `BUILD_ARGS` files set `RESOURCE_TRACKER_IMAGE` to the
per-architecture `ghcr.io/sparecores/resource-tracker:main-${ARCH}` tag, except
`benchmark-ffmpeg`, which keeps the `main` default.

CI counts an image as a `resource-tracker` consumer when it is this image or
depends on it, directly or through other images. Consumers always build with
`--pull`, so they pick up the newly pushed binary.

### Force Rebuild

The `force_rebuild_resource_tracker` option of a manual run, or the
`FORCE_REBUILD_RESOURCE_TRACKER` variable of
[`push.yml`](../../../.github/workflows/push.yml) on a push, rebuilds this
image and all consumers with `--no-cache`, without importing or exporting the
layer cache. The variable is `"false"`, so pushes use the cache. The option of
a manual run is on by default, so a manual run rebuilds all consumers without
cache unless it is turned off.

### Tags

The build pushes `ghcr.io/sparecores/resource-tracker:main-amd64` and
`main-arm64`, and a merge job then creates the multi-architecture `main` tag
from them. The merge job runs even when a build fails. When either tag is
missing, it leaves the `main` tag unchanged and prints
`Skipping <tag>: missing <ref>`. The merge job also makes the package public.

The image labels give the title, the description from
[`DESCRIPTION`](../DESCRIPTION), the documentation URL, and the category
`base`; the `main` tag carries the same values, and the source URL, as
annotations. The layers are pushed compressed with `zstd`, at level 7. The
layer cache is pushed to the same package, as the `buildcache-amd64` and
`buildcache-arm64` tags, except in a force rebuild.
