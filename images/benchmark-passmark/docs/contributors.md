# Contributor Guide

## Local Development

Build the image locally from the `sc-images` root:

```bash
docker build -t benchmark-passmark:local images/benchmark-passmark
```

CI builds the image for `amd64` and `arm64`, the default platforms, because
the image has no `PLATFORMS` file. It copies the `resource-tracker` binary from
the image set by the `RESOURCE_TRACKER_IMAGE` build argument: the
[`Dockerfile`](../Dockerfile) defaults to
`ghcr.io/sparecores/resource-tracker:main`, and CI sets
`ghcr.io/sparecores/resource-tracker:main-${ARCH}` in
[`BUILD_ARGS`](../BUILD_ARGS).

## PassMark Download

The build downloads PassMark PerformanceTest for Linux from `passmark.com`,
with browser-like request headers, and installs the binary as
`/usr/local/bin/pt_linux`. The following build arguments set the download URLs:

- `INTEL_URL` (amd64):
  `https://www.passmark.com/downloads/PerformanceTest_Linux_x86-64.zip`
- `ARM64_URL` (arm64):
  `https://www.passmark.com/downloads/PerformanceTest_Linux_ARM64.zip`

These URLs always serve the latest release, so the PassMark build is not
pinned, and a rebuild can change it. PassMark publishes no versioned download
URLs, so pinning a build would mean keeping a copy of its archive in the
[`sc-images`](https://github.com/SpareCores/sc-images) repository. See
[PassMark Version](../README.md#passmark-version) for how this affects
results.

## Terminal Libraries

The image installs the legacy `libtinfo5` and `libncurses5` terminal libraries
for PassMark from `.deb` files in [`pkgs/`](../pkgs), because they might be
deleted from the package mirrors:

- amd64: version `6.4-2`
- arm64: version `6.2-0ubuntu2.1`
