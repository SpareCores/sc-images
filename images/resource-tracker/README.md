# Resource Tracker

This image carries the `resource-tracker` binary of
[Resource Tracker](https://github.com/SpareCores/resource-tracker-rs) and
nothing else. [Other images](./docs/contributors.md#consumers) of the
[`sc-images` repository](https://github.com/SpareCores/sc-images) copy the
binary from this image and run their workload under it.

The `resource-tracker` binary samples the workload's CPU, memory, disk,
network, GPU, and GPU memory use once per second. This image measures nothing
by itself.

Other images copy the binary from this image and run their workloads under it.
This image prints nothing. In production, Resource Tracker
[streams the samples](https://sparecores.com/article/resource-tracker-rs#streaming-to-sentinel)
to the [Spare Cores Sentinel](https://sentinel.sparecores.com/) platform for
the Spare Cores team's own monitoring. Spare Cores does not publish these
samples.

## Purpose

[Resource Tracker](https://github.com/SpareCores/resource-tracker) is a
separate Spare Cores product, not part of
[Spare Cores Navigator](https://docs.sparecores.com/overview). This image
carries its Rust implementation,
[`resource-tracker-rs`](https://github.com/SpareCores/resource-tracker-rs).
The Spare Cores
team uses it to check that each Navigator benchmark uses the CPU or GPU cores
as intended.

Each benchmark image copies the binary from this image's `main` tag (in CI,
from `main-amd64` or `main-arm64`) when it is built, so every image built after
a version change gets the version that this image pins. This image publishes no
version tags.

## Usage

To build an image that runs its command under `resource-tracker`, as the
benchmark images do, use a `Dockerfile` such as the following:

```dockerfile
FROM ghcr.io/sparecores/resource-tracker:main AS resource_tracker

FROM ubuntu:24.04
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--"]
```

Replace `ubuntu:24.04` with the base image of the target image. The base
image needs glibc (GNU C Library) version 2.17 or newer, because the release
builds of Resource Tracker target glibc 2.17 and link it dynamically. With this
entrypoint, the arguments after the image name in `docker run` are the command
to track: the container runs `resource-tracker -- <command>`.

The binary cannot run inside this image. The final stage of the
[`Dockerfile`](./Dockerfile) starts `FROM scratch` and copies in only
`/usr/local/bin/resource-tracker`, so this image has no C library, shell,
entrypoint, or default command.

To run Resource Tracker on a host or in a project outside the `sc-images`
repository, download the binary from the
[`resource-tracker-rs` releases](https://github.com/SpareCores/resource-tracker-rs/releases)
instead.

The releases publish `linux-amd64` and `linux-arm64` archives, and the
project's
[documentation](https://github.com/SpareCores/resource-tracker-rs#readme)
describes its command-line options.

This image is published for AMD64 and Arm64, as the multi-architecture
`main` tag and the per-architecture `main-amd64` and `main-arm64` tags.

## Build

The [`Dockerfile`](./Dockerfile) downloads a prebuilt release of
Resource Tracker from its GitHub releases, and does not build it from
source. The version is 0.1.17, pinned in
the [`RESOURCE_TRACKER_VERSION`](./RESOURCE_TRACKER_VERSION) file, which CI
passes to the build as the build argument of the same name.

CI builds this image before any image that copies the binary from it.

## References

Common terms, such as `resource-tracker`, glibc, AMD64, and Arm64, are
defined in the [Spare Cores Glossary](https://docs.sparecores.com/glossary).

## Contributing

The [contributor guide](./docs/contributors.md) describes how to change the
pinned version and build this image locally, and how CI builds it and the
images that depend on it.
