# Context: shared

Maintainer decisions that apply to every image. Per-image decisions stay
in `.agents/context/<image>.md`.

## What are these images for?

Spare Cores publishes container images that inspect hardware and measure the
performance of cloud servers. [Navigator](https://sparecores.com/servers)
publishes the empirical results for over 5,000 cloud server types. An image
README says what that image measures.

## Who is the target audience?

Highly technical readers who are somewhat familiar with cloud server types and
at least some workloads -- such as software engineers or system administrators.

## Where is a production run defined?

This repo defines the container: Dockerfile, entrypoint, and benchmark code.
Choosing server types and any host flag that is not in the Dockerfile (e.g.
privileged mode, host networking, ulimits, process priority) is orchestration.
Fleet runs are the concern of [sc-inspector](https://github.com/sparecores/sc-inspector)

## What is the public image name?

`ghcr.io/sparecores/<folder>:main`. `<folder>` is the directory name
under `images/`.

## What wraps the process inside the container?

Images copy `resource-tracker` and typically exec it as the entrypoint
(`resource-tracker -- <command>`), with `TRACKER_QUIET=true`. The image
README documents the workload command and its output. Tracker flags and
CI metrics belong to the resource-tracker docs and the repository root
README, not to each benchmark manual.

We use Resource Tracker to track the resource usage of each benchmark,
streamed to the Spare Cores Sentinel platform. This is useful for us to
check if all CPU or GPU cores etc are utilized as intended while running
the benchmarks.

## Which folders share one manual?

vLLM methodology lives in `vllm-common/`. The image folders
`benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`, `benchmark-vllm-gpu`,
and `vllm-cpu-base-avx2` document only the base image, CPU features,
GPU, or architecture.

`benchmark-pgbench-postgres` is the PostgreSQL benchmark manual.
`benchmark-postgres-server` is the server image (`postgres:18` plus
resource-tracker). It does not get a second methodology write-up.

`stress-ng-longrun` is a pinned older stress-ng image for a long run.
Its README says how that run and version differ from `stress-ng`. It
does not repeat a full stress-ng manual.
