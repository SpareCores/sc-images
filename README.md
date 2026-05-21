# sc-images

Spare Cores container images for hardware inspection and benchmarking workloads.

## resource-tracker

Benchmark images copy the static [resource-tracker](https://github.com/SpareCores/resource-tracker-rs) binary from [`ghcr.io/sparecores/resource-tracker:main`](images/resource-tracker). The release version is defined in [`images/resource-tracker/RESOURCE_TRACKER_VERSION`](images/resource-tracker/RESOURCE_TRACKER_VERSION).

All images under `images/` build and publish in parallel. Benchmark images copy from `resource-tracker:main`; on the first CI run, they may fail until that image exists—re-run the workflow after `resource-tracker` has been published.

Local build example (after `resource-tracker:main` exists in the registry):

```bash
docker buildx build images/benchmark
```

To build everything from scratch locally:

```bash
VERSION="$(tr -d '[:space:]' < images/resource-tracker/RESOURCE_TRACKER_VERSION)"
docker buildx build images/resource-tracker \
  --build-arg "RESOURCE_TRACKER_VERSION=${VERSION}" \
  --tag ghcr.io/sparecores/resource-tracker:main --push
docker buildx build images/benchmark
```

## Images

- [benchmark](images/benchmark) - Hardware benchmarks for memory bandwidth, compression, OpenSSL crypto operations, and Geekbench tests
- [benchmark-llm](images/benchmark-llm) - LLM inference speed benchmarks using `llama.cpp` with different models and token lengths for prompt processing and text generation
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
