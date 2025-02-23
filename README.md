# sc-images

Spare Cores container images for hardware inspection and benchmarking workloads:

- [benchmark](images/benchmark) - Hardware benchmarks for memory bandwidth, compression, OpenSSL crypto operations, and Geekbench tests
- [benchmark-llm](images/benchmark-llm) - LLM inference speed benchmarks using `llama.cpp` with different models and token lengths for prompt processing and text generation
- [benchmark-passmark](images/benchmark-passmark) - PassMark Performance Test benchmarking suite for CPU and memory performance
- [benchmark-redis](images/benchmark-redis) - Redis server performance benchmarks using `memtier_benchmark`
- [benchmark-web](images/benchmark-web) - Static web server performance benchmarks using `wrk` and `binserve`
- [dmidecode](images/dmidecode) - Hardware information reader using `dmidecode`
- [hwinfo](images/hwinfo) - Hardware information tools including `likwid` and `lshw`
- [nvbandwidth](images/nvbandwidth) - NVIDIA GPU memory bandwidth measurement tool
- [stress-ng](images/stress-ng) - Stress testing CPU using the `div16` method on variable number of virtual CPU cores
- [virtualization](images/virtualization) - Check if virtualization is supported on the host machine via KVM