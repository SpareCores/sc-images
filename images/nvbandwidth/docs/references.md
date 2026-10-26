# References

Terms specific to this image, alphabetized. Common terms (e.g. CUDA, GPU,
NVLink, PCIe) are defined in the [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs, which also lists the terms each image uses.

- **`nvbandwidth`**: An open-source NVIDIA tool that measures memory copy
  bandwidth and latency between the CPU and NVIDIA GPUs, between GPUs, and
  within one GPU. It copies with the GPU's copy engines or with kernels on its
  streaming multiprocessors, and reports one matrix per test, with a row and a
  column for each CPU or GPU taking part. It prints plain text or JSON, and an
  optional multinode build measures GPUs across several servers.
  [GitHub](https://github.com/NVIDIA/nvbandwidth)
- **nvbandwidth testcase**: One named measurement of the `nvbandwidth` tool,
  such as `host_to_device_memcpy_ce`, defined by its copy direction, the GPUs
  taking part, and its copy method, given by the `_ce` or `_sm` suffix. A
  testcase that needs more GPUs than the system has is waived rather than run.
  `nvbandwidth -l` lists the testcases of a build.
  [nvbandwidth README](https://github.com/NVIDIA/nvbandwidth/blob/main/README.md)
