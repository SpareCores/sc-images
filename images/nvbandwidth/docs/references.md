# References

Terms specific to this image, alphabetized. Common terms, such as CUDA (Compute
Unified Device Architecture), GPU, NVLink, and PCIe (Peripheral Component
Interconnect Express), are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs, which
also lists the terms each image uses.

- **nvbandwidth test case**: One named measurement of the `nvbandwidth` tool,
  such as `host_to_device_memcpy_ce`, defined by its copy direction, the GPUs
  taking part, and its copy method, given by the `_ce` or `_sm` suffix. A
  test case between GPUs is waived rather than run when no two GPUs of the
  system have P2P (Peer-to-Peer) access to each other, as on a system with
  one GPU.
  The `nvbandwidth -l` command lists the test cases of a build.
  [Official documentation](https://github.com/NVIDIA/nvbandwidth/blob/v0.8/README.md)
