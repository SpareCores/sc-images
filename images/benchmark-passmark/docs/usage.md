# Usage

## Results

This benchmark prints PassMark's YAML results file to stdout:
`results_all.yml` when it runs both suites, or `results_cpu.yml` when it runs
the CPU suite only. PassMark's interactive terminal interface, including its
terminal escape sequences, goes to stderr; redirect it to a file or
`/dev/null` if you do not need it.

The YAML document has the following top-level keys:

- `Version` holds the PassMark version, such as `Major: 11`, `Minor: 0`, and
  `Build: 1002`, and the architecture of the binary in `ptArchitecture`.
- `Results` holds the scores, along with `NumTestProcesses`, the number of
  test processes PassMark ran.
- `SystemInformation` holds the OS, kernel, processor, core counts, and memory
  that PassMark detected.
- `BaselineInfo` and `Checksum` are PassMark's own metadata for the results
  file.

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes the
following `Results` fields, each as a `passmark:<measurement>` benchmark.
Lower is better for Memory Latency; for the others, higher is better:

| Field | Navigator name | Unit |
| ----- | -------------- | ---- |
| `SUMM_CPU` | CPU Mark | composite score, no unit |
| `CPU_INTEGER_MATH` | CPU Integer Maths Test | Mops/s (Million Operations/s) |
| `CPU_FLOATINGPOINT_MATH` | CPU Floating Point Maths Test | Mops/s |
| `CPU_PRIME` | CPU Prime Numbers Test | Mnums/s (Million Primes/s) |
| `CPU_SORTING` | CPU String Sorting Test | Kstrings/s (Thousand Strings/s) |
| `CPU_ENCRYPTION` | CPU Encryption Test | MB/s |
| `CPU_COMPRESSION` | CPU Compression Test | kB/s |
| `CPU_SINGLETHREAD` | CPU Single Threaded Test | Mops/s |
| `CPU_PHYSICS` | CPU Physics Test | fps (Frames Per Second) |
| `CPU_MATRIX_MULT_SSE` | CPU Extended Instructions Test | Mmat/s (Million Matrices/s) |
| `SUMM_ME` | Memory Mark | composite score, no unit |
| `ME_ALLOC_S` | Database Operations | Kops/s (Thousand Operations/s) |
| `ME_READ_S` | Memory Read Cached | MB/s |
| `ME_READ_L` | Memory Read Uncached | MB/s |
| `ME_WRITE` | Memory Write | MB/s |
| `ME_LATENCY` | Memory Latency | ns |

Memory Read Cached, Memory Read Uncached, Memory Write, and Memory Latency
are largely independent of the vCPU count within an instance family; they
mainly reflect the hardware generation and memory characteristics.

The other `Results` fields, such as `CPU_avx`, `m_CPU_enc_AES`, `ME_LARGE`,
and `ME_THREADED`, are in the output but not published.

For example, on `aws/a1.2xlarge` (8 vCPUs), `SUMM_CPU` is \~3170 and `SUMM_ME`
is \~1418
([production output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/a1.2xlarge/0/passmark/stdout)).

## Environment Variables

This benchmark reads no environment variables.
