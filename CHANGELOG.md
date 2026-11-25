## November 2026

Fixes:

- `stress-ng-longrun`: stress-ng 0.17.08 built from source on Debian instead of the
  Alpine-based upstream image, so the glibc-linked Resource Tracker can start and the
  image runs again (#43)

## October 2026

New images:

- `benchmark-storage`: raw block device speed and latency with fio 3.43, using O_DIRECT
  I/O on each group of identical disks (one representative alone, then all members
  together): sequential and random read, write and 70/30 mixed tests at block sizes from
  512 B to 16 MiB, with latency percentiles. Writable disks are preconditioned with a
  discard and a full sequential fill, and SMART and NVMe health snapshots are taken
  before and after the run. Needs `--privileged --pid=host` and refuses to run without
  the host PID namespace; it unmounts and overwrites the disks it tests

Breaking changes:

- `benchmark-ffmpeg`: NVENC encodes (`gpu_h264_encode`, `gpu_h265_encode`) use
  `-rc vbr -cq 22` instead of the default rate control, calibrated to match the quality
  of the CPU encodes at `-crf 18`; harness version 3.3.0, GPU encode results are not
  comparable with earlier runs

Improvements:

- `benchmark-ffmpeg`: FFmpeg built without `--enable-nonfree`, so the binary is
  redistributable; the NVENC and CUVID support is unchanged

Fixes:

- `benchmark-ffmpeg`: GPU scenarios run where they were skipped with
  `Cannot load libnvcuvid.so.1`, as the image now requests the NVIDIA driver's video
  (NVENC and NVDEC) libraries
- `benchmark-pgbench-postgres`: a truncated dataset download from the CDN fails the
  restore instead of passing silently with a partial schema

## September 2026

Improvements:

- `resource-tracker`: Resource Tracker 0.1.17 (was 0.1.14) in all images that run their
  workload under it
- `benchmark-pgbench-postgres`: the local PostgreSQL server runs at `nice -n -20`, like
  the other benchmark images; needs `--privileged`

Fixes:

- `nvbandwidth`: runs on GPU hosts where NVML failed to initialize, as the CUDA 12.6
  stub and compat libraries no longer shadow the host driver's libraries

## August 2026

New images:

- `benchmark-ffmpeg`: FFmpeg 9.0.1 transcoding throughput with independent
  single-threaded jobs started together, scaled up from the physical core count until
  throughput stops growing: H.264 and H.265 encode (libx264/libx265 at `-crf 18`) and
  decode on CPUs, the same with NVENC and CUVID on NVIDIA GPUs (amd64 and arm64), and
  Vorbis, Opus, AAC, MP3 and FLAC audio encodes; Tears of Steel video fixtures, output
  version 3 with raw per-repetition counts, times and rates only
- `benchmark-pgbench-postgres`: PostgreSQL 18 benchmark with pgbench on amd64 and
  arm64, against a local server tuned with pgtune or a remote one: a cached, CPU-heavy
  read-only transaction (`pgbench_ro`, eight query blocks covering joins, regex,
  full-text search, arrays, statistics and TOAST) at 1, V/2, V and 2V clients, scored
  in transactions per minute with latency, plus the built-in TPC-B-like workload
- `benchmark-postgres-server`: PostgreSQL 18 server at `nice -n -20` under Resource
  Tracker, the remote target of `benchmark-pgbench-postgres`

Breaking changes:

- `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`, `benchmark-vllm-gpu`: vLLM 0.27.1
  (was 0.22.1) and GuideLLM 0.7.3 (was 0.6.0) with tuning version 9 (was 3); results
  are not comparable with earlier runs
- `benchmark-vllm-gpu`: GPUs older than Ampere (compute capability below 8.0, such as
  the T4 and T4G) are no longer supported
- `benchmark`: Geekbench 6 results are the full upload document, printed to stderr
  between `GEEKBENCH_UPLOAD_DOCUMENT` markers, instead of the `--export-json` output;
  arm64 runs now report full results too

Improvements:

- `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`: NUMA-aware tensor and data
  parallelism using every NUMA node, with extra data-parallel replicas for small models
  sized from physical cores, explicit CPU binding and per-node KV cache budgets; new
  JSONL fields `data_parallel` and `omp_threads_bind`
- `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`, `benchmark-vllm-gpu`: memory
  preflight from the model's Hugging Face metadata (exact weight size and KV cache
  bytes per token), and new JSONL fields `runtime_memory`, `model_weight_gib`,
  `model_metadata_source` and `model_torch_dtype`

Fixes:

- `benchmark-vllm-cpu`: serves float32 on Arm CPUs without hardware bf16 (Graviton2:
  m6g, t4g, g5g), where float16 hung during warmup
- `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`: no KV cache out-of-memory failures
  when several data-parallel ranks share a NUMA node, such as on c8i hosts
- `benchmark-vllm-cpu`, `benchmark-vllm-cpu-avx2`, `benchmark-vllm-gpu`: the model
  ladder stops after a server fails its health check instead of moving on to larger
  models

## June 2026

New images:

- `benchmark-vllm-cpu`: LLM serving speed and latency with vLLM 0.22.1 and GuideLLM
  0.6.0 on arm64 CPUs and amd64 CPUs with AVX-512. Runs a model ladder from
  SmolLM2-135M to Phi-4 (models that fit in memory) with `chat` and `rag` workloads,
  sweeping load from one request at a time to peak throughput, with load and run time
  scaled to vCPU count and memory within a two-hour budget. Prints JSONL with TTFT,
  TPOT, ITL, end-to-end latency and throughput percentiles per model, workload and load
  stage
- `benchmark-vllm-cpu-avx2`: the same benchmark for amd64 CPUs with AVX2 but without
  AVX-512, on the `vllm-cpu-base-avx2` build
- `benchmark-vllm-gpu`: the same benchmark on NVIDIA GPUs (amd64 and arm64), adding a
  `long` workload and Llama-3.3-70B (bnb 4-bit), with tensor parallelism across GPUs
  where the model's attention heads allow it; JSONL rows record `tensor_parallel` and
  `gpu_count`
- `vllm-cpu-base-avx2`: vLLM 0.22.1 built for amd64 CPUs with AVX2 but without AVX-512,
  as the base of `benchmark-vllm-cpu-avx2`

Improvements:

- `benchmark`: Geekbench 6.7.1 (was 6.4.0)
- `resource-tracker`: resource-tracker 0.1.14 (was 0.1.7)

## May 2026

New images:

- `resource-tracker`: base image with the resource-tracker binary (0.1.7), which
  samples CPU, memory, disk, network and GPU usage once per second
- `stress-ng-longrun`: own image for the 24-hour sustained-load benchmark, which ran on
  the upstream `ghcr.io/colinianking/stress-ng` image since June 2024; pinned to
  stress-ng 0.17.08, the version of the earlier long runs, after `stress-ng` moved to
  0.20.01

Breaking changes:

- `stress-ng`: the entrypoint is `sh` under resource-tracker instead of `stress-ng`, so
  commands are passed as `-c 'stress-ng ...'`

Improvements:

- `benchmark`, `benchmark-llm`, `benchmark-passmark`, `benchmark-redis`,
  `benchmark-web`, `nvbandwidth`, `stress-ng`: workloads run under resource-tracker,
  which records resource usage during the benchmark
- `benchmark-redis`, `benchmark-web`: benchmarks run at `nice -n -20` priority (needs
  `--privileged`)
- `benchmark-llm`: llama-bench output is kept in stdout, and a failing llama-bench run
  (non-zero exit) skips the larger token counts and models instead of continuing

## March 2026

Breaking changes:

- `benchmark`: the compression benchmark reports lz4 `block_size` in bytes (65536,
  1048576, 4194304) instead of lz4's internal codes (0, 6, 7)

Improvements:

- `stress-ng`: stress-ng 0.20.01 (was 0.17.08) with `run_stressng_benchmarks.py`, a
  suite of 85 stressors in seven categories (CPU integer, CPU float/SIMD, crypto,
  compression, memory, algorithms, hypervisor overhead), each run with one worker and
  with one worker per CPU, reporting bogo ops/s as JSON

## February 2026

Breaking changes:

- `membench`: moved to the SpareCores/sc-membench repository

Improvements:

- `nvbandwidth`: nvbandwidth 0.8 (was 0.5)

## January 2026

New images:

- `membench`: memory bandwidth and latency with the Spare Cores membench tool 1.2.1,
  built with hwloc, NUMA and huge page support and run at `nice -n -20`
- `llama.cpp`: llama.cpp server (build b4753) with CUDA 12.4, built from source for
  amd64 and arm64

Improvements:

- `hwinfo`: hwloc's `lstopo` for the server's CPU and memory topology

Fixes:

- `benchmark-passmark`: PerformanceTest downloads from PassMark's renamed Linux
  packages, so the image builds again

## December 2025

Fixes:

- `benchmark-llm`: free disk space check before each model download, skipping models
  that do not fit (size plus 10 %) and benchmarking the rest instead of failing the run

## June 2025

Improvements:

- `benchmark-llm`: default models pinned to fixed Hugging Face revisions, so later
  uploads to the model repos no longer change what is benchmarked

## April 2025

Improvements:

- `benchmark-llm`: `--version` option that prints the llama.cpp build commit

Fixes:

- `benchmark-llm`: model download timeouts end the run with exit code 0, so the
  inspector keeps the results of the models already benchmarked

## February 2025

Breaking changes:

- `benchmark-llm`: shorter token ladders, 16 to 16,384 tokens for prompt processing
  (was 16 to 32,768) and 16 to 4,096 for text generation (was 1 to 8,192), and
  `--benchmark-timeout` replaced by `--benchmark-timeout-scale`

Improvements:

- `benchmark-llm`: llama.cpp build b4753 (was b4524), from the new `ggml-org` images
- `benchmark-llm`: per-run timeouts derived from model size, task and token count,
  with an early stop of the larger token counts when a run is unlikely to reach the
  expected tokens per second
- `benchmark-llm`: model download estimation that gives up early when a model cannot
  finish downloading within the new `--download-timeout` (default 15 minutes)

Fixes:

- `benchmark-llm`: runs on arm64, which failed while looking for the CUDA build

## January 2025

New images:

- `virtualization`: KVM hardware virtualization check with `kvm-ok`, reporting
  `{"kvm": true}` or `{"kvm": false, "kvm_error": ...}` as JSON
- `benchmark-llm`: LLM inference speed with llama.cpp (`llama-bench`, build b4524) on
  CPUs and NVIDIA GPUs, measuring prompt processing and text generation tokens per
  second for six GGUF models from SmolLM 135M to Llama 3.3 70B, with as many layers
  offloaded to the GPU as fit (#1)

Improvements:

- `benchmark`: OpenSSL 3.4.0 (was 3.3.0) and Geekbench 6.4.0 (was 6.3.0)

## November 2024

New images:

- `benchmark-passmark`: CPU and memory performance with PassMark PerformanceTest,
  running all suites, or only the CPU suite with less than 512 MiB available memory

## September 2024

New images:

- `nvbandwidth`: memory copy bandwidth between CPUs and NVIDIA GPUs and between GPUs
  with nvbandwidth 0.5 on CUDA 12.6.1
- `stress-ng`: stress-ng 0.17.08 with a helper that lists the worker counts to test,
  from one worker up to all CPUs

Breaking changes:

- `benchmark`: compression benchmark measured on one and on all CPUs for every
  compressor by running one single-threaded process per CPU, replacing the built-in
  multithreading of zstd and bzip3; best of two repeats (was three)

## August 2024

New images:

- `benchmark-redis`: Redis 7.4.0 SET throughput and latency with memtier_benchmark,
  one server and client pair per CPU at pipeline depths 1 to 512, with server and
  client CPU time, as CSV
- `benchmark-web`: static web server throughput, latency and failed requests, serving
  1 KiB to 512 KiB files with binserve and loading them with wrk 4.2.0 at thread and
  connection counts scaled by CPU count, with server and client CPU time, as CSV

Fixes:

- `benchmark`: LMbench `bw_mem` limited to the available memory and only valid
  numeric results reported, avoiding failed or garbled runs on large buffer sizes

## May 2024

New images:

- `dmidecode`: hardware details such as memory devices and firmware with dmidecode
- `hwinfo`: CPU, memory and topology details with lshw and LIKWID 5.3.0
- `benchmark`: general CPU and memory benchmarks, including Geekbench 6.3.0 (results
  uploaded on amd64, Preview build on arm64), OpenSSL 3.3.0 `speed` for hash and
  cipher algorithms on all CPUs, LMbench `bw_mem` read, write and read-write bandwidth
  from 16 KiB to 512 MiB, and compression speed and ratio of brotli, gzip, bzip2,
  lzma, zstd, bzip3, zpaq and lz4 at several levels on the Silesia `dickens` file
