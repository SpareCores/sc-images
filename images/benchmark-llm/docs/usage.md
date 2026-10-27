# Usage

## CPU Run

Run the image via Docker without options to measure the default models on the
CPU:

```bash
docker run --rm ghcr.io/sparecores/benchmark-llm:main > results.jsonl
```

No options or environment variables are required. The Arm64 image always uses
the CPU build of llama.cpp, and the AMD64 image uses it when the CUDA (Compute
Unified Device Architecture) build cannot start, for example without GPUs.

`--version` prints the commit hash of the llama.cpp build, `51f311e0`, and
exits:

```bash
docker run --rm ghcr.io/sparecores/benchmark-llm:main --version
```

## GPU Run

To run on NVIDIA GPUs, the host needs the NVIDIA driver and the
NVIDIA Container Toolkit. Pass the GPUs to the container:

```bash
docker run --rm --gpus all ghcr.io/sparecores/benchmark-llm:main > results.jsonl
```

The CUDA build is in the AMD64 image only. At startup,
[`benchmark.py`](../benchmark.py) runs `llama-cli --version` from the CUDA
build: if it succeeds, this benchmark uses the CUDA build, and otherwise the
CPU build, for example on a non-NVIDIA GPU or with NVIDIA driver issues. This
fallback runs inside the container, so it does not replace the host
requirements above. The choice is logged to stderr, and each output line
records it in `backends` (`CUDA` or `CPU`).

With the CUDA build, this benchmark then finds how many layers of each model
fit on the GPUs; see [GPU Layers](./workloads.md#gpu-layers).

## Model Downloads and Disk Space

This benchmark downloads the models from Hugging Face while it runs, so it
needs network access to `huggingface.co`, unless the model files are already in
`--models-dir` (default `/models`). The downloads run in the background, one
model after another, while the models already downloaded are measured. Each
file is written as `<file>.part` and renamed when it is complete.

`/models` is a Docker volume, so a plain run downloads the models again each
time. To reuse them across runs, mount a host directory on it:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/sparecores/benchmark-llm:main > results.jsonl
```

All six default models take up to \~58 GB. Before each download, this benchmark
reads the file size from Hugging Face and skips the model if the free space is
less than the file size plus 10%. The check reads the free space of the
container's root filesystem (`/`), not of `--models-dir`, so with a separately
mounted models directory, it does not reflect that directory's free space.

`--download-timeout` (default 900s) limits the wait for each model. Once at
least one download has finished, this benchmark also estimates from the
average download speed whether the next model can finish downloading within
that time. If a model would not finish, or has not finished when the wait
ends, this benchmark stops the downloads and exits with code 0, skipping that
model and all larger ones.

On `SIGINT` or `SIGTERM`, this benchmark deletes the incomplete `.part` files
and exits with code 128 plus the signal number.

## Run Duration

Run time depends on the server's speed, on the download speed, and on how many
models and token counts fit in their timeouts; see
[Timeouts and Early Stops](./workloads.md#timeouts-and-early-stops). Sample
production runs took the following times:

- AWS (Amazon Web Services) `c7g.large`: 18 min, stopped at the 7B model
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/llm/meta.json))
- AWS `c9g.48xlarge`: 26 min, stopped at the 70B model's download
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c9g.48xlarge/0/llm/meta.json))
- AWS `m7i.2xlarge`: 31 min
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7i.2xlarge/0/llm/meta.json))
- AWS `g4dn.xlarge` with an NVIDIA GPU: 32 min
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g4dn.xlarge/0/llm/meta.json))

To give slower servers more time, raise `--benchmark-timeout-scale`, which
multiplies every measurement's timeout. The options passed to `llama-bench`
stay the same, so such a run can report results for models and token counts
that a default run skips.

## Production Setup

The Inspector of Spare Cores Navigator
([`sc-inspector`](https://github.com/SpareCores/sc-inspector)) runs this image
on every server type except those on the
[`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
blocklist, with the following settings. A plain `docker run` differs from it as
noted below:

- The container runs privileged and with host networking. A plain
  `docker run` uses neither.
- On servers with GPUs, all GPUs are passed to the container, equivalent to
  `--gpus all`. If that run fails or times out, it is retried without GPUs.
- The task timeout is 1.5 hours. The image itself sets no overall timeout,
  only per-measurement and download timeouts.
- Production skips servers with less than 1 GiB of memory. The image does not
  enforce this minimum.
- No options are passed, so production uses the default models and timeouts.

We investigate failed production runs, manually or semi-automatically, and
then run them again. Until then, a server type's latest result in
`sc-inspector-data` can be a failed or partial run.

## Options

Pass the following options after the image name. This benchmark has no
environment variables of its own. See [Results](../README.md#results) for the
output fields.

| Option | Meaning | Default |
| --- | --- | --- |
| `--model-urls` | URLs of GGUF model files to download and measure, in order. Order them from smallest to largest: a model that fails its first measurement stops the run. | The six [default models](./workloads.md#models) |
| `--models-dir` | Directory to download and cache the models in. | `/models` |
| `--benchmark-timeout-scale` | Integer that multiplies every measurement's timeout. | `1` |
| `--download-timeout` | Maximum wait for each model download, in seconds, and the horizon of the download estimate. | `900` |
| `--version` | Print the llama.cpp commit hash and exit; `unknown` if it cannot be read. | Off |
