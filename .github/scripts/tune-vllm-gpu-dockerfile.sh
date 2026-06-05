#!/usr/bin/env bash
# Patch upstream vLLM GPU Dockerfile: cache-stable ARG/ENV + runtime parallelism via BuildKit secret.
set -euo pipefail

DOCKERFILE="${1:?dockerfile path}"
VLLM_VERSION="${2:?vllm version}"
DOCKER_MAX_JOBS="${3:?docker max_jobs (64 GiB reference)}"
DOCKER_NVCC_THREADS="${4:?docker nvcc_threads (64 GiB reference)}"
DOCKER_CARGO_JOBS="${5:?docker cargo jobs (64 GiB reference)}"

export DOCKERFILE VLLM_VERSION DOCKER_MAX_JOBS DOCKER_NVCC_THREADS DOCKER_CARGO_JOBS
python3 <<'PY'
import os
import re
from pathlib import Path

df = Path(os.environ["DOCKERFILE"])
text = df.read_text()
version = os.environ["VLLM_VERSION"]
max_jobs = os.environ["DOCKER_MAX_JOBS"]
nvcc = os.environ["DOCKER_NVCC_THREADS"]
cargo = os.environ["DOCKER_CARGO_JOBS"]

if "ARG vllm_ci_cache_bust=" not in text:
    text = text.replace("ARG max_jobs=", "ARG vllm_ci_cache_bust=0\nARG max_jobs=", 1)
text = re.sub(r"^ARG vllm_ci_cache_bust=.*$", f"ARG vllm_ci_cache_bust={version}", text, count=1, flags=re.M)
text = re.sub(r"^ARG max_jobs=.*$", f"ARG max_jobs={max_jobs}", text, flags=re.M)
text = re.sub(r"^ARG nvcc_threads=.*$", f"ARG nvcc_threads={nvcc}", text, flags=re.M)
text = re.sub(r"^ENV CARGO_BUILD_JOBS=4$", f"ENV CARGO_BUILD_JOBS={cargo}", text, flags=re.M)

if "ENV FLASHINFER_CUBIN_DOWNLOAD_THREADS=" not in text:
    text = text.replace(
        "RUN flashinfer show-config",
        "ENV FLASHINFER_CUBIN_DOWNLOAD_THREADS=32\nRUN flashinfer show-config",
        1,
    )

secret_path = "/tmp/vllm-parallelism.env"
secret_mount = f"--mount=type=secret,id=vllm_parallelism,target={secret_path}"
load_env = (
    f"MAX_JOBS=$(sed -n 's/^MAX_JOBS=//p' {secret_path} | tr -d '\\r\\n') && \\\n"
    f"    NVCC_THREADS=$(sed -n 's/^NVCC_THREADS=//p' {secret_path} | tr -d '\\r\\n') && \\\n"
    f"    CARGO_BUILD_JOBS=$(sed -n 's/^CARGO_BUILD_JOBS=//p' {secret_path} | tr -d '\\r\\n') && \\\n"
    "    export MAX_JOBS NVCC_THREADS CARGO_BUILD_JOBS &&"
)

def iter_run_blocks(content: str):
    lines = content.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if not lines[i].startswith("RUN "):
            i += 1
            continue
        start = i
        i += 1
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped == "":
                i += 1
                continue
            if not line.startswith((" ", "\t")) and not stripped.startswith("#"):
                break
            i += 1
        yield start, i, "".join(lines[start:i])

def patch_run(run_block: str) -> str:
    if secret_mount in run_block:
        return run_block
    if re.search(r"bash build_rust\.sh", run_block):
        run_block = run_block.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        return run_block.replace(
            "VLLM_RS_TARGET_PATH=",
            f"{load_env} \\\n    VLLM_RS_TARGET_PATH=",
            1,
        )
    if "export VLLM_DOCKER_BUILD_CONTEXT=1" in run_block:
        run_block = run_block.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        return run_block.replace(
            "export VLLM_DOCKER_BUILD_CONTEXT=1",
            f"{load_env} \\\n        export VLLM_DOCKER_BUILD_CONTEXT=1",
        )
    return run_block

lines = text.splitlines(keepends=True)
for start, end, block in sorted(iter_run_blocks(text), key=lambda t: t[0], reverse=True):
    patched = patch_run(block)
    if patched != block:
        lines[start:end] = [patched]
text = "".join(lines)
df.write_text(text)
print("Dockerfile parallelism (cache-stable ARG/ENV):")
for line in text.splitlines():
    if line.startswith(("ARG vllm_ci_cache_bust=", "ARG max_jobs=", "ARG nvcc_threads=", "ENV CARGO_BUILD_JOBS=", "ENV FLASHINFER")):
        print(line)
PY
