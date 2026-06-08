#!/usr/bin/env bash
# Patch upstream vLLM GPU Dockerfile: cache-stable ARG/ENV + runtime parallelism via BuildKit secret.
set -euo pipefail

DOCKERFILE="${1:?dockerfile path}"
VLLM_VERSION="${2:?vllm version}"
DOCKER_MAX_JOBS="${3:?docker max_jobs (64 GiB reference)}"
DOCKER_NVCC_THREADS="${4:?docker nvcc_threads (64 GiB reference)}"
DOCKER_CARGO_JOBS="${5:?docker cargo jobs (64 GiB reference)}"

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
export DOCKERFILE VLLM_VERSION DOCKER_MAX_JOBS DOCKER_NVCC_THREADS DOCKER_CARGO_JOBS SCRIPTS
export SCCACHE_PREFIX="${SCCACHE_PREFIX:-}"
python3 <<'PY'
import importlib.util
import os
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "vscm", os.path.join(os.environ["SCRIPTS"], "vllm-sccache-dockerfile.py"))
vscm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vscm)

df = Path(os.environ["DOCKERFILE"])
text = df.read_text()
version = os.environ["VLLM_VERSION"]
max_jobs = os.environ["DOCKER_MAX_JOBS"]
nvcc = os.environ["DOCKER_NVCC_THREADS"]
cargo = os.environ["DOCKER_CARGO_JOBS"]

if "ARG SCCACHE_S3_KEY_PREFIX" not in text:
    text = text.replace(
        "ARG SCCACHE_S3_NO_CREDENTIALS=0",
        "ARG SCCACHE_S3_KEY_PREFIX\nARG SCCACHE_S3_NO_CREDENTIALS=0",
        1,
    )
if "export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX}" not in text:
    text = text.replace(
        "export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \\",
        "export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \\\n && export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX} \\",
        1,
    )

text = vscm.inject_after_stage_header(
    text,
    "FROM ${BUILD_BASE_IMAGE} AS rust-build",
    "ARG TARGETPLATFORM\n" + vscm.SCCACHE_ARG_ENV_BLOCK,
)
text = vscm.inject_before_run(
    text,
    "# Build the release binary. Cache cargo registry/git and target/",
    vscm.SCCACHE_INSTALL_RUN,
)
csrc_sccache_anchor = (
    "ARG SCCACHE_S3_NO_CREDENTIALS=0\n\n"
    "# Flag to control whether to use pre-built vLLM wheels"
)
if vscm.SCCACHE_COMPILER_ENV_BLOCK not in text:
    text = text.replace(
        csrc_sccache_anchor,
        "ARG SCCACHE_S3_NO_CREDENTIALS=0\n\n"
        + vscm.SCCACHE_COMPILER_ENV_BLOCK
        + "\n\n# Flag to control whether to use pre-built vLLM wheels",
        1,
    )

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
    f"_vllm_pf={secret_path} && \\\n"
    f"    _mj=$(sed -n 's/^MAX_JOBS=//p' \"$_vllm_pf\" 2>/dev/null | tr -d '\\r\\n' | head -1) && \\\n"
    f"    _nt=$(sed -n 's/^NVCC_THREADS=//p' \"$_vllm_pf\" 2>/dev/null | tr -d '\\r\\n' | head -1) && \\\n"
    f"    _cj=$(sed -n 's/^CARGO_BUILD_JOBS=//p' \"$_vllm_pf\" 2>/dev/null | tr -d '\\r\\n' | head -1) && \\\n"
    "    [ -n \"$_mj\" ] && export MAX_JOBS=\"$_mj\" || true && \\\n"
    "    [ -n \"$_nt\" ] && export NVCC_THREADS=\"$_nt\" || true && \\\n"
    "    [ -n \"$_cj\" ] && export CARGO_BUILD_JOBS=\"$_cj\" || true &&"
)
load_cargo_env = (
    f"_vllm_pf={secret_path} && \\\n"
    f"    _cj=$(sed -n 's/^CARGO_BUILD_JOBS=//p' \"$_vllm_pf\" 2>/dev/null | tr -d '\\r\\n' | head -1) && \\\n"
    "    [ -n \"$_cj\" ] && export CARGO_BUILD_JOBS=\"$_cj\" || true &&"
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
    if re.search(r"bash build_rust\.sh", run_block):
        if vscm.AWS_SECRET_MOUNT not in run_block:
            run_block = run_block.replace("RUN ", f"RUN {vscm.AWS_SECRET_MOUNT} \\\n    ", 1)
        if secret_mount not in run_block:
            run_block = run_block.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        return run_block.replace(
            "VLLM_RS_TARGET_PATH=",
            f"{load_cargo_env} \\\n    VLLM_RS_TARGET_PATH=",
            1,
        )
    if secret_mount in run_block:
        return run_block
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
