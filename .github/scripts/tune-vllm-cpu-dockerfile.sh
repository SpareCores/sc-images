#!/usr/bin/env bash
# Patch upstream vLLM CPU Dockerfile: cache-stable ENV + runtime parallelism via BuildKit secret.
set -euo pipefail

DOCKERFILE="${1:?dockerfile path}"
VLLM_VERSION="${2:?vllm version}"
DOCKER_MAX_JOBS="${3:?docker max_jobs (64 GiB reference)}"
DOCKER_CARGO_JOBS="${4:?docker cargo jobs (64 GiB reference)}"

if [[ ! -f .dockerignore ]]; then
  echo '.git' > .dockerignore
elif ! grep -qxF '.git' .dockerignore; then
  echo '.git' >> .dockerignore
fi

export DOCKERFILE VLLM_VERSION DOCKER_MAX_JOBS DOCKER_CARGO_JOBS
python3 <<'PY'
import os
import re
from pathlib import Path

df = Path(os.environ["DOCKERFILE"])
text = df.read_text()
version = os.environ["VLLM_VERSION"]
max_jobs = os.environ["DOCKER_MAX_JOBS"]
cargo = os.environ["DOCKER_CARGO_JOBS"]

if "ENV SETUPTOOLS_SCM_PRETEND_VERSION=" in text:
    text = re.sub(
        r"^ENV SETUPTOOLS_SCM_PRETEND_VERSION=.*$",
        f"ENV SETUPTOOLS_SCM_PRETEND_VERSION={version}",
        text,
        count=1,
        flags=re.M,
    )
else:
    text = text.replace(
        "ENV MAX_JOBS=${max_jobs}",
        f"ENV MAX_JOBS=${{max_jobs}}\nENV SETUPTOOLS_SCM_PRETEND_VERSION={version}",
        1,
    )

text = re.sub(r"^ARG max_jobs=.*$", f"ARG max_jobs={max_jobs}", text, flags=re.M)
text = re.sub(r"^ENV CARGO_BUILD_JOBS=4$", f"ENV CARGO_BUILD_JOBS={cargo}", text, flags=re.M)

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
    if secret_mount in run_block:
        return run_block
    if re.search(r"bash build_rust\.sh", run_block):
        run_block = run_block.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        return run_block.replace(
            "VLLM_RS_TARGET_PATH=",
            f"{load_cargo_env} \\\n    VLLM_RS_TARGET_PATH=",
            1,
        )
    if re.search(r"setup\.py bdist_wheel", run_block):
        run_block = run_block.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        return run_block.replace(
            "VLLM_TARGET_DEVICE=",
            f"{load_env} \\\n    VLLM_TARGET_DEVICE=",
            1,
        )
    return run_block

lines = text.splitlines(keepends=True)
for start, end, block in sorted(iter_run_blocks(text), key=lambda t: t[0], reverse=True):
    patched = patch_run(block)
    if patched != block:
        lines[start:end] = [patched]
text = "".join(lines)
df.write_text(text)
print(re.search(r"^ENV SETUPTOOLS_SCM_PRETEND_VERSION=.*$", text, re.M).group(0))
PY
