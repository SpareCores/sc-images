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
export SCCACHE_PREFIX="${SCCACHE_PREFIX:-}"
python3 <<'PY'
import os
import re
from pathlib import Path

df = Path(os.environ["DOCKERFILE"])
text = df.read_text()
version = os.environ["VLLM_VERSION"]
max_jobs = os.environ["DOCKER_MAX_JOBS"]
cargo = os.environ["DOCKER_CARGO_JOBS"]

sccache_arg_block = """
ARG USE_SCCACHE
ARG SCCACHE_BUCKET_NAME
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_KEY_PREFIX
ARG SCCACHE_S3_NO_CREDENTIALS=0
""".strip()

if "ARG USE_SCCACHE" not in text:
    text = text.replace(
        "ENV MAX_JOBS=${max_jobs}\n\nARG GIT_REPO_CHECK=0",
        f"ENV MAX_JOBS=${{max_jobs}}\n\n{sccache_arg_block}\n\nARG GIT_REPO_CHECK=0",
        1,
    )

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
aws_secret_mount = "--mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false"
sccache_setup = (
    'if [ "$USE_SCCACHE" = "1" ]; then '
    'SCCACHE_ARCH="x86_64"; '
    '[ "$TARGETARCH" = "arm64" ] && SCCACHE_ARCH="aarch64"; '
    'curl -fsSL -o /tmp/sccache.tar.gz '
    '"https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz" && '
    'tar -xzf /tmp/sccache.tar.gz -C /tmp && '
    'install -m 0755 /tmp/sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl/sccache /usr/local/bin/sccache && '
    'rm -rf /tmp/sccache.tar.gz /tmp/sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl && '
    'export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} && '
    'export SCCACHE_REGION=${SCCACHE_REGION_NAME} && '
    'export SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX} && '
    'export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} && '
    'export SCCACHE_IDLE_TIMEOUT=0 && '
    'unset CCACHE_DIR CMAKE_CXX_COMPILER_LAUNCHER; '
    'fi && '
)
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
    if re.search(r"setup\.py bdist_wheel", run_block):
        patched = run_block
        if aws_secret_mount not in patched:
            patched = patched.replace("RUN ", f"RUN {aws_secret_mount} \\\n    ", 1)
        if secret_mount not in patched:
            patched = patched.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        if sccache_setup not in patched:
            patched = patched.replace(
                "VLLM_TARGET_DEVICE=",
                f"{sccache_setup}\\\n    VLLM_TARGET_DEVICE=",
                1,
            )
        if load_env not in patched:
            patched = patched.replace(
                "VLLM_TARGET_DEVICE=",
                f"{load_env} \\\n    VLLM_TARGET_DEVICE=",
                1,
            )
        return patched
    if secret_mount in run_block:
        return run_block
    if re.search(r"bash build_rust\.sh", run_block):
        run_block = run_block.replace("RUN ", f"RUN {secret_mount} \\\n    ", 1)
        return run_block.replace(
            "VLLM_RS_TARGET_PATH=",
            f"{load_cargo_env} \\\n    VLLM_RS_TARGET_PATH=",
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
