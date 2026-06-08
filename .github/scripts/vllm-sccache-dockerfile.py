"""Shared Dockerfile fragments for vLLM sccache (used by tune-*.sh)."""

SCCACHE_VERSION = "0.15.0"
SCCACHE_BIN = "/usr/bin/sccache"
SCCACHE_WRAPPER = "/usr/local/bin/sccache-wrapper"

SCCACHE_COMMON_ARGS = """
ARG USE_SCCACHE
ARG SCCACHE_BUCKET_NAME
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_KEY_PREFIX
ARG SCCACHE_S3_NO_CREDENTIALS=0""".strip()

SCCACHE_COMMON_ENV = """
ENV SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME}
ENV SCCACHE_REGION=${SCCACHE_REGION_NAME}
ENV SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX}
ENV SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS}
ENV SCCACHE_IDLE_TIMEOUT=0
ENV SCCACHE_IGNORE_SERVER_IO_ERROR=1
ENV SCCACHE_LOG=sccache=info""".strip()

# rust-build: cache rustc + build.rs C compiles (openssl-sys, etc.).
# CC/CXX must be absolute paths so the cc crate does not fall back to "sccache cc".
# RUSTC_WRAPPER uses sccache directly (not the cmake fallback wrapper).
SCCACHE_RUST_ENV_BLOCK = (
    SCCACHE_COMMON_ARGS
    + f"""
ARG RUSTC_WRAPPER={SCCACHE_BIN}
ARG CC=/usr/bin/gcc
ARG CXX=/usr/bin/g++
ENV RUSTC_WRAPPER=${{RUSTC_WRAPPER}}
ENV CC=${{CC}}
ENV CXX=${{CXX}}
"""
    + SCCACHE_COMMON_ENV
).strip()

# csrc-build / vllm-build wheel: CMake launcher sccache via setup.py; no RUSTC_WRAPPER.
SCCACHE_COMPILER_ENV_BLOCK = (
    SCCACHE_COMMON_ARGS
    + """
ARG CC=/usr/bin/gcc-10
ARG CXX=/usr/bin/g++-10
ENV CC=${CC}
ENV CXX=${CXX}
"""
    + SCCACHE_COMMON_ENV
).strip()

# CPU vllm-build inherits CC/CXX from base (gcc-12); only bucket env needed.
SCCACHE_WHEEL_ENV_BLOCK = (SCCACHE_COMMON_ARGS + "\n" + SCCACHE_COMMON_ENV).strip()

# Back-compat alias used by older inject sites.
SCCACHE_ARG_ENV_BLOCK = SCCACHE_RUST_ENV_BLOCK

AWS_SECRET_MOUNT = (
    "--mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false"
)

# CMake compiler launcher wrapper: fall back only on sccache infrastructure errors,
# NOT on compiler failures (sccache propagates the compiler exit code).
_WRAPPER_INSTALL = (
    f'printf \'#!/bin/sh\\n'
    f'err=/tmp/sccache-err.$$\\n'
    f'/usr/bin/sccache "$@" 2>"$err"\\n'
    f'rc=$?\\n'
    f'if [ $rc -eq 0 ]; then rm -f "$err"; exit 0; fi\\n'
    f'if grep -qE "failed to execute compile|No such file or directory|Failed to send data|Failed to read response|Connection refused|Broken pipe|resource unavailable" "$err" 2>/dev/null; then\\n'
    f'  echo "[sccache-wrapper] infrastructure failure rc=$rc, falling back: $1" >&2\\n'
    f'  tail -5 "$err" >&2\\n'
    f'  rm -f "$err"\\n'
    f'  exec "$@"\\n'
    f'fi\\n'
    f'cat "$err" >&2 2>/dev/null\\n'
    f'rm -f "$err"\\n'
    f'exit $rc\\n\' > {SCCACHE_WRAPPER} && chmod +x {SCCACHE_WRAPPER}'
)

SCCACHE_RUST_PREP = (
    r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then \
        export RUSTC_WRAPPER="${RUSTC_WRAPPER:-""" + SCCACHE_BIN + r"""}" \
        && export CC="${CC:-/usr/bin/gcc}" \
        && export CXX="${CXX:-/usr/bin/g++}" \
        && export CARGO_INCREMENTAL=0 \
        && export SCCACHE_IGNORE_SERVER_IO_ERROR=1 \
        && echo "[sccache-debug] RUSTC_WRAPPER=$RUSTC_WRAPPER CC=$CC CXX=$CXX" \
        && echo "[sccache-debug] sccache binary:" && ls -la /usr/bin/sccache && file /usr/bin/sccache \
        && for _bd in /workspace/rust/target/*/build; do \
            if [ -d "$_bd" ]; then echo "[sccache-debug] clearing stale build.rs cache: $_bd" && rm -rf "$_bd"/*; fi; \
        done 2>/dev/null || true \
        && echo | "${CC}" -x c -E -P - >/dev/null \
        && echo | "${CXX}" -x c++ -E -P - >/dev/null \
        && /usr/bin/sccache --version \
        && sccache --start-server 2>/dev/null || true \
        && sccache --show-stats || true; \
    fi &&"""
)

SCCACHE_WHEEL_PREP = (
    r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then \
        unset RUSTC_WRAPPER \
        && export SCCACHE_IGNORE_SERVER_IO_ERROR=1 \
        && if [ ! -x """ + SCCACHE_WRAPPER + r""" ]; then \
            """ + _WRAPPER_INSTALL + r"""; \
        fi \
        && export CMAKE_C_COMPILER_LAUNCHER=""" + SCCACHE_WRAPPER + r""" \
        && export CMAKE_CXX_COMPILER_LAUNCHER=""" + SCCACHE_WRAPPER + r""" \
        && echo "[sccache-debug] CXX=${CXX:-/usr/bin/g++} CMAKE_CXX_COMPILER_LAUNCHER=$CMAKE_CXX_COMPILER_LAUNCHER" \
        && echo "[sccache-debug] sccache binary:" && ls -la /usr/bin/sccache \
        && echo | "${CXX:-/usr/bin/g++}" -x c++ -E -P - >/dev/null \
        && sccache --start-server 2>/dev/null || true \
        && sccache --show-stats || true; \
    fi &&"""
)

SCCACHE_WHEEL_STATS = r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then sccache --show-stats; fi"""

# Installs the pinned sccache release + fallback wrapper when USE_SCCACHE=1.
SCCACHE_INSTALL_RUN = (
    f"RUN {AWS_SECRET_MOUNT} \\\n"
    '    if [ "$USE_SCCACHE" = "1" ]; then \\\n'
    '        echo "Installing sccache..." \\\n'
    '        && _sccache_arch="${SCCACHE_ARCH:-}" \\\n'
    '        && if [ -z "$_sccache_arch" ]; then \\\n'
    '            case "${TARGETPLATFORM:-linux/${TARGETARCH:-amd64}}" in \\\n'
    '              linux/arm64) _sccache_arch="aarch64" ;; \\\n'
    '              linux/amd64) _sccache_arch="x86_64" ;; \\\n'
    '              *) echo "Unsupported platform for sccache: ${TARGETPLATFORM:-linux/${TARGETARCH:-amd64}}" >&2; exit 1 ;; \\\n'
    '            esac; \\\n'
    '        fi \\\n'
    '        && curl -fsSL -o /tmp/sccache.tar.gz \\\n'
    f'            "https://github.com/mozilla/sccache/releases/download/v{SCCACHE_VERSION}/sccache-v{SCCACHE_VERSION}-'
    '${_sccache_arch}-unknown-linux-musl.tar.gz" \\\n'
    '        && tar -xzf /tmp/sccache.tar.gz -C /tmp \\\n'
    f'        && install -m 0755 "/tmp/sccache-v{SCCACHE_VERSION}-'
    '${_sccache_arch}-unknown-linux-musl/sccache" '
    f'{SCCACHE_BIN} \\\n'
    f'        && rm -rf /tmp/sccache.tar.gz "/tmp/sccache-v{SCCACHE_VERSION}-'
    '${_sccache_arch}-unknown-linux-musl" \\\n'
    f'        && {SCCACHE_BIN} --version \\\n'
    f'        && {_WRAPPER_INSTALL}; \\\n'
    '    fi'
)


def inject_after_stage_header(text: str, stage_marker: str, block: str) -> str:
    idx = text.find(stage_marker)
    if idx < 0:
        return text
    line_end = text.find("\n", idx)
    if line_end < 0:
        return text
    window_end = min(len(text), line_end + len(block) + 500)
    if block in text[line_end + 1 : window_end]:
        return text
    return text[: line_end + 1] + block + "\n\n" + text[line_end + 1 :]


def inject_before_run(text: str, run_needle: str, block: str) -> str:
    idx = text.find(run_needle)
    if idx < 0:
        return text
    window_start = max(0, idx - len(block) - 500)
    if block in text[window_start:idx]:
        return text
    return text[:idx] + block + "\n\n" + text[idx:]


def bump_sccache_version(text: str) -> str:
    """Rewrite upstream v0.8.1 sccache URLs/paths to our pinned release."""
    import re

    version = SCCACHE_VERSION
    text = re.sub(
        r"/download/v0\.8\.1/sccache-v0\.8\.1-",
        f"/download/v{version}/sccache-v{version}-",
        text,
    )
    text = re.sub(
        r"sccache-v0\.8\.1-",
        f"sccache-v{version}-",
        text,
    )
    return text
