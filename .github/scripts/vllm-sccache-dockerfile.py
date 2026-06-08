"""Shared Dockerfile fragments for vLLM sccache (used by tune-*.sh)."""

SCCACHE_VERSION = "0.15.0"
SCCACHE_BIN = "/usr/bin/sccache"
# cc-rs auto-wraps CC when RUSTC_WRAPPER file stem is "sccache", breaking openssl-sys
# make. Use a wrapper script (not a symlink — symlinking sccache hangs rustc -vV).
SCCACHE_RUST_WRAPPER = "/usr/local/bin/sc-rust-wrap"
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

# rust-build: cache rustc. CC/CXX are absolute paths; RUSTC_WRAPPER must NOT have
# file stem "sccache" or cc-rs wraps CC and openssl-sys make fails.
SCCACHE_RUST_ENV_BLOCK = (
    SCCACHE_COMMON_ARGS
    + f"""
ARG RUSTC_WRAPPER={SCCACHE_RUST_WRAPPER}
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
_RUST_WRAPPER_INSTALL = (
    f"printf '#!/bin/sh\\nexec {SCCACHE_BIN} \"$@\"\\n' > {SCCACHE_RUST_WRAPPER} "
    f"&& chmod +x {SCCACHE_RUST_WRAPPER}"
)

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
        export RUSTC_WRAPPER="${RUSTC_WRAPPER:-""" + SCCACHE_RUST_WRAPPER + r"""}" \
        && export CC="${CC:-/usr/bin/gcc}" \
        && export CXX="${CXX:-/usr/bin/g++}" \
        && export CARGO_INCREMENTAL=0 \
        && export SCCACHE_IGNORE_SERVER_IO_ERROR=1 \
        && sccache --start-server 2>/dev/null || true; \
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
# No AWS secret mount here (curl-only); creds are mounted on compile RUNs only.
SCCACHE_INSTALL_RUN = (
    'RUN if [ "$USE_SCCACHE" = "1" ]; then \\\n'
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
    f'        && {_RUST_WRAPPER_INSTALL} \\\n'
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
