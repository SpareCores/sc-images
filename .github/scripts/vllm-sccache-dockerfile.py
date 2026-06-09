"""Shared Dockerfile fragments for vLLM sccache (used by tune-*.sh)."""

SCCACHE_VERSION = "0.15.0"
SCCACHE_BIN = "/usr/bin/sccache"
# cc-rs auto-wraps CC when RUSTC_WRAPPER file stem is "sccache", breaking openssl-sys
# make. Use a wrapper script (not a symlink — symlinking sccache hangs rustc -vV).
SCCACHE_RUST_WRAPPER = "/usr/local/bin/sc-rust-wrap"
SCCACHE_WRAPPER = "/usr/local/bin/sccache-wrapper"
SCCACHE_ERROR_LOG = "/tmp/sccache.log"

SCCACHE_COMMON_ARGS = """
ARG USE_SCCACHE
ARG SCCACHE_VERBOSE=0
ARG SCCACHE_BUCKET_NAME
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_KEY_PREFIX
ARG SCCACHE_S3_NO_CREDENTIALS=0""".strip()

# Prod (SCCACHE_VERBOSE=0): no server log file, no stderr trace. Verbose mode enables
# SCCACHE_LOG / SCCACHE_ERROR_LOG via SCCACHE_SETUP_LOGGING.
SCCACHE_COMMON_ENV = """
ENV SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME}
ENV SCCACHE_REGION=${SCCACHE_REGION_NAME}
ENV SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX}
ENV SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS}
ENV SCCACHE_VERBOSE=${SCCACHE_VERBOSE}
ENV SCCACHE_IDLE_TIMEOUT=0
ENV SCCACHE_IGNORE_SERVER_IO_ERROR=1""".strip()

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

# Per-compile trace lines only when SCCACHE_VERBOSE=1; infra fallback always logged.
_WRAPPER_INSTALL = (
    f'printf \'#!/bin/sh\\n'
    f'if [ "${{SCCACHE_VERBOSE:-0}}" = "1" ]; then\\n'
    f'  _src=""\\n'
    f'  for _a in "$@"; do\\n'
    f'    case "$_a" in\\n'
    f'      *.cu|*.cpp|*.cc|*.c|*.C) _src="$_a" ;;\\n'
    f'    esac\\n'
    f'  done\\n'
    f'  echo "[sccache-trace] compile launcher=$1 src=${{_src:-?}} args=$(echo "$@" | head -c 240)" >&2\\n'
    f'fi\\n'
    f'err=/tmp/sccache-err.$$\\n'
    f'/usr/bin/sccache "$@" 2>"$err"\\n'
    f'rc=$?\\n'
    f'if [ $rc -eq 0 ]; then rm -f "$err"; exit 0; fi\\n'
    f'if grep -qE "failed to execute compile|No such file or directory|Failed to send data|Failed to read response|Connection refused|Broken pipe|resource unavailable" "$err" 2>/dev/null; then\\n'
    f'  echo "[sccache-wrapper] infrastructure failure rc=$rc, falling back: $1" >&2\\n'
    f'  if [ "${{SCCACHE_VERBOSE:-0}}" = "1" ]; then tail -20 "$err" | sed "s/^/[sccache-trace] /" >&2; fi\\n'
    f'  rm -f "$err"\\n'
    f'  exec "$@"\\n'
    f'fi\\n'
    f'if [ "${{SCCACHE_VERBOSE:-0}}" = "1" ] && [ -s "$err" ]; then tail -30 "$err" | sed "s/^/[sccache-trace] client: /" >&2; fi\\n'
    f'rm -f "$err"\\n'
    f'exit $rc\\n\' > {SCCACHE_WRAPPER} && chmod +x {SCCACHE_WRAPPER}'
)

# setup.py hardcodes CMAKE_*_COMPILER_LAUNCHER=sccache; point at our wrapper.
SCCACHE_PATCH_SETUP_PY = (
    r"""sed -i \
        -e 's|-DCMAKE_C_COMPILER_LAUNCHER=sccache|-DCMAKE_C_COMPILER_LAUNCHER=/usr/local/bin/sccache-wrapper|g' \
        -e 's|-DCMAKE_CXX_COMPILER_LAUNCHER=sccache|-DCMAKE_CXX_COMPILER_LAUNCHER=/usr/local/bin/sccache-wrapper|g' \
        -e 's|-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache|-DCMAKE_CUDA_COMPILER_LAUNCHER=/usr/local/bin/sccache-wrapper|g' \
        -e 's|-DCMAKE_HIP_COMPILER_LAUNCHER=sccache|-DCMAKE_HIP_COMPILER_LAUNCHER=/usr/local/bin/sccache-wrapper|g' \
        setup.py"""
)

# Enable trace log + log file (local debug only). Prod leaves logging disabled.
SCCACHE_SETUP_LOGGING = (
    r"""if [ "${SCCACHE_VERBOSE:-0}" = "1" ]; then \
        export SCCACHE_IGNORE_SERVER_IO_ERROR=0 \
        && export SCCACHE_LOG=sccache=trace \
        && export SCCACHE_LOG_MILLIS=1 \
        && export SCCACHE_ERROR_LOG="${SCCACHE_ERROR_LOG:-"""
    + SCCACHE_ERROR_LOG
    + r"""}" \
        && echo "[sccache-debug] SCCACHE_VERBOSE=1 (trace log, S3 IO errors surfaced)"; \
    else \
        unset SCCACHE_LOG SCCACHE_ERROR_LOG SCCACHE_LOG_MILLIS 2>/dev/null || true \
        && export SCCACHE_IGNORE_SERVER_IO_ERROR=1; \
    fi"""
)

# Live tail of server log onto build stdout (verbose only).
SCCACHE_LOG_TAILER_START = (
    r"""_sclog="${SCCACHE_ERROR_LOG:-"""
    + SCCACHE_ERROR_LOG
    + r"""}" \
        && _SCCACHE_TAIL_PID="" \
        && _sccache_start_tailer() { \
            ( while [ ! -f "$_sclog" ]; do sleep 0.2; done; \
              tail -n0 -F "$_sclog" 2>/dev/null \
              | grep --line-buffered -E 'Hash key:|Cache hit in|Cache miss in|Compiling locally|Cache read error|not cacheable|Non-cacheable|Compiler output|Could not|Stored in cache|S3Cache|lookup|execute compiler' \
              | sed 's/^/[sccache-trace] /' \
            ) >&2 & \
            _SCCACHE_TAIL_PID=$!; \
        } \
        && _sccache_stop_tailer() { kill ${_SCCACHE_TAIL_PID:-} 2>/dev/null || true; wait ${_SCCACHE_TAIL_PID:-} 2>/dev/null || true; }"""
)

# Verbose-only summary (also run from EXIT trap when the wheel build is stopped early).
SCCACHE_EMIT_DEBUG_SUMMARY = (
    r"""_sccache_emit_debug_summary() { \
        _label="${1:-final}" \
        && _sclog="${SCCACHE_ERROR_LOG:-"""
    + SCCACHE_ERROR_LOG
    + r"""}" \
        && echo "[sccache-debug] === summary (${_label}) ===" \
        && echo "[sccache-debug] compile env: MAX_JOBS=${MAX_JOBS:-unset} NVCC_THREADS=${NVCC_THREADS:-unset} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-unset} CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-unset}" \
        && echo "[sccache-debug] s3: bucket=${SCCACHE_BUCKET:-unset} prefix=${SCCACHE_S3_KEY_PREFIX:-unset} region=${SCCACHE_REGION:-unset} ignore_io=${SCCACHE_IGNORE_SERVER_IO_ERROR:-unset}" \
        && echo "[sccache-debug] toolchain: sccache=$(/usr/bin/sccache --version 2>/dev/null | head -1) nvcc=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1) gcc=$(${CC:-gcc} -dumpfullversion 2>/dev/null || ${CC:-gcc} -dumpversion 2>/dev/null || echo ?)" \
        && _ccache=$(find /workspace/build -name CMakeCache.txt 2>/dev/null | head -1) \
        && if [ -n "$_ccache" ]; then \
            echo "[sccache-debug] CMakeCache launchers/threads/arch:" \
            && grep -E 'CMAKE_.*COMPILER_LAUNCHER|NVCC_THREADS|CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER:' "$_ccache" 2>/dev/null | sed 's/^/[sccache-debug]   /' || true; \
        fi \
        && _bninja=$(find /workspace/build -name build.ninja 2>/dev/null | head -1) \
        && if [ -n "$_bninja" ]; then \
            echo "[sccache-debug] ninja cache_kernels rule:" \
            && grep -E 'build.*cache_kernels\.cu' "$_bninja" 2>/dev/null | head -1 | sed 's/^/[sccache-debug]   /' || true \
            && echo "[sccache-debug] ninja cache_kernels command (first 400 chars):" \
            && _rule=$(grep -E 'build.*cache_kernels\.cu' "$_bninja" 2>/dev/null | head -1 | awk '{print $1}' | sed 's/:$//') \
            && if [ -n "$_rule" ]; then \
                grep -A3 "^${_rule}:" "$_bninja" 2>/dev/null | head -4 | sed 's/^/[sccache-debug]   /' | head -c 500 || true; \
                echo ""; \
            fi; \
        fi \
        && if [ -f "$_sclog" ]; then \
            echo "[sccache-debug] log: $_sclog ($(wc -l < "$_sclog" | tr -d ' ') lines)" \
            && echo "[sccache-debug] counts: hits=$(grep -c 'Cache hit in' "$_sclog" 2>/dev/null || echo 0) misses=$(grep -c 'Cache miss in' "$_sclog" 2>/dev/null || echo 0) local=$(grep -c 'Compiling locally' "$_sclog" 2>/dev/null || echo 0) read_errors=$(grep -c 'Cache read error' "$_sclog" 2>/dev/null || echo 0) non_cacheable=$(grep -cE 'not cacheable|Non-cacheable' "$_sclog" 2>/dev/null || echo 0) hash_keys=$(grep -c 'Hash key:' "$_sclog" 2>/dev/null || echo 0)" \
            && echo "[sccache-debug] cache_kernels.cu.o log lines:" \
            && grep -F 'cache_kernels.cu' "$_sclog" | tail -10 | sed 's/^/[sccache-debug]   /' || true \
            && echo "[sccache-debug] distinct hash keys (last 20):" \
            && grep 'Hash key:' "$_sclog" | tail -20 | sed 's/^/[sccache-debug]   /' || true \
            && echo "[sccache-debug] last 80 hash/hit/miss/local/error lines:" \
            && grep -E 'Hash key:|Cache hit in|Cache miss in|Compiling locally|Cache read error|not cacheable|Non-cacheable|Stored in cache' "$_sclog" | tail -80 | sed 's/^/[sccache-debug]   /' || true; \
        else \
            echo "[sccache-debug] no log at $_sclog"; \
        fi \
        && sccache --show-stats 2>/dev/null | sed 's/^/[sccache-debug]   /' || true; \
    }"""
)

SCCACHE_VERBOSE_WHEEL_SETUP = (
    r"""if [ "${SCCACHE_VERBOSE:-0}" = "1" ]; then \
        echo "[sccache-debug] pre-wheel: MAX_JOBS=${MAX_JOBS:-unset} NVCC_THREADS=${NVCC_THREADS:-unset} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-unset}" \
        && echo "[sccache-debug] CXX=${CXX:-/usr/bin/g++} CMAKE_CXX_COMPILER_LAUNCHER=$CMAKE_CXX_COMPILER_LAUNCHER" \
        && echo "[sccache-debug] sccache binary:" && ls -la /usr/bin/sccache \
        && """
    + SCCACHE_LOG_TAILER_START
    + r""" \
        && """
    + SCCACHE_EMIT_DEBUG_SUMMARY
    + r""" \
        && _SCCACHE_DEBUG_DONE=0 \
        && _sccache_checkpoint() { \
            [ "${_SCCACHE_DEBUG_DONE:-0}" = "1" ] && return 0; \
            _sccache_stop_tailer; \
            _sccache_emit_debug_summary "checkpoint (EXIT/interrupted)"; \
        } \
        && trap _sccache_checkpoint EXIT \
        && _sccache_start_tailer \
        && echo "[sccache-debug] live trace: tailing $_sclog -> [sccache-trace] lines on stderr"; \
    else \
        _sccache_stop_tailer() { :; } \
        && _sccache_emit_debug_summary() { :; }; \
    fi"""
)

SCCACHE_SHOW_STATS = r"""sccache --show-stats || true"""

# Start sccache server (logging env set earlier by SCCACHE_SETUP_LOGGING when verbose).
SCCACHE_SERVER_START = (
    r"""sccache --stop-server 2>/dev/null || true \
        && sccache --start-server 2>/dev/null || true"""
)

# Post-compile: prod = show-stats only; verbose = full debug summary.
SCCACHE_RUST_DEBUG_SUMMARY = (
    r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then \
        if [ "${SCCACHE_VERBOSE:-0}" = "1" ]; then \
            _sclog="${SCCACHE_ERROR_LOG:-"""
    + SCCACHE_ERROR_LOG
    + r"""}" \
            && echo "[sccache-debug] === summary (rust final) ===" \
            && echo "[sccache-debug] compile env: MAX_JOBS=${MAX_JOBS:-unset} CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-unset}" \
            && if [ -f "$_sclog" ]; then \
                echo "[sccache-debug] counts: hits=$(grep -c 'Cache hit in' "$_sclog" 2>/dev/null || echo 0) misses=$(grep -c 'Cache miss in' "$_sclog" 2>/dev/null || echo 0) read_errors=$(grep -c 'Cache read error' "$_sclog" 2>/dev/null || echo 0)" \
                && grep -E 'Hash key:|Cache hit in|Cache miss in|Cache read error' "$_sclog" | tail -40 | sed 's/^/[sccache-debug]   /' || true; \
            fi \
            && sccache --show-stats 2>/dev/null | sed 's/^/[sccache-debug]   /' || true; \
        else \
            """
    + SCCACHE_SHOW_STATS
    + r"""; \
        fi; \
    fi"""
)

SCCACHE_DEBUG_SUMMARY = (
    r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then \
        _SCCACHE_DEBUG_DONE=1 \
        && trap - EXIT 2>/dev/null || true \
        && _sccache_stop_tailer \
        && if [ "${SCCACHE_VERBOSE:-0}" = "1" ]; then \
            _sccache_emit_debug_summary "final"; \
        else \
            """
    + SCCACHE_SHOW_STATS
    + r"""; \
        fi; \
    fi"""
)

SCCACHE_RUST_PREP = (
    r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then \
        export RUSTC_WRAPPER="${RUSTC_WRAPPER:-"""
    + SCCACHE_RUST_WRAPPER
    + r"""}" \
        && export CC="${CC:-/usr/bin/gcc}" \
        && export CXX="${CXX:-/usr/bin/g++}" \
        && export CARGO_INCREMENTAL=0 \
        && """
    + SCCACHE_SETUP_LOGGING
    + r""" \
        && """
    + SCCACHE_SERVER_START
    + r"""; \
    fi &&"""
)

SCCACHE_WHEEL_PREP = (
    r"""if [ "${USE_SCCACHE:-0}" = "1" ]; then \
        unset RUSTC_WRAPPER \
        && """
    + SCCACHE_SETUP_LOGGING
    + r""" \
        && if [ ! -x """
    + SCCACHE_WRAPPER
    + r""" ]; then \
            """
    + _WRAPPER_INSTALL
    + r"""; \
        fi \
        && """
    + SCCACHE_PATCH_SETUP_PY
    + r""" \
        && export CMAKE_C_COMPILER_LAUNCHER="""
    + SCCACHE_WRAPPER
    + r""" \
        && export CMAKE_CXX_COMPILER_LAUNCHER="""
    + SCCACHE_WRAPPER
    + r""" \
        && echo | "${CXX:-/usr/bin/g++}" -x c++ -E -P - >/dev/null \
        && """
    + SCCACHE_SERVER_START
    + r""" \
        && """
    + SCCACHE_VERBOSE_WHEEL_SETUP
    + r"""; \
    fi &&"""
)

# Back-compat alias used by CPU tune script.
SCCACHE_WHEEL_STATS = SCCACHE_DEBUG_SUMMARY

# Installs the pinned sccache release + fallback wrapper when USE_SCCACHE=1.
# No AWS secret mount here (curl-only); creds are mounted on compile RUNs only.
SCCACHE_INSTALL_RUN = (
    'RUN if [ "$USE_SCCACHE" = "1" ]; then \\\n'
    '        echo "Installing sccache..." \\\n'
    '        && _sccache_arch="${SCCACHE_ARCH:-}" \\\n'
    '        && if [ -z "$_sccache_arch" ]; then \\\n'
    '            case "${TARGETPLATFORM:-linux/${TARGETARCH:-amd64}}" in \\\n'
    '              linux/arm64) SCCACHE_ARCH="aarch64" ;; \\\n'
    '              linux/amd64) SCCACHE_ARCH="x86_64" ;; \\\n'
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
