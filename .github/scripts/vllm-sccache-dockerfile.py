"""Shared Dockerfile fragments for vLLM sccache (used by tune-*.sh)."""

SCCACHE_ARG_ENV_BLOCK = """
ARG USE_SCCACHE
ARG RUSTC_WRAPPER
ARG SCCACHE_BUCKET_NAME
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_KEY_PREFIX
ARG SCCACHE_S3_NO_CREDENTIALS=0
ENV RUSTC_WRAPPER=${RUSTC_WRAPPER}
ENV SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME}
ENV SCCACHE_REGION=${SCCACHE_REGION_NAME}
ENV SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX}
ENV SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS}
ENV SCCACHE_IDLE_TIMEOUT=0
""".strip()

# csrc-build / vllm-build already declare the SCCACHE_* ARGs upstream or in prior patches.
SCCACHE_COMPILER_ENV_BLOCK = """
ARG RUSTC_WRAPPER
ENV RUSTC_WRAPPER=${RUSTC_WRAPPER}
ENV SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME}
ENV SCCACHE_REGION=${SCCACHE_REGION_NAME}
ENV SCCACHE_S3_KEY_PREFIX=${SCCACHE_S3_KEY_PREFIX}
ENV SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS}
ENV SCCACHE_IDLE_TIMEOUT=0
""".strip()

AWS_SECRET_MOUNT = (
    "--mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false"
)

# Installs the upstream-pinned sccache release when USE_SCCACHE=1. Requires TARGETPLATFORM
# (GPU rust-build / csrc-build) or TARGETARCH (CPU stages).
SCCACHE_INSTALL_RUN = r"""RUN --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && _sccache_arch="${SCCACHE_ARCH:-}" \
        && if [ -z "$_sccache_arch" ]; then \
            case "${TARGETPLATFORM:-linux/${TARGETARCH:-amd64}}" in \
              linux/arm64) _sccache_arch="aarch64" ;; \
              linux/amd64) _sccache_arch="x86_64" ;; \
              *) echo "Unsupported platform for sccache: ${TARGETPLATFORM:-linux/${TARGETARCH:-amd64}}" >&2; exit 1 ;; \
            esac; \
        fi \
        && curl -fsSL -o /tmp/sccache.tar.gz \
            "https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-${_sccache_arch}-unknown-linux-musl.tar.gz" \
        && tar -xzf /tmp/sccache.tar.gz -C /tmp \
        && install -m 0755 "/tmp/sccache-v0.8.1-${_sccache_arch}-unknown-linux-musl/sccache" /usr/local/bin/sccache \
        && rm -rf /tmp/sccache.tar.gz "/tmp/sccache-v0.8.1-${_sccache_arch}-unknown-linux-musl"; \
    fi"""


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
