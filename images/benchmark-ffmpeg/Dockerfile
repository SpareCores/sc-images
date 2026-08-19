ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

FROM ubuntu:24.04 AS ffmpeg_builder
ARG TARGETARCH
ARG FFMPEG_VERSION=9.0.1
ARG NV_CODEC_HEADERS_TAG=n13.0.19.1
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      libvorbis-dev \
      libx264-dev \
      libx265-dev \
      nasm \
      pkg-config \
      yasm \
    && rm -rf /var/lib/apt/lists/*
COPY build-ffmpeg.sh /build-ffmpeg.sh
RUN chmod +x /build-ffmpeg.sh && \
    FFMPEG_VERSION="${FFMPEG_VERSION}" \
    NV_CODEC_HEADERS_TAG="${NV_CODEC_HEADERS_TAG}" \
    TARGETARCH="${TARGETARCH}" \
    /build-ffmpeg.sh

FROM ubuntu:24.04
ARG AUDIO_FIXTURE_URL="https://commons.wikimedia.org/wiki/Special:Redirect/file/Entre-dos-aguas-laucke-version-sample.flac"
ARG AUDIO_FIXTURE_SHA256="4445399abe62c9d7c546711a853fccfab8ab274226d2e80aa0e5ad948589e516"
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libvorbis0a \
      libvorbisenc2 \
      libx264-164 \
      libx265-199 \
      numactl \
      python3 \
      python3-psutil \
    && install -d /opt/benchmark-ffmpeg \
    && curl --fail --location --retry 3 --output /opt/benchmark-ffmpeg/source.flac "${AUDIO_FIXTURE_URL}" \
    && echo "${AUDIO_FIXTURE_SHA256}  /opt/benchmark-ffmpeg/source.flac" | sha256sum --check --strict \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ffmpeg_builder /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/local/bin/
COPY --from=ffmpeg_builder /usr/local/lib /usr/local/lib
RUN ldconfig
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py /benchmark.py
COPY SOURCE.md /opt/benchmark-ffmpeg/SOURCE.md

ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark.py"]
