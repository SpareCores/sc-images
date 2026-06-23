# syntax=docker/dockerfile:1

# Multi-arch HammerDB client for PostgreSQL benchmarks.
# amd64: official pre-built release (same as tpcorg/hammerdb:postgres base).
# arm64: HammerDB built from source via upstream BAWT (postgres driver only).

ARG HAMMERDB_VERSION=5.0

FROM ubuntu:24.04

ARG TARGETARCH
ARG HAMMERDB_VERSION
ARG NUM_JOBS

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get -y upgrade && \
  if [ "$TARGETARCH" = "arm64" ]; then \
    apt-get install -y --no-install-recommends \
      build-essential git tcl p7zip-full cmake pkg-config autoconf automake libtool \
      libxft-dev libcairo2-dev libpq-dev python3 python3-dev \
      libssl-dev zlib1g-dev libfontconfig1-dev libx11-dev libxrender-dev libxext-dev \
      ca-certificates curl wget unzip gnupg apt-utils iputils-ping vim netcat-traditional xauth; \
  else \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget unzip gnupg apt-utils iputils-ping vim netcat-traditional \
      libxft-dev libcairo2-dev xauth python3; \
  fi && \
  rm -rf /var/lib/apt/lists/*

# Match official postgres image: HammerDB prod tree + libpq client headers.
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "amd64" ]; then \
      curl -fsSL \
        "https://github.com/TPC-Council/HammerDB/releases/download/v${HAMMERDB_VERSION}/HammerDB-${HAMMERDB_VERSION}-Prod-Lin-UBU24.tar.gz" \
        | tar -xzf - -C /home/ && \
      ln -sfn "/home/HammerDB-${HAMMERDB_VERSION}" /home/hammerdb; \
    fi

COPY build-hammerdb-arm64.sh HammerDB-Postgres-Linux.bawt /usr/local/lib/hammerdb-build/

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      export HAMMERDB_VERSION="${HAMMERDB_VERSION}" && \
      export BAWT_SETUP=/usr/local/lib/hammerdb-build/HammerDB-Postgres-Linux.bawt && \
      export NUM_JOBS="${NUM_JOBS:-$(nproc)}" && \
      bash /usr/local/lib/hammerdb-build/build-hammerdb-arm64.sh; \
    fi

WORKDIR /home/hammerdb

CMD ["tail", "-f", "/dev/null"]
