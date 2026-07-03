# syntax=docker/dockerfile:1

# amd64 HammerDB client for PostgreSQL benchmarks (official pre-built release).

ARG HAMMERDB_VERSION=6.0

FROM ubuntu:24.04

ARG HAMMERDB_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget unzip gnupg apt-utils iputils-ping vim netcat-traditional \
      libxft-dev libcairo2-dev xauth python3 libpq-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
      "https://github.com/TPC-Council/HammerDB/releases/download/v${HAMMERDB_VERSION}/HammerDB-${HAMMERDB_VERSION}-Prod-Lin-UBU24.tar.gz" \
      | tar -xzf - -C /home/ && \
    ln -sfn "/home/HammerDB-${HAMMERDB_VERSION}" /home/hammerdb

WORKDIR /home/hammerdb

CMD ["tail", "-f", "/dev/null"]
