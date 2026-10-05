ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

FROM ubuntu:24.04 AS fio_builder
ARG FIO_VERSION=3.43
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      libaio-dev \
      liburing-dev \
      pkg-config \
      zlib1g-dev
RUN mkdir /src && \
    curl -fsSL "https://github.com/axboe/fio/archive/refs/tags/fio-${FIO_VERSION}.tar.gz" | \
      tar xzf - -C /src --strip-components=1 && \
    cd /src && \
    ./configure --disable-native --prefix=/usr/local && \
    make -j"$(nproc)" && \
    install -m 0755 fio /usr/local/bin/fio && \
    /usr/local/bin/fio --version

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      libaio1t64 \
      liburing2 \
      nvme-cli \
      python3 \
      smartmontools \
      util-linux \
      zlib1g
COPY --from=fio_builder /usr/local/bin/fio /usr/local/bin/fio
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py /usr/local/bin/benchmark.py
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/usr/local/bin/benchmark.py"]
