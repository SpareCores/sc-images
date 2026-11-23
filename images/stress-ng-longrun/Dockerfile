# V0.17.08 — pinned stress-ng for the 24h longrun benchmark
# Built on Debian (glibc) from the pinned upstream source: the resource-tracker binary is dynamically linked
# against glibc, so it cannot start on the musl-based Alpine image of ghcr.io/colinianking/stress-ng.
ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG STRESSNG_TAG=V0.17.08
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

FROM debian:bookworm-slim AS build
ARG STRESSNG_TAG
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl libaio-dev libapparmor-dev libattr1-dev libbsd-dev libcap-dev \
      libgcrypt20-dev libjudy-dev libkeyutils-dev libkmod-dev libsctp-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL "https://github.com/ColinIanKing/stress-ng/archive/refs/tags/${STRESSNG_TAG}.tar.gz" \
      | tar -xz --strip-components=1 \
    && make -j"$(nproc)" && install -m 0755 stress-ng /usr/local/bin/stress-ng

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      libaio1 libapparmor1 libattr1 libbsd0 libcap2 libgcrypt20 libjudydebian1 libkeyutils1 libkmod2 \
      libsctp1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/stress-ng /usr/local/bin/stress-ng
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "sh"]
