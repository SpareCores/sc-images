FROM ubuntu:24.04 as base
ENV DEBIAN_FRONTEND=noninteractive

FROM base as build
ARG MEMBENCH_VERSION=1.2.1
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y build-essential git libhwloc-dev libnuma-dev && \
    mkdir -p /usr/src && \
    cd /usr/src && \
    git clone https://github.com/SpareCores/sc-membench.git && \
    cd sc-membench && \
    git checkout ${MEMBENCH_VERSION} && \
    make full && \
    cp membench-full /usr/local/bin/membench && \
    chmod +x /usr/local/bin/membench

FROM base as final
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt_final,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt_final,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y libhwloc15 libnuma1 libhugetlbfs-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=${PATH}:/usr/local/bin
COPY --from=build /usr/local/bin/membench /usr/local/bin/membench

ENTRYPOINT ["nice", "-n", "-20", "/usr/local/bin/membench"]

