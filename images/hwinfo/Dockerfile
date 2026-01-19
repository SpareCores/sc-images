FROM ubuntu:24.04 as base
ENV LIKWID_VERSION 5.3.0

FROM --platform=$BUILDPLATFORM base as build
ARG BUILDPLATFORM
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y build-essential curl && \
    mkdir /tmp/likwid && \
    curl -sSLfo - https://github.com/RRZE-HPC/likwid/archive/refs/tags/v"${LIKWID_VERSION}".tar.gz | tar zxf - -C /tmp/likwid --strip-components=1 && \
    cd /tmp/likwid && \
    [ $(uname -m) = arm64 -o $(uname -m) = aarch64 ] && sed -i "s/COMPILER = GCC#/COMPILER = GCCARMv8#/" config.mk; \
    make && make install


FROM base as final
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt_final,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt_final,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y lshw jq hwloc-nox
ENV PATH=${PATH}:/usr/local/bin
COPY --from=build /usr/local /usr/local

