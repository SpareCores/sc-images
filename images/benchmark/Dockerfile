FROM ubuntu:24.04 as base
ENV TURBOBENCH_VER=d68171773d7576c4323bb7966d01b7d4d2491b65

FROM base as build
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y build-essential curl git unzip && \
    cd /tmp && \
      git clone --recursive https://github.com/powturbo/TurboBench.git && \
      cd /tmp/TurboBench && \
      git checkout $TURBOBENCH_VER && \
      # disable some parts, we don't use them and don't compile on ARM for various reasons
      make -C /tmp/TurboBench SSERC=0 TURBORC=0 TURBORLE=0 -j4 && \
      mkdir /usr/local/turbobench && \
    git clone https://github.com/MiloszKrajewski/SilesiaCorpus.git /tmp/silesia && \
      cd /tmp/silesia && \
      find . -name "*.zip" | xargs -n 1 unzip && rm -f *.zip *.md && \
  cp /tmp/TurboBench/turbobench /tmp/silesia/* /usr/local/turbobench/

FROM base as final
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt_final,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt_final,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y colorized-logs
ENV PATH=${PATH}:/usr/local/bin
ADD turbobench.ini /usr/local/turbobench/turbobench.ini
COPY --from=build /usr/local/turbobench /usr/local/turbobench

