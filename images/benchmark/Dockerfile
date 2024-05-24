FROM ubuntu:24.04 as base

FROM base as build_base
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y build-essential curl git unzip
FROM build_base as turbobench
ENV TURBOBENCH_VER=d68171773d7576c4323bb7966d01b7d4d2491b65
RUN --mount=type=tmpfs,target=/tmp,rw \
    cd /tmp && \
      git clone --recursive https://github.com/powturbo/TurboBench.git && \
      cd /tmp/TurboBench && \
      git checkout ${TURBOBENCH_VER} && \
      # disable some parts, we don't use them and don't compile on ARM for various reasons
      make -C /tmp/TurboBench SSERC=0 TURBORC=0 TURBORLE=0 -j $(nproc) && \
      mkdir /usr/local/turbobench && \
    git clone https://github.com/MiloszKrajewski/SilesiaCorpus.git /tmp/silesia && \
      cd /tmp/silesia && \
      find . -name "*.zip" | xargs -n 1 unzip && rm -f *.zip *.md && \
    cp /tmp/TurboBench/turbobench /tmp/silesia/* /usr/local/turbobench/
FROM build_base as openssl
ENV OPENSSL_VER=3.3.0
RUN --mount=type=tmpfs,target=/tmp,rw \
    mkdir /tmp/openssl && \
      curl -sSLfo - https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VER}/openssl-${OPENSSL_VER}.tar.gz | tar zxf - -C /tmp/openssl --strip-components=1 && \
      cd /tmp/openssl && ./Configure && make -j $(nproc) && make install_sw
FROM build_base as lmbench
RUN apt-get update --error-on=any && apt-get install -y lmbench && \
    # different path by CPU architecture
    cp $(dpkg -L lmbench | grep '\/bw_mem$') /


FROM base as final
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt_final,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt_final,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y colorized-logs libgomp1
ENV PATH=${PATH}:/usr/local/bin
ADD turbobench.ini /usr/local/turbobench/turbobench.ini
ADD openssl.sh /usr/local/bin/openssl.sh
ADD bw_mem.sh /usr/local/bin/bw_mem.sh
COPY --from=turbobench /usr/local/turbobench /usr/local/turbobench
COPY --from=openssl /usr/local /usr/local
COPY --from=lmbench /bw_mem /usr/local/bin/bw_mem
RUN echo -e "/usr/local/lib\n/usr/local/lib64" > /etc/ld.so.conf.d/local.conf && ldconfig -v
