FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --error-on=any && apt-get install -y build-essential curl
RUN mkdir /tmp/nvbandwidth && \
    curl -sSLfo - https://github.com/NVIDIA/nvbandwidth/archive/refs/tags/v0.5.tar.gz | tar zxf - -C /tmp/nvbandwidth --strip-components=1
# Common AWS NVIDIA GPUs and Their CUDA Architectures
#    Tesla K80 (Kepler architecture)
#        sm_37
#    Tesla M60 (Maxwell architecture)
#        sm_52
#    Tesla P4, P40, and P100 (Pascal architecture)
#        sm_60 (for P4 and P40)
#        sm_61 (for P100)
#    Tesla T4 (Turing architecture)
#        sm_75
#    Tesla V100 (Volta architecture)
#        sm_70
#    Tesla A100 (Ampere architecture)
#        sm_80
#    NVIDIA RTX A6000 or other RTX 30 series (Ampere architecture)
#        sm_86
RUN cd /tmp/nvbandwidth && ./debian_install.sh

FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get install -y cuda-nvml-dev-12-6
RUN echo "/usr/local/cuda-12.6/compat\n/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs" > /etc/ld.so.conf.d/cuda_compat.conf && ldconfig
COPY --from=builder /tmp/nvbandwidth/nvbandwidth /usr/local/bin/
# disable nvidia/cuda entrypoint, which prints a banner
ENTRYPOINT []
CMD ["/usr/local/bin/nvbandwidth"]
