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

COPY --from=builder /tmp/nvbandwidth/nvbandwidth /usr/local/bin/
CMD ["/usr/local/bin/nvbandwidth"]
