FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS builder
RUN apt-get update --error-on=any && apt-get install -y build-essential git cmake python-is-python3 curl
RUN mkdir /tmp/cutlass && \
    curl -sSLfo - https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.5.1.tar.gz | tar zxf - -C /tmp/cutlass --strip-components=1
# Common AWS NVIDIA GPUs and Their CUDA Architectures for setting up CUTLASS_NVCC_ARCHS below
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
RUN mkdir /tmp/cutlass/build && cd /tmp/cutlass/build && \
    # 37 and 53 are not supported by the current version
    cmake .. -DCUTLASS_NVCC_ARCHS="60;61;70;75;80;86" && \
    # installs to build/install
    make install -j $(nproc)

FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

COPY --from=builder /tmp/cutlass/build/install /usr/local
CMD ["/usr/local/bin/cutlass_profiler"]
