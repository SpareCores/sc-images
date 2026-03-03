# V0.20.01
FROM ghcr.io/colinianking/stress-ng:a459e41786552858e025af920d8323ee74ed5e46
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY count.awk /usr/local/bin
COPY run_stressng_benchmarks.py /usr/local/bin
