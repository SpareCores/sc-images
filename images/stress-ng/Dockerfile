# V0.20.01
FROM ghcr.io/colinianking/stress-ng:00559dcb1bae3ef12295a1420683094a7d1389e0
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY count.awk /usr/local/bin
COPY run_stressng_benchmarks.py /usr/local/bin
