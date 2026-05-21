# V0.20.01
FROM ghcr.io/colinianking/stress-ng:00559dcb1bae3ef12295a1420683094a7d1389e0
ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY --from=${RESOURCE_TRACKER_IMAGE} /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY count.awk /usr/local/bin
COPY run_stressng_benchmarks.py /usr/local/bin
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "sh"]
