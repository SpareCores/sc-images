# V0.17.08 — pinned stress-ng for the 24h longrun benchmark
ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

ARG STRESSNG_TAG=b7c7a5877501679a3b0a67d877e6274a801d1e4e
FROM ghcr.io/colinianking/stress-ng:${STRESSNG_TAG}
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "sh"]
