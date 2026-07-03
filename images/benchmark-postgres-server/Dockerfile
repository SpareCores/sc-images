ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

FROM postgres:18

COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "nice", "-n", "-20", "docker-entrypoint.sh"]
CMD ["postgres"]
