# PostgreSQL Server

This image runs a PostgreSQL 18 server for the
[PostgreSQL (pgbench) benchmark](../benchmark-pgbench-postgres/README.md) to
connect to in remote mode. It is the upstream
[`postgres:18`](https://hub.docker.com/_/postgres) image with `resource-tracker`
added, and it starts the server with `nice -n -20` and default settings. The
image measures nothing itself.

This image reports no results: its container log holds only the PostgreSQL
server log. The
[PostgreSQL (pgbench) manual](../benchmark-pgbench-postgres/README.md)
describes the benchmark, which can run `pgbench` against this server in
remote mode, and how to read its results. This page covers only how to run
the server.

## Usage

Start the server in the background via Docker:

```bash
docker run -d --rm --name postgres-server --privileged -p 5432:5432 \
  -e POSTGRES_PASSWORD=<password> \
  ghcr.io/sparecores/benchmark-postgres-server:main
```

Replace `<password>` with a password of your choice. The upstream
`postgres:18` entrypoint requires `POSTGRES_PASSWORD`. The
`--privileged` option is optional: the image needs it only so that
`nice -n -20` can raise the server's priority, which takes only the
`CAP_SYS_NICE` capability, while `--privileged` grants the container far more.
Without it, the server runs at normal priority.

Once the server accepts connections, run the benchmark against it in
[remote mode](../benchmark-pgbench-postgres/docs/usage.md#remote-mode), from the
same machine or another one, and save its JSON result. The benchmark does not
wait for the server, so check the server log with `docker logs postgres-server`
first:

```bash
docker run --rm \
  -e SC_DB_HOST=<server-host> \
  -e SC_DB_PASSWORD=<password> \
  -e SC_DB_VCPUS=<server-vcpus> \
  ghcr.io/sparecores/benchmark-pgbench-postgres:main > result.json
```

Fill in the following placeholders:

- `<server-host>`: the hostname or IP address of the machine that runs the
  server container. Port 5432 of that machine must be reachable from the
  benchmark machine. On the same machine, use an IP address of the host, such
  as its private IP address, not `localhost`, which points to the benchmark
  container itself.
- `<password>`: the `POSTGRES_PASSWORD` value of the server container.
- `<server-vcpus>`: the vCPU count of the server machine, as `nproc` prints it
  on that machine. The benchmark derives its concurrency points from it.

`SC_DB_PORT` and `SC_DB_USER` keep their defaults, `5432` and `postgres`. If
you set `POSTGRES_USER` on the server, set `SC_DB_USER` to the same value. When
both containers run on one machine, add `-e SC_TOPOLOGY=single_vm` to the
benchmark command; otherwise, the benchmark records `topology` as `multi_vm` in
its output.

Stop the server with `docker stop postgres-server`. Because of `--rm`, this
also removes the container and its database.

### Settings

This image sets no PostgreSQL settings. The only environment variable it adds
is `TRACKER_QUIET` (default: `true`), which silences `resource-tracker`. The
upstream `postgres:18` entrypoint reads the other variables, such as
`POSTGRES_PASSWORD`, `POSTGRES_USER`, and `PGDATA`; see the
[`postgres` image docs](https://hub.docker.com/_/postgres) on Docker Hub.

To change a server setting, append `postgres` and one `-c <name>=<value>` pair
per setting after the image name.

The benchmark drops the concurrency points above the server's
`max_connections` minus 50. Its highest default point is twice the vCPU
count, so set `max_connections` to at least twice the vCPU count plus 50 to
keep all points.

PostgreSQL's default `max_connections` of 100 leaves room for 50 clients, so
the benchmark drops every point above 50: the highest point on servers with
more than 25 vCPUs, and more points on servers with more than 50.

For example, the following command raises `max_connections` to 242 (twice 96,
plus 50) for a server with 96 vCPUs:

```bash
docker run -d --rm --name postgres-server --privileged -p 5432:5432 \
  -e POSTGRES_PASSWORD=<password> \
  ghcr.io/sparecores/benchmark-postgres-server:main \
  postgres -c max_connections=242
```

### Output

The container log holds the PostgreSQL server log. Read it with
`docker logs postgres-server`.

The `resource-tracker` tool samples the server's resource usage, such as CPU,
memory, disk, and network, once per second. In production runs, the Spare
Cores team
[streams these samples](https://sparecores.com/article/resource-tracker-rs#streaming-to-sentinel)
to the [Spare Cores Sentinel](https://sentinel.sparecores.com/) platform for
the team's own monitoring. This resource usage is not published.

## Differences from the Upstream Image

The image differs from the upstream `postgres:18` image in the following ways:

- **Process Tree:** The [`Dockerfile`](./Dockerfile) starts
  `resource-tracker -- nice -n -20 docker-entrypoint.sh postgres`. The upstream
  entrypoint and server run unchanged, under `resource-tracker`.
- **No Tuning:** The image adds no configuration files, initialization
  scripts, or settings, so the server runs with PostgreSQL defaults. In
  [standalone mode](../benchmark-pgbench-postgres/docs/usage.md#standalone-mode),
  the benchmark instead tunes its local server to the host's memory and
  vCPUs.
- **Minor Version:** The image builds on `postgres:18`, so only the major
  version is fixed. Minor versions vary between builds, as described in
  [Minor Engine Versions](../benchmark-pgbench-postgres/docs/limitations.md#minor-engine-versions).

## References

The terms on this page are defined in the following places:

- The [references](../benchmark-pgbench-postgres/docs/references.md) of the
  [PostgreSQL (pgbench) manual](../benchmark-pgbench-postgres/README.md)
  define the terms of the benchmark, such as concurrency points.
- The [glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs
  defines common terms, such as `nice` and `resource-tracker`.
