# hammerdb-postgres

amd64 HammerDB client image for PostgreSQL benchmarks, aligned with the [official HammerDB postgres Docker image](https://github.com/TPC-Council/HammerDB/blob/master/Docker/postgres/Dockerfile) (`tpcorg/hammerdb:postgres`).

| Arch | HammerDB install |
|------|------------------|
| amd64 | Pre-built release `HammerDB-${VERSION}-Prod-Lin-UBU24.tar.gz` (same binary as official) |

Published as `ghcr.io/sparecores/hammerdb-postgres:main` (amd64 only).

Version pin: [`HAMMERDB_VERSION`](HAMMERDB_VERSION).

## Usage

```bash
docker run --rm -it ghcr.io/sparecores/hammerdb-postgres:main bash
# inside container:
./hammerdbcli
```

For remote PostgreSQL, use host networking or ensure the container can reach the DB host:

```bash
docker run --rm -it --network=host ghcr.io/sparecores/hammerdb-postgres:main bash
```

## Local build

```bash
# from sc-images repo root
docker buildx build \
  --file images/hammerdb-postgres/Dockerfile \
  --platform linux/amd64 \
  --build-arg HAMMERDB_VERSION=6.0 \
  --tag hammerdb-postgres:local \
  images/hammerdb-postgres
```
