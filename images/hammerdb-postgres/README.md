# hammerdb-postgres

Multi-arch HammerDB client image for PostgreSQL benchmarks, aligned with the [official HammerDB postgres Docker image](https://github.com/TPC-Council/HammerDB/blob/master/Docker/postgres/Dockerfile) (`tpcorg/hammerdb:postgres`).

| Arch | HammerDB install |
|------|------------------|
| amd64 | Pre-built release `HammerDB-${VERSION}-Prod-Lin-UBU24.tar.gz` (same binary as official) |
| arm64 | Built from source with upstream [BAWT](https://www.hammerdb.com/blog/uncategorized/how-to-build-hammerdb-from-source/) using a postgres-only driver set |

Published as `ghcr.io/sparecores/hammerdb-postgres:main` (amd64 + arm64 manifest).

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
# from sc-images repo root (arm64 only compiles from source)
docker buildx build \
  --file images/hammerdb-postgres/Dockerfile \
  --platform linux/arm64 \
  --build-arg HAMMERDB_VERSION=5.0 \
  --build-arg NUM_JOBS=4 \
  --tag hammerdb-postgres:local \
  images/hammerdb-postgres
```

HammerDB sources are cloned inside the Docker build on arm64 (`git clone` in `build-hammerdb-arm64.sh`); no host-side checkout is required. amd64 builds download the official release tarball only.
