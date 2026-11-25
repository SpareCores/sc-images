# AGENTS.md

Repository-level rules for AI coding agents working on `sc-images`: the container
images that collect hardware details and benchmark cloud servers for the Spare Cores
Navigator. [sc-inspector](https://github.com/SpareCores/sc-inspector) runs them on cloud
servers, and the [docs site](https://docs.sparecores.com/images) publishes their docs.

## Repository Layout

- Each image is a folder under `images/`. CI builds every folder that has a `Dockerfile`
  or `prepare.sh`, and publishes it as `ghcr.io/sparecores/<folder>:main`, plus the
  per-arch tags `main-amd64` and `main-arm64` that dependent images build from (e.g.
  `resource-tracker:main-<arch>`); see [CI and Build Internals](docs/ci.md). There are
  no versioned tags.
- Shared manuals that cover a family of images, and are not images themselves, live in
  top-level `<family>-common/` folders (e.g. `vllm-common/`).
- `docs/` holds the pages for maintainers: [Adding an Image](docs/contributors.md) and
  [CI and Build Internals](docs/ci.md). Read them before adding an image or changing CI.

## Image Names

- The folder name is the published image name, and sc-inspector refers to images by it.
  Never rename or remove an image folder unless asked; doing so breaks the Inspector's
  tasks and is a breaking change.
- Use lowercase, hyphen-separated names.
- Benchmark images use the `benchmark-` prefix, never a suffix: `benchmark-storage`, not
  `storage-benchmark`. Name them after the workload or tool, and add the target after it
  when the tool can benchmark several (`benchmark-pgbench-postgres`).
- Variants append qualifiers to the base name, from general to specific:
  `benchmark-vllm-cpu`, `benchmark-vllm-gpu`, `benchmark-vllm-cpu-avx2`,
  `stress-ng-longrun`.
- A service that a benchmark needs is named after the benchmark it serves
  (`benchmark-postgres-server`), and a base image after what it builds
  (`vllm-cpu-base-avx2`).
- `benchmark`, `stress-ng`, `stress-ng-longrun` and `nvbandwidth` predate the prefix
  rule; keep their names.

## Image Folders

- Every image needs a `Dockerfile` (or `prepare.sh`), a one-sentence `DESCRIPTION`, a
  `CATEGORY` (`benchmark`, `hardware-discovery`, `service`, `base` or `internal`) and a
  `README.md`. The other metadata files (`DEPENDS_ON`, `PLATFORMS`, `BUILD_ARGS`, etc.)
  are optional; see [Adding an Image](docs/contributors.md).
- Benchmark images depend on `resource-tracker` (`DEPENDS_ON`) and run their workload
  under it.
- Keep the Images tables of the root `README.md` in sync with the folders: every image
  is listed under its `CATEGORY` with its `DESCRIPTION`. CI
  (`.github/scripts/check-readme-images.py`) fails otherwise.
- Image docs: `README.md` is the image's main page; further pages go in its `docs/`
  folder, and its own glossary terms in `docs/references.md`. Follow the acronym and
  glossary rules in [Adding an Image](docs/contributors.md).
- An image's own `CHANGELOG.md` records its design history and experiments (published
  as its "Design history" page); it is not the repo changelog.

## Running Images

- Never run `benchmark-storage` outside a disposable server: it unmounts and overwrites
  every disk it may write to.
- Do not run images that need `--privileged`, GPUs or long benchmark runs unless asked.

## Changelog

- Document every product change (new, removed or renamed images, changes to what an
  image measures or how, tool version bumps, output format changes, fixes that change
  results) in the root `CHANGELOG.md` in the same PR. Docs, CI and build changes and the
  `internal` images are not listed.
- Read `.agents/changelog.md` before editing `CHANGELOG.md`; it defines the sections,
  entry style and formatting.
