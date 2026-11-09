# Adding an Image

CI auto-discovers every folder under `images/` (any dir with a `Dockerfile` or `prepare.sh`) and builds it. Per-image behavior is declared with optional metadata files. The defaults cover simple single-stage images, so most folders need none of the following:

| File | Meaning | Default |
|------|---------|---------|
| `DEPENDS_ON` | image folder names that must be published **before** this one (one per line) | none |
| `PLATFORMS` | arches to build | `amd64 arm64` |
| `CONTEXT` | build context, repo-relative | `images/<folder>` |
| `DOCKERFILE` | Dockerfile path, repo-relative | `images/<folder>/Dockerfile` |
| `TARGET` | build target stage | none |
| `BUILD_ARGS` | `KEY=VALUE` lines; tokens `${ARCH}`, `${VLLM_VERSION}`, `${GUIDELLM_VERSION}`, `${RESOURCE_TRACKER_VERSION}` | none |
| `ZRAM` | enable compressed swap on the builder (`true`/`1`/`yes`, or a PERCENT e.g. `125`) | off |
| `SCCACHE` | enable sccache S3 compile cache for source builds (`true`/`1`/`yes`) | off |
| `prepare.sh` | pre-build hook (clone sources, patch Dockerfile, emit BuildKit secret + parallelism) | none |
| `DESCRIPTION` | one-sentence summary; published as the `org.opencontainers.image.description` label and index annotation, so it shows on the GHCR package page | none (CI warns) |
| `CATEGORY` | group name: `benchmark`, `hardware-discovery`, `service`, or `base`; published as the `com.sparecores.image.category` label and used to group the [Images](../README.md#images) tables | none (CI warns) |

[`resolve-build-plan.sh`](../.github/scripts/resolve-build-plan.sh) topologically sorts folders by `DEPENDS_ON` into levels (0 = no deps). [`push.yml`](../.github/workflows/push.yml) builds each level via the reusable [`build-level.yml`](../.github/workflows/build-level.yml) and publishes it before the next, so a dependency is always available as a base image; [`read-image-config.sh`](../.github/scripts/read-image-config.sh) resolves each folder's config at build time. No image names are hardcoded in the workflow.

Example: every `benchmark-*` image has `DEPENDS_ON: resource-tracker`; the `benchmark-vllm-cpu-avx2` image also depends on `vllm-cpu-base-avx2` (source-compiled via `prepare.sh`). The `benchmark-vllm-gpu` image uses Hub `vllm/vllm-openai` for amd64 and arm64. Add a new dependent image by dropping a folder in `images/` with a `DEPENDS_ON` file.

See [CI and Build Internals](ci.md) for how CI builds, caches, tracks, and publishes the images.

## Acronyms and References

Spell out an acronym on its first use in each page, as `CRF (Constant Rate Factor)` or `Constant Rate Factor (CRF)`, without linking the expansion: put the link in the term's glossary entry instead. On [docs.sparecores.com](https://docs.sparecores.com/glossary), a term spelled out like this gets a tooltip with its glossary entry, and the docs build warns when the expansion differs from the glossary, links to a URL its glossary entry has already, or an image's docs never spell out a term they use.

Common terms (e.g. AWS, vCPU, SMT) are defined in the docs site's [glossary](https://docs.sparecores.com/glossary), `glossary.yaml` in sc-docusaurus: terms used in several sections or images, and general computing, cloud, and Spare Cores Navigator terms even if one image uses them. They get tooltips on every page of the docs and in the sparecores.com articles, so a common term must mean the same thing on every page: a term that means something else in another field, such as token or decoding, goes in the `references.md` of the images that use it.

Define the terms of an image's own tool or method in its `docs/references.md`, and the terms that the images of a shared manual use, such as the vLLM options, in the manual's `docs/references.md` (e.g. `vllm-common/docs/references.md`). These terms get tooltips only on the pages of their folder, and on the pages of the images that link to that `references.md` from any of their pages, which is also how readers on GitHub find them. Two folders may define the same term differently, but not a common term. Use one bullet per term, alphabetized, with wrapped lines indented:

```md
- **CRF** (Constant Rate Factor): The quality-based rate control of x264 and x265.
  [FFmpeg wiki](https://trac.ffmpeg.org/wiki/Encode/H.264#crf)
- **`netem`** (Network Emulator): A Linux tool to add delay to a network interface.
  [SRT cookbook](https://srtlab.github.io/srt-cookbook/how-to-articles/using-netem-to-emulate-networks.html)
```

The expansion, the description and the trailing links are each optional, but an entry needs at least one of them; write commands and tools as inline code. Further bold forms after the term are its aliases, such as a plural: `- **token**, **tokens**: …`. The docs site replaces the page with one listing every glossary term the image's docs use, so the file itself only needs the image's own terms. Describe each term in itself, in one paragraph of 1–5 sentences, like a short encyclopedia entry: the Glossary page lists it under its image or manual, so don't describe the term in relation to an image.

Link an image's own `references.md` from its README, and a shared one wherever the image uses its terms, e.g. in the References section of the README:

```md
See [References](./docs/references.md) for the terms specific to this image.
The terms of the vLLM benchmarks are defined in their
[references](../../vllm-common/docs/references.md).
```

A term that pages outside its folder use, such as other images, the Benchmarks section or the Crawler docs, belongs in `glossary.yaml`. The docs build warns about such terms, and about a README without a link to its own `references.md`.

## Navigator Charts

To show a chart of the Spare Cores Navigator on [docs.sparecores.com](https://docs.sparecores.com/images), put an `<!-- embed -->` marker on its own line, right above a link to the chart's embed URL (the `src` of the embed code the Navigator gives):

```md
<!-- embed height=517px -->
[GPU Bandwidth and Latency of ecs.gn5-c4g1.xlarge](https://sparecores.com/embed/server/alicloud/ecs.gn5-c4g1.xlarge/nvbandwidth)
```

GitHub hides the marker and shows the link; the docs site shows the chart instead, with the link text as its title for screen readers. The height is optional (500px by default); the docs site sets the rest of the style. A marker that isn't right above such a link, or has an unknown or invalid setting, leaves the link as it is, and the docs build warns about it.
