# Hardware Info

This image bundles Linux command-line tools, such as `lshw` (Hardware Lister)
and `lscpu` (CPU architecture information), that report a server's CPU,
memory, disk, and topology details.
The Inspector of [Spare Cores Navigator](https://docs.sparecores.com/overview)
runs them on cloud servers. This image has no default command and no harness of
its own: each run executes the one command passed after the image name. This
image is not a benchmark, so nothing is timed or repeated.

In production, the Inspector runs up to six commands on each server type.
The commands print JSON (`lshw`, `lscpu`, and three `lsblk` commands) or XML
(`lstopo`). The Inspector stores their output in the
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data)
repository. This image adds nothing to what the command prints.

## Tools

This image is built on `ubuntu:24.04` and includes the following tools:

| Tool | Package | Version | Production task |
| ------ | --------- | --------- | ----------------- |
| `lshw` | `lshw` from Ubuntu | Ubuntu archive at build time | `lshw` |
| `lscpu` | util-linux, from the base image | 2.39.3 in production | `lscpu` |
| `lsblk` | util-linux, from the base image | 2.39.3 in production | `lsblk`, `lsblk_discard`, `lsblk_topo` |
| `lstopo` | `hwloc-nox` from Ubuntu | 2.10.0 in production | `lstopo` |
| `jq` | `jq` from Ubuntu | Ubuntu archive at build time | `lscpu` |
| LIKWID (Like I Knew What I'm Doing) tools | built from source | 5.3.0 | none |

Despite its name, this image does not install the
[`hwinfo`](https://github.com/openSUSE/hwinfo) tool.

The `hwloc-nox` package is the hwloc (Portable Hardware Locality) build
without graphical display support, and provides `lstopo`. In production, `jq`
only reformats the JSON that `lscpu` prints.

LIKWID is the only pinned tool: the [`Dockerfile`](./Dockerfile) builds
release 5.3.0 from the upstream source, installs it under `/usr/local`, and
adds `/usr/local/bin` to `PATH`. No production task runs it.

The other tools come from the Ubuntu 24.04 archive after an `apt-get
upgrade`, so their versions can change between image builds. The versions
marked "in production" come from the `version` field that each production run
records in its `meta.json` file in `sc-inspector-data`; for the `lshw` task,
the field is empty.

The Inspector collects more hardware details with the following images:

- **[`dmidecode`](../dmidecode/README.md):** Prints the SMBIOS (System
  Management BIOS) tables, from which the Navigator takes the memory
  generation and speed of each server type.
- **[Virtualization](../virtualization/README.md):** Checks whether KVM
  (Kernel-based Virtual Machine) can be used, which the Navigator shows as the
  hardware virtualization attribute of each server type.

## Usage

This image sets no default command, so append the command to run after the
image name. To print the CPU details as JSON, matching the
`lscpu` production task:

```shell
docker run --rm --privileged --network host ghcr.io/sparecores/hwinfo:main \
  bash -c 'lscpu -JB | jq'
```

To print the hardware tree as JSON, as the `lshw` production task does:

```shell
docker run --rm --privileged --network host ghcr.io/sparecores/hwinfo:main \
  lshw -json
```

To print the hwloc topology as XML, as the `lstopo` production task does:

```shell
docker run --rm --privileged --network host ghcr.io/sparecores/hwinfo:main \
  lstopo --of xml
```

These commands use the same Docker options as production. This image reads no
environment variables or configuration files of its own, and runs each command
as root in `/`.

This image is built for AMD64 and Arm64. Unlike the
[benchmark images](../../README.md#benchmarks), it
runs the command outside `resource-tracker`, so no resource usage is sampled.

### Production Setup

The Inspector runs each command as a separate task, with the following
commands and timeouts:

| Task | Command | Timeout |
| ------ | --------- | --------- |
| `lshw` | `lshw -json` | 5 min |
| `lscpu` | `bash -c 'lscpu -JB \| jq'` | 5 min |
| `lsblk` | `lsblk -O -Jdb` | 1 min |
| `lsblk_discard` | `lsblk -DJdb` | 1 min |
| `lsblk_topo` | `lsblk -tJdb` | 1 min |
| `lstopo` | `lstopo --of xml` | 30s |

Production runs also differ from a plain `docker run` in the following ways:

- **Docker Options:** Production runs every task with `--privileged` and host
  networking (`--network host`), as the commands above do. A plain
  `docker run` without these options runs unprivileged on Docker's default
  network.
- **Timeout:** Production stops each task after the timeout in the table. A
  plain `docker run` sets no time limit. Production runs take a few seconds
  per task.
- **Servers:** Production runs the tasks on every server type, except those on
  the blocklist in
  [`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
  in [`sc-crawler`](https://github.com/SpareCores/sc-crawler).
- **Server Start:** The `lsblk`, `lsblk_discard`, `lsblk_topo`, and `lstopo`
  tasks never start a server on their own; they run only when another task,
  such as `lshw` or `lscpu`, has started the server. These four tasks have
  results for fewer server types than `lshw` and `lscpu`.

The commands and timeouts are in the six tasks above in
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in [`sc-inspector`](https://github.com/SpareCores/sc-inspector), and the
default Docker options are in
[`inspector/lib.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/lib.py).

### Outputs

The production commands print the following to stdout:

- **`lshw`:** A JSON tree of hardware nodes, each with fields such as `id`,
  `class`, `description`, `product`, `vendor`, `width`, `configuration`, and
  `capabilities`.
- **`lscpu`:** A JSON object, `{"lscpu": [...]}`, with one `field` and `data`
  pair per line of the text table that `lscpu` prints by default. The `-B`
  option prints sizes in bytes.
- **`lsblk`, `lsblk_discard`, and `lsblk_topo`:** A JSON object,
  `{"blockdevices": [...]}`, with one entry per whole disk (`-d`) and sizes in
  bytes (`-b`). The `lsblk` task prints all columns (`-O`), `lsblk_discard`
  the discarding capabilities (TRIM, UNMAP) of each device (`-D`), and
  `lsblk_topo` the block-device topology columns (`-t`).
- **`lstopo`:** The hwloc topology as XML. The Inspector also renders it as an
  `lstopo.svg` image next to the XML.

### Production Results

On [`aws/m7g.large`](https://sparecores.com/server/aws/m7g.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lscpu/stdout)),
the `lscpu` task printed the following, shortened here:

```json
{
  "lscpu": [
    {
      "field": "Architecture:",
      "data": "aarch64"
    },
    ...
    {
      "field": "CPU(s):",
      "data": "2"
    },
    ...
    {
      "field": "Model name:",
      "data": "Neoverse-V1"
    },
    ...
  ]
}
```

The same server type has recorded output for the
[`lshw` task](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lshw/stdout),
the [`lsblk` task](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lsblk/stdout),
and the [`lstopo` task](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lstopo/stdout),
as well as the rendered
[`lstopo.svg`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lstopo/lstopo.svg).

## References

This image's [Glossary](./docs/references.md) defines the terms of its own
tools, such as `lshw`, hwloc, and LIKWID. The
[Spare Cores Glossary](https://docs.sparecores.com/glossary) defines common
terms, such as `lscpu`, `lsblk`, and util-linux.
