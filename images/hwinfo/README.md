# Hardware Info

This image bundles Linux command-line tools, such as `lshw` (Hardware Lister)
and `lscpu`, that report a server's CPU, memory, disk, and topology details.
The Inspector of [Spare Cores Navigator](https://docs.sparecores.com/overview)
runs them on cloud servers. This image has no default command and no harness of
its own: each run executes the one command passed after the image name. This
image is not a benchmark, so nothing is timed or repeated.

The output is whatever that command prints. In production, the Inspector runs
up to six commands on each server type, and stores their JSON or XML output in
the [`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data)
repository.

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

Despite its name, this image does not install the `hwinfo` tool.

The Inspector collects hardware details with two more images. The
[`dmidecode`](../dmidecode/README.md) image prints the SMBIOS (System
Management BIOS) tables, from which the Navigator takes the memory generation
and speed of each server type. The
[Virtualization](../virtualization/README.md) image checks whether KVM
(Kernel-based Virtual Machine) can be used.

The `hwloc-nox` package is the hwloc (Portable Hardware Locality) build
without graphical display support, and provides `lstopo`. In production, `jq`
only reformats the JSON that `lscpu` prints.

LIKWID is the only pinned tool: the [`Dockerfile`](./Dockerfile) builds
release 5.3.0 from the upstream source, installs it under `/usr/local`, and
adds `/usr/local/bin` to `PATH`. No production task runs it. The other tools
come from the Ubuntu 24.04 archive after an `apt-get upgrade`, so their
versions can change between image builds. The versions marked "in production"
come from the `version` field that each production run records in its
`meta.json` file in `sc-inspector-data`; for the `lshw` task, the field is
empty.

## Usage

This image sets no default command, so append the command to run after the
image name. To print the CPU details as JSON, as the `lscpu` production task
does:

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
as root in `/`. Unlike the benchmark images, it does not run the command under
`resource-tracker`. This image is built for AMD64 and Arm64.

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
  in `sc-crawler`.
- **Server Start:** The `lsblk`, `lsblk_discard`, `lsblk_topo`, and `lstopo`
  tasks never start a server on their own; they run only when another task,
  such as `lshw` or `lscpu`, has started the server. These four tasks have
  results for fewer server types (\~2410 as of November 2026) than `lshw` and
  `lscpu` (\~3790).

The commands and timeouts are in the six tasks above in
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in `sc-inspector`, and the default Docker options are in
[`inspector/lib.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/lib.py).

### Output

This image adds nothing to the output of the command it runs. The production
commands print the following to stdout:

- **`lshw`:** A JSON tree of hardware nodes, each with fields such as `id`,
  `class`, `description`, `product`, `vendor`, `width`, `configuration`, and
  `capabilities`.
- **`lscpu`:** A JSON object, `{"lscpu": [...]}`, with one `field` and `data`
  pair per line of the text table that `lscpu` prints by default. The `-B`
  option prints sizes in bytes.
- **`lsblk`, `lsblk_discard`, and `lsblk_topo`:** A JSON object,
  `{"blockdevices": [...]}`, with one entry per whole disk (`-d`) and sizes in
  bytes (`-b`). The `lsblk` task prints all columns (`-O`), `lsblk_discard`
  the discard columns (`-D`), and `lsblk_topo` the topology columns (`-t`).
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

The same server type has recorded outputs of the
[`lshw`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lshw/stdout),
[`lsblk`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lsblk/stdout),
and [`lstopo`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lstopo/stdout)
tasks, and the rendered
[`lstopo.svg`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7g.large/0/lstopo/lstopo.svg).

## References

The [references](./docs/references.md) define the terms of this image's own
tools, such as `lshw`, hwloc, and LIKWID. The
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs
defines common terms, such as `lscpu`, `lsblk`, and util-linux.
