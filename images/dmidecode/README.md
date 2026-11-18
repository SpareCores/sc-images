# dmidecode

This image runs `dmidecode` to read a server's SMBIOS (System Management BIOS)
tables, also called DMI (Desktop Management Interface) tables. The tables
describe the server's parts, such as its firmware, system, baseboard,
processors, and memory devices, as the cloud vendor's firmware or hypervisor
reports them. This image only installs `dmidecode` from Ubuntu 24.04, with no
harness or default command of its own. This image is not a benchmark, so
nothing is timed or repeated.

This image prints the text output of `dmidecode` on stdout, one block per
SMBIOS structure. In production, the
[Inspector](https://github.com/SpareCores/sc-inspector) of
[Spare Cores Navigator](https://docs.sparecores.com/overview), which runs the
images on each server type, also parses the text into JSON.

The Navigator takes the memory generation and speed of each server type from
that JSON, and some processor details when `lscpu` reports none. See
[How the Navigator Uses the Output](#how-the-navigator-uses-the-output) for the
fields it reads.

## Usage

To print all SMBIOS structures of a host via Docker, with the command
production runs:

```shell
docker run --rm --privileged ghcr.io/sparecores/dmidecode:main dmidecode
```

The output describes the machine whose kernel runs the container.

This image keeps the `bash` default command of `ubuntu:24.04`, so pass
`dmidecode` after the image name. The `--privileged` option is required.
To print only the memory structures, append a `dmidecode` option such as
`-t memory` after the command:

```shell
docker run --rm --privileged ghcr.io/sparecores/dmidecode:main \
  dmidecode -t memory
```

This image defines no environment variables or build arguments, and runs
`dmidecode` as root. Unlike the benchmark images, it does not run
`dmidecode` under `resource-tracker`: measuring the resource usage of a
hardware inspection that takes seconds would add nothing.

The [`Dockerfile`](./Dockerfile) installs `dmidecode` from the Ubuntu 24.04
archive after an `apt-get upgrade`, without a pinned version, so the version
can change between image builds. As of November 2026, all successful
production runs in the
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data)
repository recorded version 3.5. This image is built for AMD64 and Arm64.

### Production Setup

The Navigator's Inspector runs the same command as the first example above,
`dmidecode` with no options. Its runs differ from the commands above in the
following ways:

- **Docker Options:** Production runs the container detached, with
  `--privileged` and host networking (`--network host`). The commands above
  use Docker's default network.
- **Timeout:** Production stops the container if it still runs 5 minutes
  after it starts. A plain `docker run` sets no time limit. Counted from when
  the Inspector picks up the run (so including the image pull), the median
  successful production run took \~10 seconds, and 95% of successful runs
  finished within \~27 seconds.
- **Version and Parsing:** Production records the output of
  `dmidecode --version` as the version of each run, and parses the text into
  JSON, as [Output](#output) describes. A plain `docker run` prints the text
  only.
- **Servers:** Production runs `dmidecode` on every server type, except those
  on the blocklist in
  [`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
  in `sc-crawler`.

The production settings are in the `dmidecode` task of
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in `sc-inspector`, with the default Docker options of
[`inspector/lib.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/lib.py).

### Output

This image adds nothing to the output of `dmidecode` and writes no files. The
text on stdout starts with a header that gives the `dmidecode` and SMBIOS
versions and the number of structures. Then comes one block per SMBIOS
structure, with the following lines:

1. A header line with the structure's handle (the number other structures use
   to refer to it, as `Array Handle` does), DMI type, and size in bytes, such
   as `Handle 0x0009, DMI type 17, 34 bytes`.
2. The structure's name, such as `Memory Device`.
3. The structure's fields as indented `Key: Value` lines. A field with several
   values, such as `Characteristics`, lists one value per line, indented
   further.

The content depends on what the cloud vendor's firmware or hypervisor exposes.
On AWS (Amazon Web Services), the `System Information` structure gives
`Amazon EC2` as the manufacturer and the server type, such as `m5.large`, as
the product name. On Hetzner Cloud, it gives `Hetzner` and `vServer`. The
SMBIOS version and the number of structures also vary by server type: for
example, [`aws/m5.large`](https://sparecores.com/server/aws/m5.large) reports
SMBIOS 2.7 with 13 structures, and
[`hcloud/cx22`](https://sparecores.com/server/hcloud/cx22) reports SMBIOS 2.8
with 11.

Fields that the vendor does not fill in read `Not Specified`, `Unknown`, or
`Not Provided` on `aws/m5.large`, and the parser keeps them as text.

The output also includes identifiers of the individual server, such as the
`Serial Number` and `UUID` fields and, on AWS, the instance ID as the
baseboard's `Asset Tag`.

Production parses the text with the Inspector's
[`parse.dmidecode`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/parse.py)
parser and saves the result as `parsed.json`, a JSON array with one object per
SMBIOS structure. The parser is part of the Inspector, not of this image. Each
object has the following fields:

| Field | Meaning |
| ------ | ------ |
| `handle` | The `id`, `type`, and `bytes` of the structure's header line, all as strings |
| `name` | The structure's name, such as `BIOS Information` or `Memory Device` |
| `props` | The structure's fields as key-value pairs, with a list for a field with several values. Values with a unit, such as sizes, speeds, and widths, become numbers in the base unit (`B`, `Hz`, `T/s`, `bits`, `V`, or `s`), with the unit in a separate `<key>:unit` field, such as `"ROM Size": 65536` and `"ROM Size:unit": "B"`. Byte sizes use multiples of 1,024 (`8 GB` becomes `8589934592`), and other units multiples of 1,000 (`2666 MT/s` becomes `2666000000`). Dates become timestamps, such as `2017-10-16T00:00:00` |

### Production Results

On [`aws/m5.large`](https://sparecores.com/server/aws/m5.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/dmidecode/stdout)),
`dmidecode` printed the following, shortened here to the header and the
`Memory Device` structure:

```text
# dmidecode 3.5
Getting SMBIOS data from sysfs.
SMBIOS 2.7 present.
13 structures occupying 567 bytes.
Table at 0xBBE5A000.
...
Handle 0x0009, DMI type 17, 34 bytes
Memory Device
    Array Handle: 0x0008
    ...
    Size: 8 GB
    Form Factor: DIMM
    ...
    Type: DDR4
    ...
    Speed: 2666 MT/s
    ...
```

The Inspector parsed the same structure into the following object
([recorded parsed output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/dmidecode/parsed.json)),
shortened here:

```json
{
  "handle": {
    "id": "0x0009",
    "type": "17",
    "bytes": "34"
  },
  "name": "Memory Device",
  "props": {
    "Array Handle": "0x0008",
    ...
    "Size": 8589934592,
    "Type": "DDR4",
    "Speed": 2666000000,
    ...
    "Size:unit": "B",
    "Speed:unit": "T/s"
  }
}
```

## How the Navigator Uses the Output

The Navigator reads only the following SMBIOS structures from the parsed
output to fill in the attributes of each server type:

- **Memory Device:** The memory generation comes from `Type`, and the memory
  speed from `Speed`. The memory amount comes from the memory devices only
  when the tools of the [Hardware Info](../hwinfo/README.md) image report
  none.
- **Processor Information:** The CPU speed (`Max Speed`) and the CPU
  manufacturer, family, and model (`Version`) come from this structure, each
  only when `lscpu` reports none. The Navigator discards a CPU speed of 2 GHz
  on GCP (Google Cloud Platform) as unreliable.

## References

The [references](./docs/references.md) define the terms of this image's own
tool, such as DMI type. The [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs defines common terms, such as SMBIOS, SMBIOS structure,
and `dmidecode`.
