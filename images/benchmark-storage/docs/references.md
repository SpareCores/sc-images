# References

Terms specific to this image, alphabetized. Common terms (e.g. GCP, I/O, IOPS,
PID) are defined in the [glossary](https://docs.sparecores.com/glossary) of the
Spare Cores docs, which also lists the terms each image uses.

- **`blkdiscard`**: A Linux command from util-linux that discards (trims) a
  range of blocks, by default all blocks of a device, and so deletes their
  data. An SSD then treats those blocks as unwritten and can erase them in the
  background, which restores the write speed of a fresh drive.
  [man page](https://man7.org/linux/man-pages/man8/blkdiscard.8.html)
- **DRAM** (Dynamic Random-Access Memory): The most common type of volatile
  computer memory, which stores each bit as a charge in a capacitor that has to
  be refreshed periodically. Servers use it as main memory, and many storage
  devices as a cache.
  [Wikipedia](https://en.wikipedia.org/wiki/Dynamic_random-access_memory)
- **`fio`**: The Flexible I/O Tester, an open-source storage benchmark and I/O
  workload generator. Each job describes an access pattern, such as random or
  sequential reads and writes, the block size, the queue depth, and the I/O
  engine, and `fio` reports the resulting IOPS, bandwidth, and latency.
  [fio docs](https://fio.readthedocs.io/en/latest/)
- **FTL** (Flash Translation Layer): The firmware layer of an SSD controller
  that maps the logical blocks the host sees to physical flash pages. Because
  flash has to be erased in large blocks before it is rewritten, the FTL writes
  new data to free pages, and runs garbage collection and wear leveling in the
  background.
  [Wikipedia](https://en.wikipedia.org/wiki/Flash_memory_controller#Flash_translation_layer_(FTL)_and_mapping)
- **HDD** (Hard Disk Drive): A storage device that reads and writes data with
  magnetic heads on spinning platters. It offers cheap capacity, but each
  random access waits for a head to move and a platter to turn, so it completes
  far fewer random operations per second than an SSD.
  [Wikipedia](https://en.wikipedia.org/wiki/Hard_disk_drive)
- **LVM** (Logical Volume Manager): The Linux tool that pools disks or
  partitions into volume groups and divides these into logical volumes, which
  can be resized while in use. It also supports snapshots, and striping or
  mirroring across disks.
  [man page](https://man7.org/linux/man-pages/man8/lvm.8.html)
- **NVMe** (Non-Volatile Memory Express): The interface specification for
  accessing non-volatile storage, most often SSDs, over PCI Express. It is
  designed for the parallelism of flash memory, with many deep command queues,
  and NVMe over Fabrics extends it to network transports.
  [NVM Express](https://nvmexpress.org/)
- **OVHcloud**: A French cloud provider, founded in 1999, with data centers in
  Europe, North America and Asia-Pacific. It offers dedicated (bare-metal)
  servers, public cloud instances based on OpenStack, and private cloud.
  [OVHcloud](https://www.ovhcloud.com/)
- **RAID** (Redundant Array of Independent Disks): A way to combine several
  disks into one logical device, for redundancy, speed or both. Common levels
  are RAID 0 (striping), RAID 1 (mirroring) and RAID 5 or 6 (striping with
  parity); on Linux, the `md` driver implements RAID in software.
  [Wikipedia](https://en.wikipedia.org/wiki/RAID)
- **SLC** (Single-Level Cell): Flash memory that stores one bit per cell, which
  makes it faster and more durable than flash storing more bits per cell, such
  as TLC or QLC. Many SSDs run part of their flash in SLC mode as a write
  cache, and write more slowly once a long write fills it.
  [Wikipedia](https://en.wikipedia.org/wiki/Multi-level_cell)
- **SMART** (Self-Monitoring, Analysis, and Reporting Technology): A
  monitoring system built into HDDs and SSDs that records health and error
  counters, such as reallocated sectors, power-on hours or wear level. Tools
  such as `smartctl` read these counters to spot failing disks early.
  [Wikipedia](https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology)
- **`smartctl`**: The command-line tool of the smartmontools package that
  reads SMART data, self-test logs and device information from ATA, SCSI and
  NVMe disks, and can start self-tests.
  [smartmontools](https://www.smartmontools.org/)
- **UTC** (Coordinated Universal Time): The primary time standard by which the
  world sets its clocks, the successor of Greenwich Mean Time. It has no time
  zone offset or daylight saving time, so UTC timestamps compare directly
  across regions.
  [Wikipedia](https://en.wikipedia.org/wiki/Coordinated_Universal_Time)
