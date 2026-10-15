# References

Terms specific to this image, alphabetized. Common terms (e.g. GCP, I/O, IOPS,
PID) are defined in the [glossary](https://docs.sparecores.com/glossary) of the
Spare Cores docs, which also lists the terms each image uses.

- **`blkdiscard`**: A Linux command that discards (trims) all blocks of a
  device, so an SSD treats them as unwritten.
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
- **FTL** (Flash Translation Layer): The part of an SSD controller that maps
  logical blocks to physical flash pages and runs garbage collection.
  [Wikipedia](https://en.wikipedia.org/wiki/Flash_memory_controller#Flash_translation_layer_(FTL)_and_mapping)
- **HDD** (Hard Disk Drive): A storage device that reads and writes data on
  spinning magnetic disks.
  [Wikipedia](https://en.wikipedia.org/wiki/Hard_disk_drive)
- **LVM** (Logical Volume Manager): The Linux tool that pools disks into volume
  groups and divides them into logical volumes.
  [man page](https://man7.org/linux/man-pages/man8/lvm.8.html)
- **NVMe** (Non-Volatile Memory Express): The interface specification for
  accessing non-volatile storage, most often SSDs, over PCI Express. It is
  designed for the parallelism of flash memory, with many deep command queues,
  and NVMe over Fabrics extends it to network transports.
  [NVM Express](https://nvmexpress.org/)
- **OVHcloud**: A European cloud provider.
  [OVHcloud](https://www.ovhcloud.com/)
- **RAID** (Redundant Array of Independent Disks): Several disks combined into
  one device for redundancy or speed, on Linux often with `md`.
  [Wikipedia](https://en.wikipedia.org/wiki/RAID)
- **SLC** (Single-Level Cell): Flash memory that stores one bit per cell. SSDs
  run part of their flash in SLC mode as a fast write cache.
  [Wikipedia](https://en.wikipedia.org/wiki/Multi-level_cell)
- **SMART** (Self-Monitoring, Analysis, and Reporting Technology): The health
  and error counters that disks report about themselves.
  [Wikipedia](https://en.wikipedia.org/wiki/Self-Monitoring,_Analysis_and_Reporting_Technology)
- **`smartctl`**: The command of `smartmontools` that reads SMART data from
  disks. [smartmontools](https://www.smartmontools.org/)
- **UTC** (Coordinated Universal Time): The time standard of the timestamps in
  the output, without a time zone offset.
  [Wikipedia](https://en.wikipedia.org/wiki/Coordinated_Universal_Time)
