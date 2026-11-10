# References

Terms specific to this image, alphabetized. Common terms (e.g. GCP, I/O, IOPS,
PID) are defined in the [glossary](https://docs.sparecores.com/glossary) of the
Spare Cores docs, which also lists the terms each image uses.

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
- **garbage collection**: The background work in which an SSD frees flash for
  new writes. Flash can only be erased in large blocks, so the drive copies the
  still-valid pages out of a block, then erases the whole block. It runs when
  spare area runs low, and it slows the writes that arrive meanwhile, which is
  why an SSD's sustained write speed is lower than its burst speed. Programming
  languages use the same term for reclaiming unused memory.
  [Wikipedia](https://en.wikipedia.org/wiki/Garbage_collection_%28SSD%29)
- **SLC** (Single-Level Cell): Flash memory that stores one bit per cell, which
  makes it faster and more durable than flash storing more bits per cell, such
  as TLC or QLC. Many SSDs run part of their flash in SLC mode as a write
  cache, and write more slowly once a long write fills it.
  [Wikipedia](https://en.wikipedia.org/wiki/Multi-level_cell)
