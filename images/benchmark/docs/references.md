# References

Terms specific to this image, alphabetized. Common terms (e.g. HTTPS, PyPI,
SMT, zstd) are defined in the [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs, which also lists the terms each image uses.

- **`bw_mem`**: The memory bandwidth tool of lmbench. It moves data through a
  buffer of a given size in one of several modes, such as `rd` (read and sum
  an array), `wr` (write a constant to each element), `rdwr` (read, then write
  the same location), and the copy modes `cp`, `fcp`, and `bcopy`. It reports
  the buffer size in MB and the bandwidth in MB/s; with `-P`, it runs several
  processes, each with its own buffer, and reports their total bandwidth.
  [`bw_mem` manual](https://github.com/intel/lmbench/blob/master/doc/bw_mem.8)
- **Geekbench**: A cross-platform CPU and GPU compute benchmark by Primate
  Labs. Its CPU benchmark runs workloads modeled on real applications, such as
  file compression, image processing, and web browsing, first on one core and
  then on all cores, and reports a Single-Core and a Multi-Core score. Geekbench
  6 scores are calibrated so that a baseline system scores 2500.
  [Geekbench](https://www.geekbench.com/)
- **Geekbench Browser**: The online database of Geekbench results run by
  Primate Labs. The free version of Geekbench uploads every result there, where
  it is public; the paid Pro version can also keep results offline.
  [Geekbench Browser](https://browser.geekbench.com/)
- **lmbench**: A suite of portable micro-benchmarks for Unix-like systems,
  written by Larry McVoy and Carl Staelin. Its tools each measure one basic
  operation, such as memory bandwidth and latency, context switches, or system
  call overhead.
  [lmbench](https://lmbench.sourceforge.net/)
- **`openssl speed`**: The OpenSSL command that measures the throughput of
  cryptographic algorithms, such as hashes and ciphers. It runs each algorithm
  for a fixed time on each of several block sizes and reports the bytes
  processed per second. With `-multi`, it runs several processes in parallel
  and reports their total.
  [`openssl speed` manual](https://docs.openssl.org/3.4/man1/openssl-speed/)
- **Silesia corpus**: A set of 12 files, \~211 MB in total, compiled by
  Sebastian Deorowicz at the Silesian University of Technology to test
  lossless compression on a mix of data types: text, executables, databases,
  images, and XML (Extensible Markup Language). Its `dickens` file holds the
  collected works of Charles Dickens, \~10 MB of English text.
  [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)
