# References

Terms specific to this image, alphabetized. Common terms (e.g. HTTPS, PyPI,
SMT, zstd) are defined in the [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs, which also lists the terms each image uses.

- **`brotli`**: A lossless compression format by Google, standardized in RFC
  7932. It combines LZ77-style matching, Huffman coding, and context modeling
  with a built-in dictionary of common web text, and is widely used to compress
  HTTP responses. Its quality levels range from 0 (fastest) to 11 (smallest
  output).
  [RFC 7932](https://www.rfc-editor.org/rfc/rfc7932)
- **`bw_mem`**: The memory bandwidth tool of LMbench. It moves data through a
  buffer of a given size in one of several modes, such as `rd` (read and sum an
  array), `wr` (write a constant to each element), `rdwr` (read, then write the
  same location), and the copy modes `cp`, `fcp`, and `bcopy`. It reports the
  buffer size in MB and the bandwidth in MB/s; with `-P`, it runs several
  processes, each with its own buffer, and reports their total bandwidth.
  [`bw_mem` manual](https://github.com/intel/lmbench/blob/master/doc/bw_mem.8)
- **`bzip3`**: A lossless compressor by Kamila Szewczyk that, like bzip2, is
  built on the Burrows-Wheeler transform, combined with context modeling. It
  typically compresses text better than bzip2, and its configurable block size
  trades memory use for compression ratio.
  [bzip3 on GitHub](https://github.com/kspalaiologos/bzip3)
- **Geekbench Browser**: The online database of Geekbench results run by
  Primate Labs. The free version of Geekbench uploads every result there, where
  it is public; the paid Pro version can also keep results offline.
  [Geekbench Browser](https://browser.geekbench.com/)
- **LMbench**: A suite of portable micro-benchmarks for Unix-like systems,
  written by Larry McVoy and Carl Staelin. Its tools each measure one basic
  operation, such as memory bandwidth and latency, context switches, or system
  call overhead.
  [LMbench](https://lmbench.sourceforge.net/)
- **`openssl speed`**: The OpenSSL command that measures the throughput of
  cryptographic algorithms, such as hashes and ciphers. It runs each algorithm
  for a fixed time on each of several block sizes and reports the bytes
  processed per second. With `-multi`, it runs several processes in parallel
  and reports their total.
  [`openssl speed` manual](https://docs.openssl.org/3.4/man1/openssl-speed/)
- **Silesia corpus**: A set of 12 files, ~211 MB in total, compiled by
  Sebastian Deorowicz at the Silesian University of Technology to test lossless
  compression on a mix of data types: text, executables, databases, images, and
  XML. Its `dickens` file holds the collected works of Charles Dickens, ~10 MB
  of English text.
  [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia)
- **`zpaq`**: A lossless compressor and archive format by Matt Mahoney, based
  on context mixing, which predicts each bit from many models and combines their
  predictions. It reaches very high compression ratios, but compresses and
  decompresses slowly, at about the same speed in both directions. Its methods
  range from 1 (fastest) to 5 (smallest output).
  [ZPAQ](https://mattmahoney.net/dc/zpaq.html)
