# References

Terms specific to this image, alphabetized. Common terms (e.g. HTTPS, PyPI,
SMT, zstd) are defined in the [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs, which also lists the terms each image uses.

- **Geekbench Browser**: The online database of Geekbench results run by
  Primate Labs. The free version of Geekbench uploads every result there, where
  it is public; the paid Pro version can also keep results offline.
  [Geekbench Browser](https://browser.geekbench.com/)
- **`openssl speed`**: The OpenSSL command that measures the throughput of
  cryptographic algorithms, such as hashes and ciphers. It runs each algorithm
  for a fixed time on each of several block sizes and reports the bytes
  processed per second. With `-multi`, it runs several processes in parallel
  and reports their total.
  [`openssl speed` manual](https://docs.openssl.org/3.4/man1/openssl-speed/)
