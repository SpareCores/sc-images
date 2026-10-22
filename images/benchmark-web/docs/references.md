# References

Terms specific to this image, alphabetized. Common terms (e.g. HTTP, `nice`,
`resource-tracker`) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs, which
also lists the terms each image uses.

- **`binserve`**: An open-source static web server written in Rust, distributed
  as a single binary and configured with one JSON file. It maps URL routes to
  files and directories, can keep files cached in memory, and can also serve
  encrypted connections and list directory contents.
  [binserve on GitHub](https://github.com/mufeedvh/binserve)
- **`wrk`**: An open-source HTTP benchmarking tool written in C. It generates
  load from a few threads, each handling many open connections through an event
  loop, and reports latency and requests-per-second statistics. Lua scripts can
  customize the requests it sends and process the responses.
  [wrk on GitHub](https://github.com/wg/wrk)
