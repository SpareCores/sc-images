# References

Terms specific to this image, alphabetized. Common terms (e.g. Redis, `nice`,
`resource-tracker`) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs, which
also lists the terms each image uses.

- **`memtier_benchmark`**: An open-source load generator for Redis and
  Memcached, developed by Redis Ltd. It runs many concurrent connections over
  one or more threads, sends a configurable mix of SET and GET requests, and
  reports operations per second, latency averages, and latency percentiles per
  request type. [memtier_benchmark on GitHub](https://github.com/RedisLabs/memtier_benchmark)
- **Redis pipelining**: A technique in which a client sends several Redis
  commands without waiting for the reply to each one, then reads all replies
  together. It reduces the number of network round trips and system calls per
  command, which raises throughput, while each command waits longer for its
  reply. [Redis docs](https://redis.io/docs/latest/develop/use/pipelining/)
- **Redis SET command**: The Redis command that stores a string value under a
  key, replacing any existing value. Options can set an expiry time, or store
  the value only if the key does or does not exist yet.
  [Redis docs](https://redis.io/docs/latest/commands/set/)
