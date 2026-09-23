# Purpose

[Spare Cores](https://sparecores.com) monitors and publishes empirical performance data for over 5,000 cloud server types as part of our [Navigator](https://sparecores.com/servers) project. Among the measured metrics are the following and more:

- Memory bandwidth
- OpenSSL speed
- Compression algorithms
- Redis and static web-serving throughput
- LLM inference speed

In many other benchmarks, the missing piece is the lack of actual, proper database measurements - a gap this benchmark closes by scoring how how cloud servers perform under [Relational Database Management System](https://en.wikipedia.org/wiki/Relational_database) (RDBMS) workloads.

In this benchmark, two deployment models are measured with the same client:

- **Infrastructure as a Service (IaaS)**: self-hosted PostgreSQL on a cloud VM, driven by a separate client VM.
- **Database as a Service (DBaaS)**: the cloud vendor's managed PostgreSQL offering - similar hardware, but the vendor provisions, manages, and tunes the engine.

  This benchmark helps users compare same-hardware, self-managed, and managed instances through a single, comparable headline score in TPM, and a concurrency profile that measures throughput at various numbers of connected clients.
