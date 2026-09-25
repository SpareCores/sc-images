# Purpose

In many other benchmarks, the missing piece is the lack of actual, proper database measurements - a gap this benchmark closes by scoring how how cloud servers perform under RDBMS ([Relational Database Management System](https://en.wikipedia.org/wiki/Relational_database)) workloads.

In this benchmark, two deployment models are measured with the same client:

- **IaaS (Infrastructure as a Service)**: self-hosted PostgreSQL on a cloud VM, driven by a separate client VM.
- **DBaaS (Database as a Service)**: the cloud vendor's managed PostgreSQL offering - similar hardware, but the vendor provisions, manages, and tunes the engine.

This benchmark helps users compare same-hardware, self-managed, and managed instances through a single, comparable headline score in TPM, and a concurrency profile that measures throughput at various numbers of connected clients.
