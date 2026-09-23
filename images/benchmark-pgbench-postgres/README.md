# benchmark-pgbench-postgres

This `pgbench`-driven PostgreSQL benchmark client for the Spare Cores fleet runs against a `postgres:18` server over the network (`SC_DB_HOST`). It simulates how a real application communicates with a managed or remote database instead of hosting the client and the server together.

## Contents

- **[Purpose](./docs/purpose.md)** - A brief summary of the reasons this benchmark exists for.
- **[Limitations](./docs/limitations.md)** - A detailed explanation of what exactly the benchmark is not meant for, or cannot do by either design or circumstance.
- **[Usage](./docs/usage.md)** - Instructions on how to run this benchmark and key environment variables.
- **[Workloads](./docs/workloads.md)** - Composed of the following parts:
  - A brief description of the [`pgbench_tpcb`](./docs/workloads.md#pgbench_tpcb) standard workload.
  - A detailed description of the [`pgbench_ro`](./docs/workloads.md#pgbench_ro) workload and its working mechanism.
- **[Design History](./docs/design-history.md)** - A historical description of the development of `pgbench_ro`, as well as a detailed version history.
