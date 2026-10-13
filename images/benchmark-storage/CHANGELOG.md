# Changelog

## Renamed to `benchmark-storage`

The image was renamed from `storage-benchmark` to `benchmark-storage`.

## Drive Preconditioning and Health Snapshots

Partial fills and short write runs were measuring write-buffer burst instead of
sustained media performance. This change added the following:

- a discard (`blkdiscard`) and a sequential fill of each writable disk before
  testing
- longer write tests
- best-effort SMART and NVMe health snapshots before and after the run

Together, these make results comparable across vendors.

## Initial Version

The first version built `fio` 3.43 and ran a disk inventory and a single-disk
and group test plan against the host's real disks, for the upcoming production
task.
