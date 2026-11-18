# References

Terms specific to this image, alphabetized. Common terms (e.g. KVM
(Kernel-based Virtual Machine), `/dev/kvm`) are defined in the [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs, which also lists the terms each image uses.

- **`kvm-ok`**: A command of Ubuntu's `cpu-checker` package that reports
  whether a system can run VMs accelerated by KVM (Kernel-based Virtual Machine). On x86 CPUs, it checks for the
  Intel or AMD hardware virtualization flag and whether `/dev/kvm` exists; on
  Arm and other architectures, it checks only for `/dev/kvm`. It prints what it
  found and exits 0 when KVM acceleration can be used.
  [Ubuntu manpage](https://manpages.ubuntu.com/manpages/noble/man1/kvm-ok.1.html)
