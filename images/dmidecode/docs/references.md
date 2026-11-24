# Glossary

Terms specific to this image, alphabetized. Common terms (e.g. SMBIOS (System
Management BIOS), `dmidecode`) are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary), which also
lists the terms each image uses.

- **DMI type**: The number that the SMBIOS (System Management BIOS)
  specification assigns to identify a structure's kind, such as 0 for the
  firmware information, 4 for a processor, or 17 for a memory device.
  The `dmidecode` command prints the type in each structure's header line.
  The command's `--type` (`-t`) option takes
  these numbers, or keywords such as `memory` or `processor` that stand for one
  or more types, to print only the matching structures.
  [Ubuntu man page](https://manpages.ubuntu.com/manpages/noble/man8/dmidecode.8.html)
