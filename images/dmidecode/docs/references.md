# References

Terms specific to this image, alphabetized. Common terms (e.g. SMBIOS (System
Management BIOS), `dmidecode`) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs, which
also lists the terms each image uses.

- **DMI type**: The number that identifies the kind of a structure in SMBIOS
  (System Management BIOS), as the SMBIOS specification assigns it, such as 0
  for the firmware information, 4 for a processor, or 17 for a memory device.
  The `dmidecode` command prints
  it in the header line of each structure. Its `--type` (`-t`) option takes
  these numbers, or keywords such as `memory` or `processor` that stand for one
  or more types, to print only the matching structures.
  [Ubuntu manpage](https://manpages.ubuntu.com/manpages/noble/man8/dmidecode.8.html)
