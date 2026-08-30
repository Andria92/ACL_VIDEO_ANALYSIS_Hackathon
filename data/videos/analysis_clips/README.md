# Canonical analysis clips

This directory contains only the short video clips used by registered analysis cases.
Full matches, source downloads, failed downloads, and regenerable QC overlays do not belong here.

- `manifest.csv` maps every registered case to a project-relative clip path and SHA-256 checksum.
- Keep existing clip filenames stable because saved annotations and analysis outputs reference them.
- Store new cutter output in this directory.
- Track video binaries with Git LFS using the repository `.gitattributes` rules.
