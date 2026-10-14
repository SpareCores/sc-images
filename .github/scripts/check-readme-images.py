#!/usr/bin/env python3
"""Check that the Images tables of README.md match the image folders.

Usage: check-readme-images.py   (run from repo root)

Every image folder (any dir under images/ with a Dockerfile or prepare.sh, as
in resolve-build-plan.sh) must be listed exactly once, in the table under the
heading of its CATEGORY, with its DESCRIPTION (wrapped lines joined, as by
read_text in read-image-config.sh) as the description. The docs site
(docs.sparecores.com/images) shows this README, and groups its sidebar by
these tables.
"""

import re
import sys
from pathlib import Path

IMAGES_DIR = Path("images")
README = Path("README.md")
# table heading in README.md of each CATEGORY value
HEADINGS = {
    "benchmark": "Benchmarks",
    "hardware-discovery": "Hardware Discovery",
    "service": "Services",
    "base": "Base Images",
}


def read_text(path: Path) -> str:
    """Same as read_text in read-image-config.sh: wrapped lines joined into one."""
    if not path.is_file():
        return ""
    return " ".join(path.read_text().split())


def readme_rows() -> list[tuple[str, str, str]]:
    """(heading, folder, description) of each row of the tables under ## Images."""
    rows = []
    heading = None
    in_images = False
    link = re.compile(r"^\|\s*\[[^\]]*\]\(images/([^/)#]+)/?\)\s*\|(.*)\|\s*$")
    for line in README.read_text().splitlines():
        if line.startswith("## "):
            in_images = line.strip() == "## Images"
            heading = None
        elif in_images and line.startswith("### "):
            heading = line[4:].strip()
        elif in_images and (row := link.match(line)):
            description = " ".join(row.group(2).replace("\\|", "|").split())
            rows.append((heading, row.group(1), description))
    return rows


def main() -> int:
    images = sorted(
        d.name
        for d in IMAGES_DIR.iterdir()
        if d.is_dir() and ((d / "Dockerfile").is_file() or (d / "prepare.sh").is_file())
    )
    rows = readme_rows()
    errors = []

    listed = [folder for _, folder, _ in rows]
    for folder in sorted(set(listed)):
        if listed.count(folder) > 1:
            errors.append(f"{folder}: listed {listed.count(folder)} times")
        if folder not in images:
            errors.append(f"{folder}: listed, but not an image folder")

    for image in images:
        matches = [(heading, description) for heading, folder, description in rows if folder == image]
        if not matches:
            errors.append(f"{image}: not listed")
            continue
        heading, description = matches[0]
        category = read_text(IMAGES_DIR / image / "CATEGORY")
        if not category:
            errors.append(f"{image}: no CATEGORY file")
        elif category not in HEADINGS:
            errors.append(f"{image}: CATEGORY {category!r} has no heading in HEADINGS of {sys.argv[0]}")
        elif heading != HEADINGS[category]:
            errors.append(f"{image}: listed under {heading!r} instead of {HEADINGS[category]!r} (CATEGORY {category})")
        expected = read_text(IMAGES_DIR / image / "DESCRIPTION")
        if not expected:
            errors.append(f"{image}: no DESCRIPTION file")
        elif description != expected:
            errors.append(f"{image}: description differs from DESCRIPTION\n    README:      {description}\n    DESCRIPTION: {expected}")

    if errors:
        print(f"README.md does not match the image folders:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    print(f"README.md lists all {len(images)} images, under their CATEGORY, with their DESCRIPTION.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
