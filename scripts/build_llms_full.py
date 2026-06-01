"""Build docs/llms-full.txt by concatenating the key docs pages.

Produces a single plaintext file that an LLM agent can load in one shot to
ground its tdfpy code generation. Re-run this script whenever the docs
change; the output is committed to the repo so it ships with the deployed
docs site.

Usage:
    python scripts/build_llms_full.py
"""

from __future__ import annotations

from pathlib import Path

# Ordered list of docs to include, relative to the docs/ directory.
DOCS_ORDER = [
    "index.md",
    "getting-started.md",
    "utilities.md",
    "api/readers.md",
    "api/frames.md",
    "api/precursor.md",
    "api/windows.md",
    "api/prm.md",
    "api/metadata.md",
    "api/lookup.md",
    "api/centroiding.md",
    "api/pipeline.md",
    "api/noise.md",
    "api/regions.md",
    "api/low-level.md",
]

HEADER = """\
# tdfpy — Full Documentation Bundle

This file concatenates the tdfpy docs into a single plaintext bundle for
one-shot loading into an LLM context. It is generated from the MkDocs
sources by ``scripts/build_llms_full.py`` and committed to the repo so it
ships with the deployed docs site at /llms-full.txt.

For a curated index instead of the full text, see /llms.txt.

"""


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    out_path = docs_dir / "llms-full.txt"

    parts: list[str] = [HEADER]
    for rel_path in DOCS_ORDER:
        src = docs_dir / rel_path
        if not src.exists():
            print(f"  skip (missing): {rel_path}")
            continue
        parts.append(f"\n{'=' * 78}\n")
        parts.append(f"# {rel_path}\n")
        parts.append(f"{'=' * 78}\n\n")
        parts.append(src.read_text())
        parts.append("\n")
        print(f"  added: {rel_path}")

    out_path.write_text("".join(parts))
    size_kb = out_path.stat().st_size / 1024
    print(f"\nWrote {out_path.relative_to(repo_root)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
