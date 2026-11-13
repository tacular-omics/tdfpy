"""Fail if a docstring's Google sections did not render in the built docs site.

``mkdocs build --strict`` does not check this. It exits 0 when a directive's
options, a handler setting or a malformed docstring stop griffe from parsing
``Args:`` / ``Returns:`` / ``Raises:``, and the parameter, return and exception
tables silently disappear from the page while the build stays green. That is
how ``docs/utilities.md`` carried a ``docstring_style: numpy`` override for
months (removed in #79): the override was load-bearing, and dropping it alone
would have deleted the slicer's three tables with no failing check anywhere.

What this does: for every ``::: tdfpy.X`` directive in ``docs/``, read the
object's docstring from the installed package, note which Google sections it
declares, and require the matching rendered section titles inside that
object's block of the built HTML page.

Known limit: for a class, a section title anywhere in its block counts,
including one belonging to a method. Functions, where the block is the object
itself, carry the check.

Usage::

    uv run --group docs mkdocs build
    uv run python scripts/check_rendered_docstrings.py site
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

from build_llms_full import DOCS_DIR, REPO_ROOT, resolve

DIRECTIVE = re.compile(r"^::: (\S+)\s*$", re.M)

#: Google section name in a docstring -> the title mkdocstrings renders for it.
#: Matched as the rendered ``doc-section-title`` span and not as bare text,
#: because ``show_source: true`` embeds the docstring itself in the page: a
#: plain substring search for "Returns:" is satisfied by the source listing
#: even when the section did not render. Measured, so this is not a guess.
SECTION_TITLES = {
    "Args": "Parameters:",
    "Returns": "Returns:",
    "Raises": "Raises:",
    "Attributes": "Attributes:",
    "Yields": "Yields:",
}
SECTION_HEADING = re.compile(rf"^[ \t]*({'|'.join(SECTION_TITLES)}):[ \t]*$", re.M)


def built_page(md_path: Path, site: Path) -> Path:
    """Return the HTML file mkdocs writes for a docs page (directory URLs)."""
    rel = md_path.relative_to(DOCS_DIR).with_suffix("")
    return site / "index.html" if rel.name == "index" and rel.parent == Path() else site / rel / "index.html"


def declared_sections(obj: object) -> list[str]:
    """Google section names the object's own docstring declares, in order."""
    doc = inspect.getdoc(obj) or ""
    return [m.group(1) for m in SECTION_HEADING.finditer(doc)]


def blocks(html: str, targets: list[str]) -> dict[str, str]:
    """Split a rendered page into one block per directive target."""
    spans = []
    for target in targets:
        match = re.search(rf'id="{re.escape(target)}"', html)
        if match is not None:
            spans.append((match.start(), target))
    spans.sort()
    out = {}
    for i, (start, target) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(html)
        out[target] = html[start:end]
    return out


def main(argv: list[str]) -> int:
    site = Path(argv[1]) if len(argv) > 1 else REPO_ROOT / "site"
    if not site.is_dir():
        print(f"error: no built site at {site}; run mkdocs build first", file=sys.stderr)
        return 2

    pages = checked_objects = checked_sections = 0
    failures: list[str] = []
    for md_path in sorted(DOCS_DIR.rglob("*.md")):
        targets = DIRECTIVE.findall(md_path.read_text(encoding="utf-8"))
        if not targets:
            continue
        pages += 1
        html_path = built_page(md_path, site)
        if not html_path.is_file():
            failures.append(f"{md_path.relative_to(REPO_ROOT)}: no built page at {html_path}")
            continue
        found = blocks(html_path.read_text(encoding="utf-8"), targets)
        for target in targets:
            sections = declared_sections(resolve(target))
            if not sections:
                continue
            checked_objects += 1
            block = found.get(target)
            if block is None:
                failures.append(f"{target}: no rendered block in {html_path.relative_to(REPO_ROOT)}")
                continue
            for name in sections:
                checked_sections += 1
                title = SECTION_TITLES[name]
                if f'class="doc-section-title">{title}</span>' not in block:
                    failures.append(f"{target}: docstring declares '{name}:' but the page has no '{title}' section")

    print(f"checked {checked_sections} rendered sections on {checked_objects} objects across {pages} pages")
    if not checked_sections:
        print("error: nothing was checked, so this check proves nothing", file=sys.stderr)
        return 2
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
