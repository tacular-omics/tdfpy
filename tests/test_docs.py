"""Test code examples in docs/getting-started.md and docs/analysis.md using pytest-examples."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

# Resolve everything from this file so the tests pass from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"
D_PATH = str(DATA_DIR / "example_dda.d")
DIA_D_PATH = str(DATA_DIR / "example_dia.d")
PRM_D_PATH = str(DATA_DIR / "example_prm.d")
DOC_PAGES = [REPO_ROOT / "docs" / "getting-started.md", REPO_ROOT / "docs" / "analysis.md"]


def _example_id(example: CodeExample) -> str:
    return f"{example.path.relative_to(REPO_ROOT).as_posix()}:{example.start_line}-{example.end_line}"


# Examples that sweep every precursor in the DDA file (~2 s each) run with --run-slow.
_SLOW_EXAMPLE_MARKERS = ("iter_precursor_spectra(reader.precursors)",)


def _example_param(example: CodeExample):
    marks = [pytest.mark.slow] if any(m in example.source for m in _SLOW_EXAMPLE_MARKERS) else []
    return pytest.param(example, id=_example_id(example), marks=marks)


@pytest.mark.parametrize("example", [_example_param(example) for page in DOC_PAGES for example in find_examples(page)])
def test_getting_started(example: CodeExample, eval_example: EvalExample) -> None:
    if "from tdfpy import PRM" in example.source:
        d_path = PRM_D_PATH
    elif "from tdfpy import DIA" in example.source:
        d_path = DIA_D_PATH
    else:
        d_path = D_PATH
    eval_example.run(example, module_globals={"D_PATH": d_path})


@pytest.mark.slow  # pytest subprocess
def test_docs_collect_outside_repo_root(tmp_path: Path) -> None:
    """Collecting this module must not depend on the working directory."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(Path(__file__).resolve())],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_getting_started[docs/getting-started.md:" in result.stdout


#: Public names in ``tdfpy.__all__`` that no docs page has a ``:::`` directive for
#: yet, each with the work that adds it. Both halves of the test below fail: a name
#: missing from the docs and not listed here, and a name listed here that has since
#: been documented. The second half is what stops this list from going stale.
PENDING_DIRECTIVES: dict[str, str] = {}


def test_every_public_name_is_on_a_docs_page() -> None:
    """Every ``__all__`` name is rendered somewhere, or is listed as pending.

    ``llms-full.txt`` already lists the uncovered names in its trailing section, but
    listing is not failing, which is how five public names went undocumented. The rule
    for what counts as covered comes from the generator, so there is only one of it.
    """
    generator = runpy.run_path(str(REPO_ROOT / "scripts" / "build_llms_full.py"))
    uncovered = set(generator["uncovered_names"]())

    undocumented = sorted(uncovered - set(PENDING_DIRECTIVES))
    assert not undocumented, f"no ::: directive and not pending: {undocumented}"

    now_documented = sorted(set(PENDING_DIRECTIVES) - uncovered)
    assert not now_documented, f"documented now, remove from PENDING_DIRECTIVES: {now_documented}"
