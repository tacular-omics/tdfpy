"""Test code examples in docs/getting-started.md and docs/analysis.md using pytest-examples."""

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
