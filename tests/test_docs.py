"""Test code examples in docs/getting-started.md using pytest-examples."""

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

D_PATH = "tests/data/example_dda.d"
DIA_D_PATH = "tests/data/example_dia.d"
PRM_D_PATH = "tests/data/example_prm.d"


@pytest.mark.parametrize(
    "example",
    list(find_examples("docs/getting-started.md"))
    + list(find_examples("docs/analysis.md")),
    ids=str,
)
def test_getting_started(example: CodeExample, eval_example: EvalExample) -> None:
    if "from tdfpy import PRM" in example.source:
        d_path = PRM_D_PATH
    elif "from tdfpy import DIA" in example.source:
        d_path = DIA_D_PATH
    else:
        d_path = D_PATH
    eval_example.run(example, module_globals={"D_PATH": d_path})
