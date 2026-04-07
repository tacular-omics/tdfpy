"""Test code examples in docs/getting-started.md using pytest-examples."""

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

D_PATH = "tests/data/200ngHeLaPASEF_1min.d"
DIA_D_PATH = "tests/data/example_dia.d"


@pytest.mark.parametrize("example", find_examples("docs/getting-started.md"), ids=str)
def test_getting_started(example: CodeExample, eval_example: EvalExample) -> None:
    d_path = DIA_D_PATH if "from tdfpy import DIA" in example.source else D_PATH
    eval_example.run(example, module_globals={"D_PATH": d_path})
