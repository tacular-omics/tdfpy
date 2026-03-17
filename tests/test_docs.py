"""Test code examples in docs/getting-started.md using pytest-examples."""

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

D_PATH = "tests/data/200ngHeLaPASEF_1min.d"


@pytest.mark.parametrize("example", find_examples("docs/getting-started.md"), ids=str)
def test_getting_started(example: CodeExample, eval_example: EvalExample) -> None:
    if "from tdfpy import DIA" in example.source:
        pytest.skip("No DIA test data available")
    eval_example.run(example, module_globals={"D_PATH": D_PATH})
