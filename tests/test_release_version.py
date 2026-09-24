"""scripts/release_version.py: ``sync --set`` keeps CITATION.cff's release date current."""

import runpy
import shutil
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = runpy.run_path(str(ROOT / "scripts" / "release_version.py"))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name in ("pyproject.toml", "CITATION.cff", "CHANGELOG.md", "src/tdfpy/__init__.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    return tmp_path


def test_set_version_updates_date_released(repo: Path) -> None:
    SCRIPT["sync"](repo, "99.0.0")
    citation = (repo / "CITATION.cff").read_text(encoding="utf-8")
    assert 'version: "99.0.0"' in citation
    assert f'date-released: "{date.today().isoformat()}"' in citation
    assert citation.count("date-released:") == 1


def test_set_version_adds_missing_date_released(repo: Path) -> None:
    path = repo / "CITATION.cff"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(line for line in lines if not line.startswith("date-released:")), encoding="utf-8")
    SCRIPT["sync"](repo, "99.0.0")
    assert f'date-released: "{date.today().isoformat()}"' in path.read_text(encoding="utf-8")


def test_plain_sync_keeps_date_released(repo: Path) -> None:
    before = [line for line in (repo / "CITATION.cff").read_text(encoding="utf-8").splitlines() if line.startswith("date-released:")]
    SCRIPT["sync"](repo)
    after = [line for line in (repo / "CITATION.cff").read_text(encoding="utf-8").splitlines() if line.startswith("date-released:")]
    assert after == before
