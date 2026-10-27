"""Run pydocstyle as an ament test."""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """Check Python docstrings."""
    return_code = main(argv=['.', 'test'])
    assert return_code == 0, 'Found code style errors / warnings'
