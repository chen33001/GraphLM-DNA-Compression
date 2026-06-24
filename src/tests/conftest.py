"""Test fixtures that avoid pytest's shared Windows temporary directory."""

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Provide an isolated directory without using pytest-of-<user>."""
    with TemporaryDirectory(prefix="graphlm-dna-") as directory:
        yield Path(directory)
