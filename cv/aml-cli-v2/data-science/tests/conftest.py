import os
import sys
import logging
import pytest
import tempfile
from unittest.mock import Mock

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

if SRC_ROOT not in sys.path:
    logging.info(f"Adding {SRC_ROOT} to path")
    sys.path.append(str(SRC_ROOT))


@pytest.fixture()
def temporary_dir():
    """Creates a temporary directory for the tests"""
    temp_directory = tempfile.TemporaryDirectory()
    yield temp_directory.name
    temp_directory.cleanup()
