"""Pytest path setup: make the `endothelial_simulation` package importable when
tests are collected from anywhere (there is no installed package / setup.py)."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Headless plotting for the paper-path integration test.
import matplotlib  # noqa: E402
matplotlib.use('Agg')
