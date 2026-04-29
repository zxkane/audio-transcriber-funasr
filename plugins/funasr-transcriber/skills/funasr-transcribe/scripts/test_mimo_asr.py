#!/usr/bin/env python3
"""Tests for mimo_asr module. All CUDA / MiMo / HF calls are mocked.

Safe for CI: does not require a GPU, MiMo weights, or network access.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import mimo_asr


def test_module_imports():
    """mimo_asr module exists and exposes expected public names."""
    assert hasattr(mimo_asr, "require_cuda_and_vram")
    assert hasattr(mimo_asr, "require_mimo_installed")
    assert hasattr(mimo_asr, "transcribe_with_mimo")
    assert hasattr(mimo_asr, "save_partial")
    assert hasattr(mimo_asr, "load_partial")
