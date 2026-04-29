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


class TestRequireCudaAndVram:
    def test_no_cuda_raises_clear_error(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with pytest.raises(RuntimeError, match="CUDA"):
                mimo_asr.require_cuda_and_vram(min_gb=20)

    def test_insufficient_vram_raises_clear_error(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.name = "NVIDIA GeForce RTX 4080"
        props.total_memory = 16 * 1024**3  # 16 GB
        fake_torch.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with pytest.raises(RuntimeError, match=r"≥20|at least 20|20 GB"):
                mimo_asr.require_cuda_and_vram(min_gb=20)

    def test_sufficient_vram_passes(self):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.name = "NVIDIA A100-SXM4-40GB"
        props.total_memory = 40 * 1024**3  # 40 GB
        fake_torch.cuda.get_device_properties.return_value = props
        with patch.dict(sys.modules, {"torch": fake_torch}):
            mimo_asr.require_cuda_and_vram(min_gb=20)  # must not raise


class TestRequireMimoInstalled:
    def test_missing_repo_raises(self, tmp_path):
        weights = tmp_path / "hf"
        weights.mkdir()
        repo = tmp_path / "mimo"  # does NOT exist
        with pytest.raises(RuntimeError, match=r"INSTALL_MIMO|setup_env\.sh"):
            mimo_asr.require_mimo_installed(str(weights), str(repo))

    def test_missing_weights_raises(self, tmp_path):
        weights = tmp_path / "hf"
        weights.mkdir()
        repo = tmp_path / "mimo"
        repo.mkdir()
        (repo / "src").mkdir()  # looks like a clone

        from huggingface_hub.errors import LocalEntryNotFoundError
        fake_hf = MagicMock()
        fake_hf.snapshot_download.side_effect = LocalEntryNotFoundError("not cached")
        fake_errs = MagicMock()
        fake_errs.LocalEntryNotFoundError = LocalEntryNotFoundError
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf,
                                      "huggingface_hub.errors": fake_errs}):
            with pytest.raises(RuntimeError, match=r"MiMo weights not found"):
                mimo_asr.require_mimo_installed(str(weights), str(repo))

    def test_everything_present_passes(self, tmp_path):
        weights = tmp_path / "hf"
        weights.mkdir()
        repo = tmp_path / "mimo"
        repo.mkdir()
        (repo / "src").mkdir()

        fake_hf = MagicMock()
        fake_hf.snapshot_download.return_value = str(tmp_path / "snap")
        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            mimo_asr.require_mimo_installed(str(weights), str(repo))  # no raise
        # Called once per repo id
        assert fake_hf.snapshot_download.call_count == 2


class TestPartialState:
    def test_audio_hash_is_stable_for_same_bytes(self, tmp_path):
        p = tmp_path / "a.flac"
        p.write_bytes(b"hello" * 1000)
        h1 = mimo_asr.compute_audio_hash(str(p))
        h2 = mimo_asr.compute_audio_hash(str(p))
        assert h1 == h2
        assert h1.startswith("sha256:")
        assert len(h1) == 7 + 64

    def test_audio_hash_differs_for_different_bytes(self, tmp_path):
        p1 = tmp_path / "a.flac"
        p2 = tmp_path / "b.flac"
        p1.write_bytes(b"hello")
        p2.write_bytes(b"world")
        assert mimo_asr.compute_audio_hash(str(p1)) != mimo_asr.compute_audio_hash(str(p2))

    def test_save_load_partial_roundtrip(self, tmp_path):
        partial = tmp_path / "pod_mimo_partial.json"
        mimo_asr.save_partial(
            partial,
            audio_hash="sha256:abc",
            audio_tag="<chinese>",
            weights_path="/mnt/hf",
            vad_segments=[[0, 1000], [2000, 3000]],
            completed=[{"idx": 0, "text": "hi", "start_ms": 0, "end_ms": 1000}],
            failed_at={"idx": 1, "start_ms": 2000, "error": "CUDA OOM"},
        )
        state = mimo_asr.load_partial(partial, audio_hash="sha256:abc",
                                      audio_tag="<chinese>")
        assert state["vad_segments"] == [[0, 1000], [2000, 3000]]
        assert state["completed"][0]["text"] == "hi"
        assert state["failed_at"]["idx"] == 1

    def test_load_partial_hash_mismatch_raises(self, tmp_path):
        partial = tmp_path / "pod_mimo_partial.json"
        mimo_asr.save_partial(partial, "sha256:OLD", "<chinese>", "/mnt/hf",
                              [[0, 100]], [], {"idx": 0, "error": "x"})
        with pytest.raises(RuntimeError, match=r"audio file changed"):
            mimo_asr.load_partial(partial, "sha256:NEW", "<chinese>")

    def test_load_partial_tag_mismatch_raises(self, tmp_path):
        partial = tmp_path / "pod_mimo_partial.json"
        mimo_asr.save_partial(partial, "sha256:X", "<chinese>", "/mnt/hf",
                              [[0, 100]], [], {"idx": 0, "error": "x"})
        with pytest.raises(RuntimeError, match=r"audio_tag"):
            mimo_asr.load_partial(partial, "sha256:X", "<english>")


class TestInferWithRetry:
    def test_first_attempt_success(self):
        mimo = MagicMock()
        mimo.asr_sft.return_value = "hello world"
        text = mimo_asr.infer_with_retry(mimo, "/tmp/a.wav", "<chinese>",
                                         max_retries=3, backoffs=[0.0, 0.0, 0.0])
        assert text == "hello world"
        assert mimo.asr_sft.call_count == 1

    def test_retry_then_success(self):
        mimo = MagicMock()
        mimo.asr_sft.side_effect = [
            RuntimeError("CUDA OOM"),
            RuntimeError("CUDA OOM"),
            "clean text",
        ]
        with patch.object(mimo_asr, "_cuda_cleanup") as cleanup:
            text = mimo_asr.infer_with_retry(mimo, "/tmp/a.wav", "<chinese>",
                                             max_retries=3, backoffs=[0.0, 0.0, 0.0])
        assert text == "clean text"
        assert mimo.asr_sft.call_count == 3
        assert cleanup.call_count == 2  # called before each retry

    def test_all_attempts_fail_raises(self):
        mimo = MagicMock()
        mimo.asr_sft.side_effect = RuntimeError("CUDA OOM")
        with patch.object(mimo_asr, "_cuda_cleanup"):
            with pytest.raises(RuntimeError, match=r"after 3 retries"):
                mimo_asr.infer_with_retry(mimo, "/tmp/a.wav", "<chinese>",
                                          max_retries=3, backoffs=[0.0, 0.0, 0.0])
        assert mimo.asr_sft.call_count == 3
