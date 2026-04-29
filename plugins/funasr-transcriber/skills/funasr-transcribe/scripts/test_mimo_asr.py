#!/usr/bin/env python3
"""Tests for mimo_asr module. All CUDA / MiMo / HF calls are mocked.

Safe for CI: does not require a GPU, MiMo weights, or network access.
"""

import json
import subprocess
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


class TestVadAndExtract:
    def test_run_fsmn_vad_parses_intervals(self):
        # FunASR VAD output shape: [{"value": [[start_ms, end_ms], ...], "key": "..."}]
        fake_autom = MagicMock()
        fake_model = MagicMock()
        fake_model.generate.return_value = [{
            "value": [[0, 1200], [1800, 5200], [5300, 8000]],
            "key": "clip",
        }]
        fake_autom.AutoModel.return_value = fake_model
        with patch.dict(sys.modules, {"funasr": fake_autom}):
            segs = mimo_asr.run_fsmn_vad("/tmp/x.flac",
                                         model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                         device="cpu")
        assert segs == [(0, 1200), (1800, 5200), (5300, 8000)]

    def test_extract_segment_invokes_ffmpeg(self, tmp_path):
        src = tmp_path / "in.flac"
        src.write_bytes(b"fake")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        calls = []
        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # ffmpeg creates the output file
            out_idx = cmd.index("-y") + 1 if "-y" in cmd else -1
            Path(cmd[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with patch("subprocess.run", side_effect=fake_run):
            path = mimo_asr.extract_segment(str(src), 1000, 4200, str(out_dir))
        assert Path(path).exists()
        assert Path(path).name.endswith(".wav")
        # Verify ffmpeg was called with -ss 1.0 -to 4.2 on the source file
        cmd = calls[0]
        assert "ffmpeg" in cmd[0]
        assert "-ss" in cmd and "1.000" in cmd
        assert "-to" in cmd and "4.200" in cmd
        assert str(src) in cmd


class TestAssignSpeakersViaCam:
    def test_assigns_speaker_ids_from_embeddings(self, tmp_path):
        # 4 segments; embeddings designed so KMeans(k=2) splits {0,2} vs {1,3}
        segments = [
            {"idx": 0, "text": "a", "start_ms": 0,     "end_ms": 1000},
            {"idx": 1, "text": "b", "start_ms": 1500,  "end_ms": 2500},
            {"idx": 2, "text": "c", "start_ms": 3000,  "end_ms": 4000},
            {"idx": 3, "text": "d", "start_ms": 4500,  "end_ms": 5500},
        ]

        import numpy as np
        embeddings = {
            (0, 1000):     np.array([1.0, 0.0], dtype=np.float32),
            (1500, 2500):  np.array([0.0, 1.0], dtype=np.float32),
            (3000, 4000):  np.array([1.0, 0.1], dtype=np.float32),
            (4500, 5500):  np.array([0.0, 0.9], dtype=np.float32),
        }

        def fake_embed(start_ms, end_ms, *a, **kw):
            return embeddings[(start_ms, end_ms)]

        fake_sf = MagicMock()
        # sf.read returns (audio_data, sample_rate); 10s of silence at 16kHz
        fake_sf.read.return_value = (np.zeros(160000, dtype=np.float32), 16000)
        fake_funasr = MagicMock()
        # AutoModel call returns a mock model; _extract_speaker_embedding is patched anyway
        with patch.dict(sys.modules, {"soundfile": fake_sf, "funasr": fake_funasr}), \
             patch.object(mimo_asr, "_extract_speaker_embedding",
                          side_effect=fake_embed):
            out = mimo_asr.assign_speakers_via_cam(
                segments, "/tmp/fake.flac",
                num_speakers=2, spk_model_id="iic/x", device="cpu",
            )
        assert [s["speaker"] for s in out] == [out[0]["speaker"], out[1]["speaker"],
                                                out[0]["speaker"], out[1]["speaker"]]
        assert out[0]["speaker"] != out[1]["speaker"]


class TestTranscribeWithMimo:
    def _mock_heavy_deps(self, vad_segments, asr_outputs):
        """Return a context manager stack that mocks every heavy dep.

        asr_outputs: list of str or Exception instances (one per call to asr_sft).
        """
        fake_mimo_module = MagicMock()
        fake_mimo_instance = MagicMock()
        # each asr_sft call pops from asr_outputs
        def asr_sft(wav, audio_tag):
            val = asr_outputs.pop(0)
            if isinstance(val, Exception):
                raise val
            return val
        fake_mimo_instance.asr_sft.side_effect = asr_sft
        fake_mimo_module.MimoAudio.return_value = fake_mimo_instance
        return fake_mimo_module, fake_mimo_instance

    def test_happy_path(self, tmp_path):
        audio = tmp_path / "pod.flac"
        audio.write_bytes(b"fake")
        vad = [(0, 1000), (1500, 3000), (3500, 5000)]
        mimo_mod, mimo_inst = self._mock_heavy_deps(vad, ["hello", "world", "!"])

        with patch.object(mimo_asr, "require_cuda_and_vram"), \
             patch.object(mimo_asr, "require_mimo_installed"), \
             patch.object(mimo_asr, "run_fsmn_vad", return_value=vad), \
             patch.object(mimo_asr, "extract_segment",
                          side_effect=lambda *a, **k: str(tmp_path / "seg.wav")), \
             patch.object(mimo_asr, "_load_mimo", return_value=mimo_inst), \
             patch.object(mimo_asr, "assign_speakers_via_cam",
                          side_effect=lambda segs, *a, **k: [
                              {**s, "speaker": i % 2} for i, s in enumerate(segs)
                          ]), \
             patch.object(mimo_asr, "_cuda_cleanup"):
            out = mimo_asr.transcribe_with_mimo(
                str(audio), num_speakers=2, audio_tag="<chinese>",
                weights_path=str(tmp_path),
            )
        assert len(out) == 3
        assert out[0]["text"] == "hello"
        assert out[1]["text"] == "world"
        assert out[0]["start_ms"] == 0
        assert out[0]["end_ms"] == 1000
        assert out[0]["speaker"] == 0
        assert out[1]["speaker"] == 1
        # Partial file should NOT exist on success
        partial = audio.parent / f"{audio.stem}_mimo_partial.json"
        assert not partial.exists()

    def test_failure_writes_partial_and_raises(self, tmp_path):
        audio = tmp_path / "pod.flac"
        audio.write_bytes(b"fake")
        vad = [(0, 1000), (1500, 3000), (3500, 5000)]
        # Segment 0 ok, segment 1 fails 3x
        outputs = ["ok", RuntimeError("OOM"), RuntimeError("OOM"), RuntimeError("OOM")]
        mimo_mod, mimo_inst = self._mock_heavy_deps(vad, outputs)

        with patch.object(mimo_asr, "require_cuda_and_vram"), \
             patch.object(mimo_asr, "require_mimo_installed"), \
             patch.object(mimo_asr, "run_fsmn_vad", return_value=vad), \
             patch.object(mimo_asr, "extract_segment",
                          side_effect=lambda *a, **k: str(tmp_path / "seg.wav")), \
             patch.object(mimo_asr, "_load_mimo", return_value=mimo_inst), \
             patch.object(mimo_asr, "_cuda_cleanup"):
            with pytest.raises(RuntimeError, match=r"segment 1"):
                mimo_asr.transcribe_with_mimo(
                    str(audio), num_speakers=2, audio_tag="<chinese>",
                    weights_path=str(tmp_path),
                    backoffs=[0.0, 0.0, 0.0],
                )
        partial = audio.parent / f"{audio.stem}_mimo_partial.json"
        assert partial.exists()
        state = json.loads(partial.read_text(encoding="utf-8"))
        assert state["failed_at"]["idx"] == 1
        assert [c["idx"] for c in state["completed"]] == [0]
        assert state["completed"][0]["text"] == "ok"

    def test_resume_skips_completed(self, tmp_path):
        audio = tmp_path / "pod.flac"
        audio.write_bytes(b"fake")
        vad = [(0, 1000), (1500, 3000), (3500, 5000)]

        # Pre-populate partial: segment 0 done, failed at 1
        partial = audio.parent / f"{audio.stem}_mimo_partial.json"
        audio_hash = mimo_asr.compute_audio_hash(str(audio))
        mimo_asr.save_partial(
            partial, audio_hash=audio_hash, audio_tag="<chinese>",
            weights_path=str(tmp_path), vad_segments=[list(s) for s in vad],
            completed=[{"idx": 0, "text": "ok", "start_ms": 0, "end_ms": 1000}],
            failed_at={"idx": 1, "start_ms": 1500, "error": "OOM"},
        )

        # Resume: only segments 1 and 2 should be called
        outputs = ["resumed", "done"]
        mimo_mod, mimo_inst = self._mock_heavy_deps(vad, outputs)

        with patch.object(mimo_asr, "require_cuda_and_vram"), \
             patch.object(mimo_asr, "require_mimo_installed"), \
             patch.object(mimo_asr, "run_fsmn_vad",
                          side_effect=AssertionError("VAD must not run on resume")), \
             patch.object(mimo_asr, "extract_segment",
                          side_effect=lambda *a, **k: str(tmp_path / "seg.wav")), \
             patch.object(mimo_asr, "_load_mimo", return_value=mimo_inst), \
             patch.object(mimo_asr, "assign_speakers_via_cam",
                          side_effect=lambda segs, *a, **k: [
                              {**s, "speaker": 0} for s in segs
                          ]), \
             patch.object(mimo_asr, "_cuda_cleanup"):
            out = mimo_asr.transcribe_with_mimo(
                str(audio), num_speakers=2, audio_tag="<chinese>",
                weights_path=str(tmp_path), resume=True,
            )
        assert len(out) == 3
        assert out[0]["text"] == "ok"       # from partial
        assert out[1]["text"] == "resumed"  # from resume
        assert out[2]["text"] == "done"
        assert mimo_inst.asr_sft.call_count == 2  # only 2 new calls
        assert not partial.exists()  # cleaned on success


class TestCliWiring:
    def test_lang_mimo_in_supported(self):
        import transcribe_funasr as tf
        assert "mimo" in tf.SUPPORTED_LANGS
        assert "mimo" in tf.MODEL_PRESETS

    def test_hotwords_warning_with_mimo(self, capsys):
        import transcribe_funasr as tf
        resolved = tf.warn_on_incompatible_flags(
            lang="mimo", hotwords="foo bar", batch_size=300, default_batch=300,
        )
        captured = capsys.readouterr()
        assert resolved["hotwords"] is None
        assert "hotwords" in captured.out.lower()

    def test_batch_size_warning_with_mimo(self, capsys):
        import transcribe_funasr as tf
        tf.warn_on_incompatible_flags(
            lang="mimo", hotwords=None, batch_size=100, default_batch=300,
        )
        captured = capsys.readouterr()
        assert "batch-size" in captured.out.lower() or "batch_size" in captured.out.lower()

    def test_no_warning_with_zh(self, capsys):
        import transcribe_funasr as tf
        tf.warn_on_incompatible_flags(
            lang="zh", hotwords="foo", batch_size=300, default_batch=300,
        )
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_weights_path_precedence(self, monkeypatch):
        import transcribe_funasr as tf
        monkeypatch.setenv("HF_HOME", "/env/hf")
        assert tf.resolve_mimo_weights_path("/cli/hf") == "/cli/hf"
        assert tf.resolve_mimo_weights_path(None) == "/env/hf"
        monkeypatch.delenv("HF_HOME", raising=False)
        assert tf.resolve_mimo_weights_path(None) == str(
            Path.home() / ".cache" / "huggingface"
        )
