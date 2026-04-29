# MiMo-V2.5-ASR Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--lang mimo` preset that runs `XiaomiMiMo/MiMo-V2.5-ASR` locally on a CUDA GPU, reusing FSMN VAD + CAM++ diarization so output format matches `--lang zh` exactly.

**Architecture:** New `mimo_asr.py` module encapsulates GPU pre-checks, MiMo repo/weights checks, per-VAD-segment inference with retry, and `*_mimo_partial.json` resume state. `transcribe_funasr.py` adds `--lang mimo` to its dispatch table, calling `mimo_asr.transcribe_with_mimo()` which returns a `sentence_info`-shaped list that Phase 2 / Phase 3 consume unchanged. `setup_env.sh` is upgraded to require Python 3.12; `setup_mimo.sh` (new, opt-in via `INSTALL_MIMO=1`) clones the MiMo GitHub repo into `.venv/mimo/`, installs `flash-attn`, and downloads weights to `$MIMO_WEIGHTS_PATH`.

**Tech Stack:** Python 3.12, PyTorch, FunASR (existing), `huggingface_hub`, `transformers`, `flash-attn==2.7.4.post1`, pytest, bash.

**Spec:** [docs/superpowers/specs/2026-04-29-mimo-asr-integration-design.md](../specs/2026-04-29-mimo-asr-integration-design.md)

**Key paths:**
- `$SKILL` = `plugins/funasr-transcriber/skills/funasr-transcribe`
- `$SCRIPTS` = `$SKILL/scripts`
- Venv dir: `.venv` (not `venv`) — current `setup_env.sh` convention
- MiMo repo clone: `.venv/mimo/`
- Weights: `$MIMO_WEIGHTS_PATH` → `$HF_HOME` → `~/.cache/huggingface`

---

## Task 1: Scaffold `mimo_asr.py` module + test file

**Files:**
- Create: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Create: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

This task establishes the module boundary and the test harness. No logic yet — just a placeholder module and a single smoke test that imports it.

- [ ] **Step 1.1: Write failing smoke test**

Write `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`:

```python
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
```

- [ ] **Step 1.2: Run test — expect import failure**

Run:
```bash
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 -m pytest test_mimo_asr.py::test_module_imports -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'mimo_asr'`.

- [ ] **Step 1.3: Create skeleton `mimo_asr.py`**

Write `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`:

```python
#!/usr/bin/env python3
"""MiMo-V2.5-ASR local inference integration for funasr-transcribe.

Runs XiaomiMiMo/MiMo-V2.5-ASR on a local CUDA GPU, reusing the existing
pipeline's FSMN VAD segmentation and CAM++ speaker clustering so the output
format matches --lang zh exactly.

Public entry points:
  - require_cuda_and_vram(min_gb): pre-flight GPU capacity check
  - require_mimo_installed(weights_path, repo_path): pre-flight install check
  - transcribe_with_mimo(audio_path, num_speakers, ...): full Phase 1 path
  - save_partial / load_partial: resume state management
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence


def require_cuda_and_vram(min_gb: int = 20) -> None:
    """Pre-flight: require a CUDA device with at least min_gb VRAM. Not implemented yet."""
    raise NotImplementedError


def require_mimo_installed(weights_path: str, repo_path: str) -> None:
    """Pre-flight: require MiMo repo cloned and weights downloaded. Not implemented yet."""
    raise NotImplementedError


def transcribe_with_mimo(audio_path: str,
                         num_speakers: Optional[int] = None,
                         audio_tag: str = "<chinese>",
                         batch: int = 1,
                         weights_path: Optional[str] = None,
                         resume: bool = False,
                         device: str = "cuda:0",
                         spk_model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common",
                         vad_model_id: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch") -> list:
    """Phase 1 MiMo path: VAD -> MiMo ASR -> CAM++ speaker labels. Not implemented yet."""
    raise NotImplementedError


def save_partial(partial_path: Path, audio_hash: str, audio_tag: str,
                 weights_path: str, vad_segments: list, completed: list,
                 failed_at: dict) -> None:
    """Persist in-progress MiMo inference state. Not implemented yet."""
    raise NotImplementedError


def load_partial(partial_path: Path, audio_hash: str, audio_tag: str) -> dict:
    """Load resume state; raise on hash/tag mismatch. Not implemented yet."""
    raise NotImplementedError
```

- [ ] **Step 1.4: Run test — expect pass**

Run:
```bash
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 -m pytest test_mimo_asr.py::test_module_imports -v
```
Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): scaffold mimo_asr module with smoke test"
```

---

## Task 2: GPU pre-flight — `require_cuda_and_vram`

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py` (replace `require_cuda_and_vram`)
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py` (add 3 tests)

The check must fail with clear messages in two distinct scenarios (no CUDA, insufficient VRAM) and pass silently when a suitable GPU is present.

- [ ] **Step 2.1: Write three failing tests**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 2.2: Run tests — expect 3 FAIL (NotImplementedError)**

Run:
```bash
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 -m pytest test_mimo_asr.py::TestRequireCudaAndVram -v
```
Expected: 3 FAIL.

- [ ] **Step 2.3: Implement the function**

In `mimo_asr.py`, replace the `require_cuda_and_vram` stub with:

```python
def require_cuda_and_vram(min_gb: int = 20) -> None:
    """Pre-flight: require a CUDA device with at least min_gb VRAM.

    Raises:
        RuntimeError: if CUDA is unavailable or VRAM is below min_gb.
    """
    import torch  # imported lazily so test can patch sys.modules
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--lang mimo requires a CUDA GPU. CUDA is not available "
            "on this machine. Use --lang zh for CPU."
        )
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    if total_gb < min_gb:
        raise RuntimeError(
            f"--lang mimo requires ≥{min_gb} GB VRAM. "
            f"Detected: {props.name} ({total_gb:.1f} GB). "
            f"MiMo-V2.5-ASR is 8B params in fp16 + tokenizer + KV cache. "
            f"Use --lang zh for low-VRAM GPUs."
        )
```

- [ ] **Step 2.4: Run tests — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestRequireCudaAndVram -v
```
Expected: 3 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): add require_cuda_and_vram preflight"
```

---

## Task 3: MiMo install pre-flight — `require_mimo_installed`

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

Checks that (a) the MiMo GitHub repo is cloned at `repo_path` and (b) both HF weight repos are present in `weights_path`, using `huggingface_hub.snapshot_download(..., local_files_only=True)` as the authoritative existence probe.

- [ ] **Step 3.1: Write three failing tests**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 3.2: Run tests — expect 3 FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestRequireMimoInstalled -v
```
Expected: 3 FAIL.

- [ ] **Step 3.3: Implement**

Replace the `require_mimo_installed` stub in `mimo_asr.py`:

```python
def require_mimo_installed(weights_path: str, repo_path: str) -> None:
    """Pre-flight: require the MiMo GitHub repo cloned and HF weights downloaded.

    Raises:
        RuntimeError: with a user-facing message pointing to the install command.
    """
    repo = Path(repo_path)
    if not (repo.is_dir() and (repo / "src").is_dir()):
        raise RuntimeError(
            f"--lang mimo requires MiMo to be installed, but the MiMo repo "
            f"was not found at {repo_path}. "
            f"Run: INSTALL_MIMO=1 bash $SCRIPTS/setup_env.sh"
        )

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    for repo_id in ("XiaomiMiMo/MiMo-V2.5-ASR",
                    "XiaomiMiMo/MiMo-Audio-Tokenizer"):
        try:
            snapshot_download(repo_id, cache_dir=weights_path,
                              local_files_only=True)
        except LocalEntryNotFoundError as e:
            raise RuntimeError(
                f"MiMo weights not found at {weights_path} "
                f"(missing: {repo_id}). "
                f"Run: INSTALL_MIMO=1 MIMO_WEIGHTS_PATH={weights_path} "
                f"bash $SCRIPTS/setup_env.sh"
            ) from e
```

- [ ] **Step 3.4: Run tests — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestRequireMimoInstalled -v
```
Expected: 3 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): add require_mimo_installed preflight"
```

---

## Task 4: Audio hash + partial state I/O

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

Adds a helper to SHA256 a file (streaming, memory-safe), plus `save_partial` / `load_partial` with hash + audio_tag validation.

- [ ] **Step 4.1: Write failing tests**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 4.2: Run tests — expect 5 FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestPartialState -v
```
Expected: 5 FAIL.

- [ ] **Step 4.3: Implement**

In `mimo_asr.py`, add (new function) and replace the two stubs:

```python
def compute_audio_hash(path: str, _chunk: int = 1 << 20) -> str:
    """SHA256 of the file at path, streamed, prefixed with 'sha256:'."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(_chunk)
            if not buf:
                break
            h.update(buf)
    return f"sha256:{h.hexdigest()}"


def save_partial(partial_path: Path, audio_hash: str, audio_tag: str,
                 weights_path: str, vad_segments: list, completed: list,
                 failed_at: dict) -> None:
    """Persist MiMo inference state so --resume-mimo can continue."""
    payload = {
        "audio_hash": audio_hash,
        "audio_tag": audio_tag,
        "mimo_weights_path": weights_path,
        "vad_segments": vad_segments,
        "completed": completed,
        "failed_at": failed_at,
    }
    tmp = Path(str(partial_path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(partial_path)


def load_partial(partial_path: Path, audio_hash: str, audio_tag: str) -> dict:
    """Load resume state, verifying audio_hash and audio_tag match the current run."""
    state = json.loads(Path(partial_path).read_text(encoding="utf-8"))
    if state.get("audio_hash") != audio_hash:
        raise RuntimeError(
            f"audio file changed since partial was saved "
            f"({state.get('audio_hash')} != {audio_hash}). "
            f"Delete {partial_path} to restart."
        )
    if state.get("audio_tag") != audio_tag:
        raise RuntimeError(
            f"audio_tag changed since partial was saved "
            f"({state.get('audio_tag')} != {audio_tag}). "
            f"Use the same --mimo-audio-tag as the original run, or "
            f"delete {partial_path} to restart."
        )
    return state
```

- [ ] **Step 4.4: Run tests — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestPartialState -v
```
Expected: 5 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): add audio hash + partial state save/load"
```

---

## Task 5: Per-segment inference with retry — `infer_with_retry`

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

A pure wrapper around `mimo.asr_sft(wav, audio_tag=...)` that retries 3× with backoffs [0.5, 2.0, 5.0], clearing `gc.collect()` + `torch.cuda.empty_cache()` between attempts. Returns `str` on success, raises `RuntimeError` on final failure.

- [ ] **Step 5.1: Write failing tests**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 5.2: Run tests — expect 3 FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestInferWithRetry -v
```
Expected: 3 FAIL.

- [ ] **Step 5.3: Implement**

Add to `mimo_asr.py`:

```python
def _cuda_cleanup() -> None:
    """Best-effort VRAM defragmentation between retry attempts."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def infer_with_retry(mimo, audio_path: str, audio_tag: str,
                     max_retries: int = 3,
                     backoffs: Sequence[float] = (0.5, 2.0, 5.0)) -> str:
    """Call mimo.asr_sft with up to max_retries attempts. Raises on final failure.

    Between attempts, run gc + torch.cuda.empty_cache() to recover from
    fragmentation-driven OOMs. Re-raises the last exception wrapped with a
    clear "after N retries" message so callers can distinguish retry
    exhaustion from a single unrecoverable failure.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        if attempt > 0:
            _cuda_cleanup()
            time.sleep(backoffs[min(attempt - 1, len(backoffs) - 1)])
        try:
            return mimo.asr_sft(audio_path, audio_tag=audio_tag)
        except Exception as e:
            last_exc = e
            err_class = type(e).__name__
            print(f"    attempt {attempt + 1}/{max_retries} failed: {err_class}: {e}")
    raise RuntimeError(
        f"MiMo inference failed after {max_retries} retries: "
        f"{type(last_exc).__name__}: {last_exc}"
    ) from last_exc
```

- [ ] **Step 5.4: Run tests — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestInferWithRetry -v
```
Expected: 3 PASS.

- [ ] **Step 5.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): add infer_with_retry with backoff and cuda cleanup"
```

---

## Task 6: VAD segmentation + segment extraction

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

Two helpers: `run_fsmn_vad` wraps `funasr.AutoModel` to return `[(start_ms, end_ms), ...]`; `extract_segment` uses ffmpeg to cut a WAV chunk. These are thin wrappers; we test the data-transformation logic of `run_fsmn_vad` (parsing FunASR's response) with a mocked AutoModel, and we test `extract_segment` returns an existing file with correct naming.

- [ ] **Step 6.1: Write failing tests**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 6.2: Run tests — expect 2 FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestVadAndExtract -v
```
Expected: 2 FAIL.

- [ ] **Step 6.3: Implement**

Add to `mimo_asr.py`:

```python
def run_fsmn_vad(audio_path: str,
                 model_id: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                 device: str = "cuda:0",
                 max_single_segment_time: int = 60000) -> list:
    """Run FSMN VAD and return a list of (start_ms, end_ms) intervals."""
    from funasr import AutoModel
    model = AutoModel(
        model=model_id,
        vad_kwargs={"max_single_segment_time": max_single_segment_time},
        device=device,
        disable_update=True,
    )
    res = model.generate(input=audio_path)
    if not res or "value" not in res[0]:
        return []
    return [(int(s), int(e)) for s, e in res[0]["value"]]


def extract_segment(audio_path: str, start_ms: int, end_ms: int,
                    out_dir: str) -> str:
    """Cut [start_ms, end_ms] from audio_path into a 16kHz mono WAV in out_dir.

    Returns the output file path. Uses ffmpeg so we don't load the full
    audio into memory for each chunk.
    """
    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0
    out_path = Path(out_dir) / f"seg_{start_ms:010d}_{end_ms:010d}.wav"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", audio_path,
        "-ac", "1", "-ar", "16000", "-f", "wav",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(out_path)
```

- [ ] **Step 6.4: Run tests — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestVadAndExtract -v
```
Expected: 2 PASS.

- [ ] **Step 6.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): add run_fsmn_vad and extract_segment helpers"
```

---

## Task 7: Speaker assignment adapter — `assign_speakers_via_cam`

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

Reuses the existing CAM++ embedding logic from `transcribe_funasr.rescore_montage_speakers` but in standalone form: given a list of segments (with `start_ms`, `end_ms`, `text`), compute an embedding per segment and cluster to `num_speakers` groups. Returns the same segments with `speaker` populated (int ID).

This function is a smaller version of the existing `rescore_montage_speakers` — we cluster all segments, not just the montage prefix.

- [ ] **Step 7.1: Write failing test**

Append to `test_mimo_asr.py`:

```python
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

        with patch.object(mimo_asr, "_extract_speaker_embedding",
                          side_effect=fake_embed):
            out = mimo_asr.assign_speakers_via_cam(
                segments, "/tmp/fake.flac",
                num_speakers=2, spk_model_id="iic/x", device="cpu",
            )
        assert [s["speaker"] for s in out] == [out[0]["speaker"], out[1]["speaker"],
                                                out[0]["speaker"], out[1]["speaker"]]
        assert out[0]["speaker"] != out[1]["speaker"]
```

- [ ] **Step 7.2: Run test — expect FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestAssignSpeakersViaCam -v
```
Expected: FAIL.

- [ ] **Step 7.3: Implement**

Add to `mimo_asr.py`:

```python
def _extract_speaker_embedding(start_ms: int, end_ms: int, spk_model,
                               audio_data, sample_rate: int):
    """Extract a CAM++ embedding for one audio segment. Returns np.ndarray or None."""
    import numpy as np
    start = int(start_ms * sample_rate / 1000)
    end = int(end_ms * sample_rate / 1000)
    segment = audio_data[start:end]
    if len(segment) < sample_rate * 0.3:
        return None
    try:
        result = spk_model.generate(input=segment)
        if result and isinstance(result, list) and len(result) > 0:
            emb = result[0].get("spk_embedding")
            if emb is not None:
                return np.asarray(emb, dtype=np.float32).flatten()
    except Exception as e:
        print(f"    WARNING: embedding extraction failed at {start_ms}ms: {e}")
    return None


def assign_speakers_via_cam(segments: list, audio_path: str,
                            num_speakers: Optional[int],
                            spk_model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common",
                            device: str = "cuda:0") -> list:
    """Attach speaker IDs to segments via CAM++ embeddings + KMeans clustering.

    Falls back to speaker=0 for every segment if num_speakers is None or 1,
    or if too few segments have valid embeddings.
    """
    import numpy as np

    if num_speakers is None or num_speakers <= 1 or len(segments) < 2:
        for s in segments:
            s["speaker"] = 0
        return segments

    import soundfile as sf
    from funasr import AutoModel
    from sklearn.cluster import KMeans

    spk_model = AutoModel(model=spk_model_id, device=device, disable_update=True)
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    embeds = []
    kept_idx = []
    for i, seg in enumerate(segments):
        emb = _extract_speaker_embedding(
            seg["start_ms"], seg["end_ms"],
            spk_model, audio_data, sample_rate,
        )
        if emb is not None:
            embeds.append(emb)
            kept_idx.append(i)

    if len(embeds) < num_speakers:
        print(f"  WARNING: only {len(embeds)} segments with embeddings "
              f"(need ≥{num_speakers}); assigning speaker=0 to all.")
        for s in segments:
            s["speaker"] = 0
        return segments

    X = np.stack(embeds)
    labels = KMeans(n_clusters=num_speakers, n_init=10,
                    random_state=0).fit_predict(X)
    label_map = dict(zip(kept_idx, labels))
    for i, seg in enumerate(segments):
        seg["speaker"] = int(label_map.get(i, 0))
    return segments
```

- [ ] **Step 7.4: Run test — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestAssignSpeakersViaCam -v
```
Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): add CAM++ speaker clustering adapter"
```

---

## Task 8: Orchestrator — `transcribe_with_mimo` (happy path + failure + resume)

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

Wires together: preflights → VAD → load MiMo → loop with retry → save partial on failure → free MiMo VRAM → CAM++ → delete partial on success.

- [ ] **Step 8.1: Write failing tests**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 8.2: Run tests — expect 3 FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestTranscribeWithMimo -v
```
Expected: 3 FAIL.

- [ ] **Step 8.3: Implement**

Add to `mimo_asr.py`:

```python
def _format_time(ms: int) -> str:
    s = ms // 1000
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _load_mimo(weights_path: str):
    """Resolve snapshot dirs and instantiate MimoAudio. Isolated for test mocking."""
    from huggingface_hub import snapshot_download
    venv_root = Path(sys.prefix)
    sys.path.insert(0, str(venv_root / "mimo"))
    from src.mimo_audio.mimo_audio import MimoAudio  # type: ignore
    model_dir = snapshot_download("XiaomiMiMo/MiMo-V2.5-ASR",
                                  cache_dir=weights_path, local_files_only=True)
    tokenizer_dir = snapshot_download("XiaomiMiMo/MiMo-Audio-Tokenizer",
                                      cache_dir=weights_path, local_files_only=True)
    return MimoAudio(model_path=model_dir, tokenizer_path=tokenizer_dir)


def _free_mimo(mimo) -> None:
    """Release MiMo VRAM so CAM++ can load in the same process."""
    del mimo
    _cuda_cleanup()


def transcribe_with_mimo(audio_path: str,
                         num_speakers: Optional[int] = None,
                         audio_tag: str = "<chinese>",
                         batch: int = 1,
                         weights_path: Optional[str] = None,
                         resume: bool = False,
                         device: str = "cuda:0",
                         spk_model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common",
                         vad_model_id: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                         repo_path: Optional[str] = None,
                         backoffs: Sequence[float] = (0.5, 2.0, 5.0)) -> list:
    """Phase 1 MiMo path: VAD -> per-segment MiMo ASR -> CAM++ speaker labels.

    Returns a sentence_info-shaped list of
    {speaker: int, start_ms: int, end_ms: int, text: str}
    matching the format produced by the FunASR presets, so downstream
    Phase 2/3 code is identical regardless of --lang.
    """
    weights_path = weights_path or os.environ.get("HF_HOME") \
        or str(Path.home() / ".cache" / "huggingface")
    repo_path = repo_path or str(Path(sys.prefix) / "mimo")

    # 1. Preflights — cheap, fail fast before any load
    require_cuda_and_vram(min_gb=20)
    require_mimo_installed(weights_path, repo_path)

    # 2. Resolve state (resume or fresh VAD)
    audio_p = Path(audio_path)
    partial_path = audio_p.with_name(f"{audio_p.stem}_mimo_partial.json")
    audio_hash = compute_audio_hash(audio_path)

    if resume:
        state = load_partial(partial_path, audio_hash, audio_tag)
        vad_segments = [tuple(s) for s in state["vad_segments"]]
        completed = {c["idx"]: c for c in state["completed"]}
        start_idx = state["failed_at"]["idx"]
        print(f"  Resuming from {partial_path.name} "
              f"({len(completed)}/{len(vad_segments)} completed)")
    else:
        print(f"[Phase 1a] VAD segmentation (FSMN)...")
        vad_segments = run_fsmn_vad(audio_path, vad_model_id, device)
        print(f"  Detected {len(vad_segments)} segments")
        completed = {}
        start_idx = 0

    # 3. Load MiMo
    print(f"[Phase 1b] MiMo ASR (local, GPU)")
    print(f"  Loading MiMo from {weights_path}...")
    t_load = time.time()
    mimo = _load_mimo(weights_path)
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    # 4. Per-segment loop with retry
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(start_idx, len(vad_segments)):
                if i in completed:
                    continue
                s_ms, e_ms = vad_segments[i]
                chunk_wav = extract_segment(audio_path, s_ms, e_ms, tmpdir)
                try:
                    text = infer_with_retry(mimo, chunk_wav, audio_tag,
                                            max_retries=3, backoffs=list(backoffs))
                except RuntimeError as e:
                    # Final failure: persist partial then re-raise
                    save_partial(
                        partial_path, audio_hash, audio_tag, weights_path,
                        [list(v) for v in vad_segments],
                        [completed[k] for k in sorted(completed)],
                        failed_at={"idx": i, "start_ms": s_ms, "error": str(e)},
                    )
                    raise RuntimeError(
                        f"MiMo inference failed at segment {i}/{len(vad_segments)} "
                        f"({_format_time(s_ms)}): {e}. "
                        f"Partial saved: {partial_path.name}. "
                        f"Resume with: --resume-mimo"
                    ) from e
                completed[i] = {"idx": i, "text": text,
                                "start_ms": s_ms, "end_ms": e_ms}
                print(f"  [{i+1:3d}/{len(vad_segments)}] {_format_time(s_ms)} "
                      f"({e_ms - s_ms}ms) -> {text[:40]}")
    finally:
        _free_mimo(mimo)

    wall = time.time() - t0
    if vad_segments:
        duration = (vad_segments[-1][1] - vad_segments[0][0]) / 1000
        if duration > 0:
            print(f"  MiMo inference: {wall:.1f}s (RTF {wall / duration:.3f})")

    # 5. Speaker clustering on the same VAD segments
    print(f"[Phase 1c] CAM++ speaker clustering...")
    segments = [completed[i] for i in range(len(vad_segments))]
    segments = assign_speakers_via_cam(segments, audio_path, num_speakers,
                                       spk_model_id, device)

    # 6. Clean up partial on success
    if partial_path.exists():
        partial_path.unlink()

    return segments
```

- [ ] **Step 8.4: Run tests — expect PASS**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestTranscribeWithMimo -v
```
Expected: 3 PASS.

- [ ] **Step 8.5: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/mimo_asr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): wire transcribe_with_mimo orchestrator with resume"
```

---

## Task 9: CLI wiring in `transcribe_funasr.py`

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/transcribe_funasr.py`
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py`

Add `mimo` preset to `MODEL_PRESETS`, new CLI flags, warnings for incompatible flag combinations, dispatch to `mimo_asr.transcribe_with_mimo` when `--lang mimo`.

- [ ] **Step 9.1: Write failing tests for CLI integration**

Append to `test_mimo_asr.py`:

```python
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
```

- [ ] **Step 9.2: Run tests — expect 5 FAIL**

Run:
```bash
python3 -m pytest test_mimo_asr.py::TestCliWiring -v
```
Expected: 5 FAIL (missing preset entry, missing helpers).

- [ ] **Step 9.3: Add `mimo` preset to `MODEL_PRESETS`**

In `transcribe_funasr.py`, in the `MODEL_PRESETS` dict (ends at line ~122, just before `SUPPORTED_LANGS = list(MODEL_PRESETS.keys())`), add a new entry after `"whisper"`:

```python
    "mimo": {
        "label": "MiMo-V2.5-ASR (local 8B, GPU-only, VAD+CAM++ diarization)",
        # Placeholder IDs — mimo preset is dispatched to mimo_asr module,
        # which loads weights from HuggingFace rather than ModelScope.
        "asr": "XiaomiMiMo/MiMo-V2.5-ASR",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": None,
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
```

- [ ] **Step 9.4: Add helper functions near the top of `main`-related utilities**

Insert above the `def main():` line in `transcribe_funasr.py`:

```python
def warn_on_incompatible_flags(lang: str, hotwords, batch_size: int,
                               default_batch: int) -> dict:
    """Warn and scrub flags that don't apply to the chosen language preset.

    Returns a dict of resolved values (currently {'hotwords': ...}).
    """
    resolved = {"hotwords": hotwords}
    if lang == "mimo":
        if hotwords:
            print("  Warning: --hotwords ignored for --lang mimo "
                  "(MiMo does not support hotword biasing)")
            resolved["hotwords"] = None
        if batch_size != default_batch:
            print(f"  Warning: --batch-size ignored for --lang mimo "
                  f"(use --mimo-batch instead; got {batch_size})")
    return resolved


def resolve_mimo_weights_path(cli_value: Optional[str]) -> str:
    """CLI flag > $HF_HOME > ~/.cache/huggingface, as per the design spec."""
    if cli_value:
        return cli_value
    env = os.environ.get("HF_HOME")
    if env:
        return env
    return str(Path.home() / ".cache" / "huggingface")
```

- [ ] **Step 9.5: Add CLI flags in `main()`**

In `transcribe_funasr.py`, find the `--model-cache-dir` argument (line ~1155) and add these flags immediately after it (before the `# Backwards compatibility` comment):

```python
    p.add_argument("--mimo-audio-tag", default="<chinese>",
                   choices=["<chinese>", "<english>", "<auto>"],
                   help="MiMo language hint (default: <chinese>). "
                        "Only used with --lang mimo.")
    p.add_argument("--mimo-batch", type=int, default=1,
                   help="Concurrent VAD segments per MiMo inference call "
                        "(default: 1). Increase only on H100/80GB+ cards.")
    p.add_argument("--mimo-weights-path", type=str, default=None,
                   help="Cache directory for MiMo weights. "
                        "Default: $HF_HOME, then ~/.cache/huggingface. "
                        "Also honored by setup_mimo.sh.")
    p.add_argument("--resume-mimo", action="store_true",
                   help="Resume MiMo Phase 1 from *_mimo_partial.json "
                        "(after a mid-run failure).")
```

- [ ] **Step 9.6: Dispatch `--lang mimo` to `mimo_asr`**

In `transcribe_funasr.py`, replace the flags block that sets `hotwords = None` when the preset doesn't support hotwords (lines ~1203–1208):

Old:
```python
    # Resolve hotwords
    hotwords = resolve_hotwords(args.hotwords) if args.hotwords else None
    preset = MODEL_PRESETS[args.lang]
    if hotwords and not preset.get("hotword_support"):
        print(f"  Warning: --hotwords ignored for --lang {args.lang} "
              f"(only supported with --lang zh / SeACo-Paraformer)")
        hotwords = None
```

New:
```python
    # Resolve hotwords
    hotwords = resolve_hotwords(args.hotwords) if args.hotwords else None
    preset = MODEL_PRESETS[args.lang]
    if hotwords and not preset.get("hotword_support"):
        print(f"  Warning: --hotwords ignored for --lang {args.lang} "
              f"(only supported with --lang zh / SeACo-Paraformer)")
        hotwords = None

    # MiMo-specific flag compatibility warnings
    if args.lang == "mimo":
        resolved_compat = warn_on_incompatible_flags(
            args.lang, hotwords, args.batch_size, default_batch=300,
        )
        hotwords = resolved_compat["hotwords"]
```

Then, in the Phase 1 branch (line ~1273), replace the single-line call:

Old:
```python
        transcript = transcribe_with_funasr(asr_audio, args.lang, num_speakers,
                                            args.device, args.batch_size, hotwords)
```

New:
```python
        if args.lang == "mimo":
            import mimo_asr
            mimo_weights = resolve_mimo_weights_path(args.mimo_weights_path)
            os.environ["HF_HOME"] = mimo_weights  # propagate to HF libs
            transcript = mimo_asr.transcribe_with_mimo(
                asr_audio,
                num_speakers=num_speakers,
                audio_tag=args.mimo_audio_tag,
                batch=args.mimo_batch,
                weights_path=mimo_weights,
                resume=args.resume_mimo,
                device=args.device,
                spk_model_id=preset["spk"],
                vad_model_id=preset["vad"],
            )
        else:
            transcript = transcribe_with_funasr(
                asr_audio, args.lang, num_speakers,
                args.device, args.batch_size, hotwords,
            )
```

Also update the `validate_lang_diarization` function (line ~130) to NOT flag `mimo` as incompatible — MiMo has diarization via our CAM++ layer:

No code change needed; `validate_lang_diarization` only checks `lang in ("auto", "whisper")`, which correctly excludes `mimo`.

- [ ] **Step 9.7: Run tests — expect PASS**

Run:
```bash
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 -m pytest test_mimo_asr.py::TestCliWiring -v
python3 -m pytest test_speaker_verification.py -v  # regression
```
Expected: all PASS (both files).

- [ ] **Step 9.8: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/transcribe_funasr.py \
        plugins/funasr-transcriber/skills/funasr-transcribe/scripts/test_mimo_asr.py
git commit -m "feat(mimo): wire --lang mimo into transcribe_funasr CLI"
```

---

## Task 10: `setup_mimo.sh`

**Files:**
- Create: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_mimo.sh`

Opt-in installer. Called by `setup_env.sh` when `INSTALL_MIMO=1`, and runnable standalone. Idempotent (skips clone/download when already present).

- [ ] **Step 10.1: Create `setup_mimo.sh`**

Write `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_mimo.sh`:

```bash
#!/usr/bin/env bash
# Install MiMo-V2.5-ASR: clone the reference repo, install flash-attn, and
# download HF weights. Called automatically by setup_env.sh when
# INSTALL_MIMO=1, or can be run standalone.
#
# Environment:
#   VENV_DIR             — path to the active venv (default: .venv)
#   MIMO_WEIGHTS_PATH    — cache dir for MiMo weights (default: $HF_HOME
#                          or ~/.cache/huggingface)

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
MIMO_WEIGHTS_PATH="${MIMO_WEIGHTS_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}"
MIMO_REPO_URL="https://github.com/XiaomiMiMo/MiMo-V2.5-ASR.git"
MIMO_REPO_DIR="$VENV_DIR/mimo"

echo "=== MiMo-V2.5-ASR Install ==="
echo "  venv:     $VENV_DIR"
echo "  weights:  $MIMO_WEIGHTS_PATH"
echo ""

# Require active venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR. Run setup_env.sh first."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 1. Clone MiMo repo (idempotent)
if [ ! -d "$MIMO_REPO_DIR/.git" ]; then
    echo "[1/4] Cloning $MIMO_REPO_URL into $MIMO_REPO_DIR..."
    git clone --depth 1 "$MIMO_REPO_URL" "$MIMO_REPO_DIR"
else
    echo "[1/4] MiMo repo already present at $MIMO_REPO_DIR — skipping clone."
fi

# 2. Install MiMo's Python dependencies
if [ -f "$MIMO_REPO_DIR/requirements.txt" ]; then
    echo "[2/4] Installing MiMo requirements..."
    pip install -q -r "$MIMO_REPO_DIR/requirements.txt"
else
    echo "  WARNING: $MIMO_REPO_DIR/requirements.txt missing — skipping."
fi

# 3. Install flash-attn (requires nvcc; compile can take 10–30 min)
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "[3/4] flash-attn already installed — skipping."
else
    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: nvcc not found."
        echo "  flash-attn requires the CUDA toolkit (not just the driver)."
        echo "  Install: https://developer.nvidia.com/cuda-toolkit"
        exit 1
    fi
    echo "[3/4] Installing flash-attn==2.7.4.post1 (this can take 10–30 min)..."
    pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

# 4. Download weights (idempotent — huggingface-cli skips cached files)
echo "[4/4] Downloading MiMo weights to $MIMO_WEIGHTS_PATH..."
mkdir -p "$MIMO_WEIGHTS_PATH"
HF_HOME="$MIMO_WEIGHTS_PATH" \
    python3 -m huggingface_hub.commands.huggingface_cli download \
        XiaomiMiMo/MiMo-V2.5-ASR
HF_HOME="$MIMO_WEIGHTS_PATH" \
    python3 -m huggingface_hub.commands.huggingface_cli download \
        XiaomiMiMo/MiMo-Audio-Tokenizer

echo ""
echo "=== MiMo install complete ==="
echo "  Use with: python3 transcribe_funasr.py <audio> --lang mimo \\"
echo "               --mimo-weights-path $MIMO_WEIGHTS_PATH"
```

- [ ] **Step 10.2: Make executable**

```bash
chmod +x plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_mimo.sh
```

- [ ] **Step 10.3: Syntax check**

Run:
```bash
bash -n plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_mimo.sh
echo "syntax OK: $?"
```
Expected: `syntax OK: 0`.

- [ ] **Step 10.4: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_mimo.sh
git commit -m "feat(mimo): add opt-in setup_mimo.sh (clone + flash-attn + weights)"
```

---

## Task 11: Upgrade `setup_env.sh` to Python 3.12 + MiMo hook

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_env.sh`

Detect / require Python 3.12, rebuild venv if it was created with an older version, optionally call `setup_mimo.sh`.

- [ ] **Step 11.1: Replace `setup_env.sh`**

Replace the entire contents of `plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_env.sh` with:

```bash
#!/usr/bin/env bash
# Set up Python venv with FunASR and dependencies.
# Detects GPU/CUDA automatically; falls back to CPU-only PyTorch.
#
# Requires Python 3.12 (needed by the --lang mimo preset even if INSTALL_MIMO
# is not set, to keep dependency resolution consistent across presets).
#
# Usage:
#   bash setup_env.sh            # auto-detect CUDA
#   bash setup_env.sh cpu        # force CPU-only
#   bash setup_env.sh cu121      # force CUDA 12.1
#
#   INSTALL_MIMO=1 bash setup_env.sh
#       After the base install, also run setup_mimo.sh to clone the MiMo repo,
#       install flash-attn, and download MiMo weights. Opt-in because the
#       download is ~20 GB and flash-attn compile takes 10–30 min.

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
FORCE_VARIANT="${1:-auto}"
AUTO_YES="${AUTO_YES:-}"
INSTALL_MIMO="${INSTALL_MIMO:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== FunASR Environment Setup ==="
echo ""
echo "This script will:"
echo "  - Install ffmpeg (system package) if not present"
echo "  - Require Python 3.12; rebuild $VENV_DIR if it was made with another version"
echo "  - Install PyTorch, FunASR, modelscope, boto3 into the venv"
echo "  - Patch FunASR's clustering for long-audio performance"
if [ -n "$INSTALL_MIMO" ]; then
    echo "  - INSTALL_MIMO=1 set: also clone MiMo repo + flash-attn + download weights"
fi
echo ""

if [ -z "$AUTO_YES" ]; then
    if [ ! -t 0 ]; then
        echo "Error: Running non-interactively without AUTO_YES=1."
        echo "  Set AUTO_YES=1 to skip the confirmation prompt."
        exit 1
    fi
    read -rp "Proceed? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy] ]]; then
        echo "Aborted."
        exit 2
    fi
fi

# Ensure ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg not found. Install it manually."
        exit 1
    fi
fi

# Require Python 3.12
if ! command -v python3.12 &>/dev/null; then
    echo "ERROR: python3.12 is required but not found."
    echo ""
    echo "Install instructions:"
    echo "  Ubuntu 24.04:  sudo apt install python3.12 python3.12-venv"
    echo "  Ubuntu 22.04:  add deadsnakes PPA, then:"
    echo "                 sudo apt install python3.12 python3.12-venv"
    echo "                 https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa"
    echo "  macOS:         brew install python@3.12"
    echo "  Other:         https://www.python.org/downloads/"
    exit 1
fi
PY=python3.12

# Rebuild venv if it exists but isn't 3.12
if [ -d "$VENV_DIR" ]; then
    EXISTING_PY="$VENV_DIR/bin/python3"
    if [ -x "$EXISTING_PY" ]; then
        EXISTING_VER=$("$EXISTING_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
        if [ "$EXISTING_VER" != "3.12" ]; then
            echo "Existing venv is Python $EXISTING_VER, rebuilding for 3.12..."
            rm -rf "$VENV_DIR"
        fi
    else
        echo "Existing venv looks broken (no python3), rebuilding..."
        rm -rf "$VENV_DIR"
    fi
fi

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv: $VENV_DIR (Python 3.12)"
    "$PY" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Detect CUDA
if [ "$FORCE_VARIANT" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        if [ "$MAJOR" -ge 12 ] 2>/dev/null; then
            FORCE_VARIANT="cu121"
        elif [ "$MAJOR" -ge 11 ] 2>/dev/null; then
            FORCE_VARIANT="cu118"
        else
            FORCE_VARIANT="cpu"
        fi
        echo "Detected CUDA $CUDA_VER -> $FORCE_VARIANT"
    else
        FORCE_VARIANT="cpu"
        echo "No NVIDIA GPU detected -> CPU mode"
    fi
fi

# Install PyTorch
echo "Installing PyTorch ($FORCE_VARIANT)..."
if [ "$FORCE_VARIANT" = "cpu" ]; then
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install -q torch torchaudio --index-url "https://download.pytorch.org/whl/$FORCE_VARIANT"
fi

# Install FunASR + deps (scikit-learn is new: MiMo path uses KMeans)
echo "Installing FunASR and dependencies..."
pip install -q -U funasr modelscope boto3 scikit-learn soundfile

# Patch clustering for long audio
if [ -f "$SCRIPT_DIR/patch_clustering.py" ]; then
    echo "Applying clustering optimization patch..."
    if ! python3 "$SCRIPT_DIR/patch_clustering.py" --yes; then
        echo "WARNING: Clustering patch failed. Long recordings (>1h) may be very slow."
        echo "  You can retry manually: python3 $SCRIPT_DIR/patch_clustering.py --yes"
    fi
else
    echo "WARNING: patch_clustering.py not found at $SCRIPT_DIR"
    echo "  Long-audio clustering optimization will not be applied."
fi

# Optional: install MiMo
if [ -n "$INSTALL_MIMO" ]; then
    echo ""
    echo "=== INSTALL_MIMO=1 detected — invoking setup_mimo.sh ==="
    bash "$SCRIPT_DIR/setup_mimo.sh"
fi

echo ""
echo "=== Setup complete ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import funasr
print(f'FunASR {funasr.__version__}')
"
echo ""
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Optional LLM provider SDKs (for Phase 3 cleanup):"
echo "  pip install anthropic   # for --model claude-*"
echo "  pip install openai      # for --model gpt-* / deepseek-* / vLLM / Ollama"
echo "  (boto3 is already installed for AWS Bedrock, the default provider)"
```

- [ ] **Step 11.2: Syntax check**

Run:
```bash
bash -n plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_env.sh
echo "syntax OK: $?"
```
Expected: `syntax OK: 0`.

- [ ] **Step 11.3: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/scripts/setup_env.sh
git commit -m "feat(mimo): require python 3.12 in setup_env.sh, hook setup_mimo.sh"
```

---

## Task 12: Regression — full test suite + help text smoke check

**Files:**
- No changes.

- [ ] **Step 12.1: Run all tests**

Run:
```bash
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 -m pytest test_speaker_verification.py test_mimo_asr.py -v
```
Expected: all tests PASS.

- [ ] **Step 12.2: Help text smoke check**

Run:
```bash
cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts
python3 transcribe_funasr.py --help 2>&1 | grep -E "mimo|mimo-audio-tag|resume-mimo"
```
Expected output includes lines like:
```
  --lang {zh,zh-basic,en,auto,whisper,mimo}
  --mimo-audio-tag {<chinese>,<english>,<auto>}
  --mimo-batch MIMO_BATCH
  --mimo-weights-path MIMO_WEIGHTS_PATH
  --resume-mimo
```

- [ ] **Step 12.3: GPU-less preflight smoke check**

On a machine without a GPU (or with CUDA_VISIBLE_DEVICES=-1), verify the preflight fails cleanly:

Run:
```bash
CUDA_VISIBLE_DEVICES="-1" python3 -c "
import sys, os
sys.path.insert(0, 'plugins/funasr-transcriber/skills/funasr-transcribe/scripts')
import mimo_asr
try:
    mimo_asr.require_cuda_and_vram(min_gb=20)
    print('UNEXPECTED: no error raised')
    sys.exit(1)
except RuntimeError as e:
    print(f'OK: got expected error: {e}')
"
```
Expected: `OK: got expected error: --lang mimo requires a CUDA GPU...`

- [ ] **Step 12.4: No commit**

This task verifies behavior; no files changed.

---

## Task 13: Documentation — `SKILL.md`

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/SKILL.md`

Bump version, add `mimo` to the language table, document new flags, add a "MiMo-V2.5-ASR" section explaining setup and limitations.

- [ ] **Step 13.1: Bump version**

In `SKILL.md`, change the `version:` frontmatter field:

Old:
```
version: 1.6.0
```

New:
```
version: 1.7.0
```

- [ ] **Step 13.2: Update language table**

In `SKILL.md`, find the "Supported Languages" table (around line 55) and add a `mimo` row under the `whisper` row:

Old:
```
| `whisper` | Whisper-large-v3-turbo | 99 languages | No |
```

New:
```
| `whisper` | Whisper-large-v3-turbo | 99 languages | No |
| `mimo` | MiMo-V2.5-ASR (local 8B, GPU-only) | zh/en/code-switch/dialects | No |
```

And update the "All presets include..." line immediately below to:

Old:
```
All presets include **speaker diarization** (CAM++) and **VAD** (FSMN).
```

New:
```
All presets include **speaker diarization** (CAM++) and **VAD** (FSMN).
`mimo` reuses the FSMN VAD + CAM++ stack around MiMo's text output.
```

- [ ] **Step 13.3: Update the diarization caveat**

Old:
```
> **Diarization caveat:** `auto` and `whisper` do not output per-sentence timestamps,
> so speaker diarization does not work with these presets. Use `zh`, `zh-basic`, or
> `en` when speaker identification is needed (e.g., podcasts, meetings).
```

New:
```
> **Diarization caveat:** `auto` and `whisper` do not output per-sentence timestamps,
> so speaker diarization does not work with these presets. Use `zh`, `zh-basic`,
> `en`, or `mimo` when speaker identification is needed (e.g., podcasts, meetings).
```

- [ ] **Step 13.4: Add "MiMo-V2.5-ASR" section before "Key Flags"**

In `SKILL.md`, locate the `## Key Flags` heading. Insert this new section immediately above it:

```markdown
## MiMo-V2.5-ASR (optional, GPU-only)

`--lang mimo` runs Xiaomi's
[MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) locally on a
CUDA GPU. Use it when:
- You want to evaluate MiMo against Paraformer on Chinese audio.
- The recording has heavy code-switching, dialects (Wu, Cantonese, Hokkien,
  Sichuanese), lyrics, or rare proper nouns that other presets mis-transcribe.

**Requirements:**
- CUDA ≥12.0 and **≥20 GB VRAM** (16 GB cards OOM during inference).
- Python 3.12 (enforced by `setup_env.sh`).
- ~20 GB weight download (one-time) and `flash-attn==2.7.4.post1` compile
  (needs `nvcc` from the CUDA toolkit, takes 10–30 min).

**Install (opt-in):**

```bash
# One-time: install MiMo on top of the standard environment
AUTO_YES=1 INSTALL_MIMO=1 \
    MIMO_WEIGHTS_PATH=/mnt/models/hf \
    bash $SCRIPTS/setup_env.sh
```

**Run:**

```bash
python3 $SCRIPTS/transcribe_funasr.py podcast.m4a \
    --lang mimo --num-speakers 2 \
    --mimo-weights-path /mnt/models/hf
```

**Resume after failure:**

```bash
python3 $SCRIPTS/transcribe_funasr.py podcast.m4a \
    --lang mimo --resume-mimo --mimo-weights-path /mnt/models/hf
```

**Limitations:**
- No hotword biasing (MiMo has no API for it — `--hotwords` is ignored).
- No CPU fallback.
- Inference is slower than Paraformer on the same GPU (8B model vs ~0.3B);
  expect RTF around 0.1–0.2 on an A100.
```

- [ ] **Step 13.5: Add MiMo flags to "Key Flags" table**

In `SKILL.md`, find the `## Key Flags` table and append these rows at the end (before `## Outputs`):

```
| `--mimo-audio-tag` | MiMo language hint: `<chinese>` (default), `<english>`, `<auto>` |
| `--mimo-batch N` | Concurrent VAD segments per MiMo call (default 1; H100/80GB can go higher) |
| `--mimo-weights-path DIR` | Cache dir for MiMo weights (default: `$HF_HOME` → `~/.cache/huggingface`) |
| `--resume-mimo` | Resume MiMo Phase 1 from `*_mimo_partial.json` after a mid-run failure |
```

- [ ] **Step 13.6: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/SKILL.md
git commit -m "docs(mimo): document --lang mimo preset in SKILL.md"
```

---

## Task 14: Documentation — `references/pipeline-details.md`, `README.md`, CHANGELOG

**Files:**
- Modify: `plugins/funasr-transcriber/skills/funasr-transcribe/references/pipeline-details.md`
- Modify: `README.md`
- Create: `CHANGELOG.md` (repo root — does not currently exist; add at this task)

- [ ] **Step 14.1: Update pipeline-details.md**

Append the following section to the end of `plugins/funasr-transcriber/skills/funasr-transcribe/references/pipeline-details.md`:

```markdown
## `--lang mimo` — Xiaomi MiMo-V2.5-ASR (local, GPU)

MiMo is an 8B-parameter LLM-based ASR model from Xiaomi. It outputs plain text
with no per-sentence timestamps and no speaker labels. The `funasr-transcribe`
skill wraps it in a VAD + speaker-clustering sandwich so output format matches
the FunASR presets:

```
Phase 1a  FSMN VAD           → [(start_ms, end_ms), ...]
Phase 1b  MiMo asr_sft()     → text per VAD segment
Phase 1c  CAM++ + KMeans     → speaker ID per VAD segment
```

Files: `scripts/mimo_asr.py` (orchestrator), `scripts/setup_mimo.sh` (installer).

### Expected RTF

On a single A100 (40 GB), 4h audio → ~24 min wall clock for Phase 1 (RTF ≈
0.1). Compare to `--lang zh` on the same GPU at RTF ≈ 0.02–0.05. Trade speed
for reported accuracy gains on dialects, code-switching, and lyrics.

### Resume (`--resume-mimo`)

Segment-level failures (OOM, CUDA error) retry 3× with
[0.5s, 2s, 5s] backoff after `gc.collect()` + `torch.cuda.empty_cache()`. If all
retries fail, a `*_mimo_partial.json` file captures VAD segments, completed
transcriptions, and the failed index. `--resume-mimo` picks up from the failed
segment, verifying audio SHA256 + `--mimo-audio-tag` match before continuing.
```

- [ ] **Step 14.2: Update README.md**

Read the top of `README.md` to understand its structure:

```bash
head -60 README.md
```

Add `--lang mimo` to any feature bullet list / supported-languages summary
near the top. If there's a "Features" section, add a bullet like:

```markdown
- **Local MiMo-V2.5-ASR support (new in 1.7.0):** opt-in `--lang mimo` runs
  Xiaomi's 8B ASR model locally on a CUDA GPU for dialect-heavy or
  code-switching audio, with diarization preserved via FSMN VAD + CAM++.
  Requires Python 3.12, ≥20 GB VRAM, and `INSTALL_MIMO=1 bash setup_env.sh`.
```

If the README doesn't have a features section, skip this step — just ensure
`--lang mimo` appears somewhere in the README so search engines and
skill-install users can see the feature.

- [ ] **Step 14.3: Create `CHANGELOG.md` (repo root)**

Write `CHANGELOG.md`:

```markdown
# Changelog

## 1.7.0 (2026-04-29)

### Added
- **`--lang mimo`:** local inference with Xiaomi's MiMo-V2.5-ASR (8B,
  GPU-only), reusing FSMN VAD + CAM++ diarization so output format matches
  `--lang zh`.
- **`scripts/setup_mimo.sh`:** opt-in installer (`INSTALL_MIMO=1 bash
  setup_env.sh`) that clones the MiMo repo, installs `flash-attn`, and
  downloads ~20 GB of weights to `$MIMO_WEIGHTS_PATH` (defaults to
  `$HF_HOME` → `~/.cache/huggingface`).
- **`--mimo-audio-tag`, `--mimo-batch`, `--mimo-weights-path`,
  `--resume-mimo`:** CLI flags for the new preset. `--resume-mimo` picks up
  from a mid-run failure using `*_mimo_partial.json` with audio-hash
  verification.
- New tests in `scripts/test_mimo_asr.py` (mocked; GPU-free, CI-safe).

### Changed — BREAKING ENVIRONMENT CHANGE
- **`setup_env.sh` now requires Python 3.12.** Existing `.venv/`
  directories created with earlier Python versions are detected and
  rebuilt on the next run of `setup_env.sh`. Expect a 2–3 GB re-download
  of FunASR dependencies.
- New dependencies: `scikit-learn` (for KMeans clustering in the MiMo
  path), `soundfile` (already transitively present, now explicit).

### Notes
- `--lang mimo` hard-fails if CUDA is unavailable or VRAM < 20 GB. There
  is no CPU fallback for this preset.
- `--hotwords` and `--batch-size` are silently ignored with
  `--lang mimo`; use `--mimo-batch` for per-call concurrency.
```

- [ ] **Step 14.4: Commit**

```bash
git add plugins/funasr-transcriber/skills/funasr-transcribe/references/pipeline-details.md \
        README.md CHANGELOG.md
git commit -m "docs(mimo): update pipeline-details, README, and add CHANGELOG for 1.7.0"
```

---

## Self-Review Checklist

Run through this after completing all tasks. This is YOUR check, not a subagent dispatch.

1. **Spec coverage** — does every requirement in
   `docs/superpowers/specs/2026-04-29-mimo-asr-integration-design.md` map to
   at least one task?
   - Preflights (CUDA/VRAM, weights/repo) → Tasks 2, 3 ✓
   - VAD + per-segment extraction → Task 6 ✓
   - Per-segment retry with backoff → Task 5 ✓
   - Partial save + resume with hash validation → Tasks 4, 8 ✓
   - CAM++ speaker clustering on VAD segments → Task 7 ✓
   - End-to-end orchestrator (failure + happy + resume paths) → Task 8 ✓
   - New CLI flags + dispatch → Task 9 ✓
   - Python 3.12 migration in `setup_env.sh` → Task 11 ✓
   - `setup_mimo.sh` opt-in installer → Task 10 ✓
   - Docs (SKILL.md, pipeline-details, README, CHANGELOG) → Tasks 13, 14 ✓
   - Version bump to 1.7.0 → Task 13.1 ✓

2. **Placeholder scan** — no "TBD", "TODO", "handle edge cases", or similar.
   All code steps show complete code. All commands show expected output.

3. **Type / name consistency**
   - `require_cuda_and_vram(min_gb=20)` — defined Task 2, called Task 8 ✓
   - `require_mimo_installed(weights_path, repo_path)` — defined Task 3,
     called Task 8 ✓
   - `save_partial(partial_path, audio_hash, audio_tag, weights_path,
     vad_segments, completed, failed_at)` — defined Task 4, called Task 8 ✓
   - `load_partial(partial_path, audio_hash, audio_tag)` — defined Task 4,
     called Task 8 ✓
   - `compute_audio_hash(path)` — defined Task 4, called Task 8 ✓
   - `infer_with_retry(mimo, audio_path, audio_tag, max_retries, backoffs)` —
     defined Task 5, called Task 8 ✓
   - `_cuda_cleanup()` — defined Task 5, called Tasks 5 and 8 ✓
   - `run_fsmn_vad(audio_path, model_id, device, max_single_segment_time)` —
     defined Task 6, called Task 8 ✓
   - `extract_segment(audio_path, start_ms, end_ms, out_dir)` — defined
     Task 6, called Task 8 ✓
   - `assign_speakers_via_cam(segments, audio_path, num_speakers,
     spk_model_id, device)` — defined Task 7, called Task 8 ✓
   - `_extract_speaker_embedding(start_ms, end_ms, spk_model, audio_data,
     sample_rate)` — defined Task 7, called Task 7 ✓
   - `_load_mimo(weights_path)` — defined Task 8, used Task 8 ✓
   - `_free_mimo(mimo)` — defined Task 8, used Task 8 ✓
   - `warn_on_incompatible_flags(lang, hotwords, batch_size, default_batch)` —
     defined Task 9, called Task 9 ✓
   - `resolve_mimo_weights_path(cli_value)` — defined Task 9, called Task 9 ✓
   - CLI flag names consistent: `--mimo-audio-tag`, `--mimo-batch`,
     `--mimo-weights-path`, `--resume-mimo` ✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-29-mimo-asr-integration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
