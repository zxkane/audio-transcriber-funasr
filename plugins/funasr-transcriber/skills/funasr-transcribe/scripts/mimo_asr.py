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
    """Phase 1 MiMo path: VAD -> MiMo ASR -> CAM++ speaker labels. Not implemented yet."""
    raise NotImplementedError


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
