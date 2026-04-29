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
        if not partial_path.exists():
            raise RuntimeError(
                f"--resume-mimo requested but {partial_path.name} not found. "
                f"Run without --resume-mimo to start fresh."
            )
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

    # 3. Load MiMo. Initialize to None so the `finally` block is safe even if
    # a future refactor moves _load_mimo inside the try. Today, _load_mimo is
    # outside the try so an exception here exits the function before `finally`
    # is reached, but the guard costs nothing and future-proofs the pattern.
    print(f"[Phase 1b] MiMo ASR (local, GPU)")
    print(f"  Loading MiMo from {weights_path}...")
    t_load = time.time()
    mimo = None
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
        if mimo is not None:
            _free_mimo(mimo)

    wall = time.time() - t0
    if vad_segments:
        duration = (vad_segments[-1][1] - vad_segments[0][0]) / 1000
        if duration > 0:
            print(f"  MiMo inference: {wall:.1f}s (RTF {wall / duration:.3f})")

    # 5. Speaker clustering on the same VAD segments. Guard against the
    # "impossible" state where the loop exited without raising and yet some
    # segments are missing from `completed` — this would indicate a future bug
    # or a concurrent-modification scenario and is cheaper to surface clearly.
    print(f"[Phase 1c] CAM++ speaker clustering...")
    if len(completed) != len(vad_segments):
        raise RuntimeError(
            f"Incomplete transcription: {len(completed)}/{len(vad_segments)} "
            f"segments completed without raising. The transcript may be "
            f"corrupt. Try resuming with --resume-mimo, or delete "
            f"{partial_path.name} to restart."
        )
    segments = [completed[i] for i in range(len(vad_segments))]
    segments = assign_speakers_via_cam(segments, audio_path, num_speakers,
                                       spk_model_id, device)

    # 6. Clean up partial on success
    if partial_path.exists():
        partial_path.unlink()

    return segments


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

    `subprocess.run` is called with a list (no shell=True) so values are
    passed as individual argv elements to ffmpeg without shell interpretation
    — there is no shell-injection path. We still validate that the audio
    exists so failures surface with a clear message instead of ffmpeg's
    cryptic "No such file or directory" output hundreds of segments later.
    """
    audio_p = Path(audio_path)
    if not audio_p.is_file():
        raise RuntimeError(f"Audio file not found: {audio_path}")

    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0
    out_path = Path(out_dir) / f"seg_{start_ms:010d}_{end_ms:010d}.wav"
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(audio_p),
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
