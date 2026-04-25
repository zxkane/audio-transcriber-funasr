#!/usr/bin/env python3
"""FunASR: Multi-language meeting transcription with speaker diarization + LLM cleanup.

Four-phase pipeline optimized for multi-speaker meetings:
  Phase 0: Audio preprocessing (ffmpeg conversion + duration validation)
  Phase 1: FunASR ASR + speaker diarization (+ optional hotword biasing)
  Phase 2: Post-processing (merge, speaker mapping, auto-verify via self-intro)
  Phase 3: LLM cleanup via Bedrock/Anthropic/OpenAI (opt-in, requires --model)

Language presets:
  zh        — SeACo-Paraformer (best Chinese, CER 1.95%, hotword support)
  zh-basic  — Paraformer-large (Chinese, no hotword, lighter)
  en        — Paraformer-en (English)
  auto      — SenseVoiceSmall (auto-detect: zh/en/ja/ko/yue)
  whisper   — Whisper-large-v3-turbo (99 languages)

Supports GPU (recommended) and CPU (slower but functional).
First run auto-downloads models from ModelScope.

Usage:
  # Chinese meeting with hotwords (names, terms)
  python3 transcribe_funasr.py meeting.wav --lang zh --num-speakers 9 \\
      --hotwords "张三 李四 ClawCon Rebase"

  # Hotwords from file (one per line)
  python3 transcribe_funasr.py meeting.wav --lang zh --hotwords hotwords.txt

  # English meeting
  python3 transcribe_funasr.py meeting.wav --lang en --num-speakers 4

  # Auto-detect language (no speaker diarization — use zh/en for that)
  python3 transcribe_funasr.py meeting.wav --lang auto

  # Whisper for any language (no speaker diarization — use zh/en for that)
  python3 transcribe_funasr.py meeting.wav --lang whisper

  # With real speaker names
  python3 transcribe_funasr.py meeting.wav --speakers "Alice,Bob,Carol"

  # CPU mode
  python3 transcribe_funasr.py meeting.wav --lang zh --device cpu

  # Raw transcription only (no LLM)
  python3 transcribe_funasr.py meeting.wav --skip-llm

  # Resume interrupted LLM cleanup
  python3 transcribe_funasr.py meeting.wav --skip-transcribe

  # Speaker context JSON to help LLM identify speakers
  python3 transcribe_funasr.py meeting.wav --speaker-context context.json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from llm_utils import call_llm, detect_llm_provider


# ──────────────────────────────────────────────
# Language-specific model presets
# ──────────────────────────────────────────────

MODEL_PRESETS = {
    "zh": {
        "label": "Chinese (SeACo-Paraformer, hotword-enabled)",
        # SeACo-Paraformer: hotword-customizable Paraformer variant
        # Paper: "SeACo-Paraformer: A Non-Autoregressive ASR System with
        # Flexible and Effective Hotword Customization Ability"
        "asr": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": True,
    },
    "zh-basic": {
        "label": "Chinese (Paraformer-large, no hotword)",
        "asr": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
    "en": {
        "label": "English (Paraformer-en)",
        "asr": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
    "auto": {
        "label": "Auto-detect (SenseVoiceSmall: zh/en/ja/ko/yue)",
        "asr": "iic/SenseVoiceSmall",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": None,  # SenseVoiceSmall includes punctuation
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
    "whisper": {
        "label": "Multilingual (Whisper-large-v3-turbo, 99 languages)",
        "asr": "iic/Whisper-large-v3-turbo",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": None,  # Whisper includes punctuation
        "spk": "iic/speech_campplus_sv_zh-cn_16k-common",
        "hotword_support": False,
    },
}

SUPPORTED_LANGS = list(MODEL_PRESETS.keys())


def validate_lang_diarization(lang: str, num_speakers: Optional[int]) -> None:
    """Fail fast if language preset is incompatible with speaker diarization."""
    if lang in ("auto", "whisper") and num_speakers:
        print(f"ERROR: --lang {lang} does not support speaker diarization "
              f"(no per-sentence timestamps). Use --lang zh or --lang en instead.")
        sys.exit(1)


# ──────────────────────────────────────────────
# Audio preprocessing with duration validation
# ──────────────────────────────────────────────

def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    raw = result.stdout.strip()
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(
            f"ffprobe returned non-numeric duration for {path}: {raw!r}. "
            f"The file may be corrupt or missing duration metadata."
        )


def preprocess_audio(input_path: str, output_format: str = "flac") -> str:
    """Convert audio to 16kHz mono for ASR. Validates output duration matches input.

    Returns the path to the converted file (or the original if already suitable).
    """
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"'{tool}' not found. Install ffmpeg: "
                f"sudo apt-get install ffmpeg (or use --skip-preprocess)")
    inp = Path(input_path)
    # Skip conversion for formats FunASR reads natively at 16kHz
    if inp.suffix.lower() in (".wav", ".flac") and _is_16k_mono(input_path):
        print(f"  Audio already 16kHz mono: {input_path}")
        return input_path

    out_path = inp.with_suffix(f".{output_format}")
    if out_path.exists():
        # Validate pre-existing file is not corrupt from a previous failed run
        try:
            get_audio_duration(str(out_path))
            print(f"  Converted file exists: {out_path}")
        except RuntimeError:
            print(f"  WARNING: Existing {out_path} appears corrupt, re-converting...")
            out_path.unlink()
    if not out_path.exists():
        print(f"  Converting {inp.name} → {out_path.name} ...")
        codec_args = {
            "opus": ["-c:a", "libopus", "-b:a", "32k"],
            "flac": ["-sample_fmt", "s16"],
            "wav": [],
        }.get(output_format, [])
        cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1",
               *codec_args, str(out_path), "-y"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")

    # Duration validation — catch silent truncation
    in_dur = get_audio_duration(input_path)
    out_dur = get_audio_duration(str(out_path))
    diff = abs(in_dur - out_dur)
    print(f"  Input duration: {in_dur:.1f}s, output duration: {out_dur:.1f}s (diff: {diff:.1f}s)")
    if diff > 5.0:
        raise RuntimeError(
            f"Audio truncation detected! Input: {in_dur:.1f}s, output: {out_dur:.1f}s "
            f"(lost {diff:.1f}s). Aborting to prevent incomplete transcription. "
            f"Try converting to FLAC instead: ffmpeg -i {input_path} -ar 16000 -ac 1 "
            f"-sample_fmt s16 {inp.with_suffix('.flac')}"
        )
    return str(out_path)


def _is_16k_mono(path: str) -> bool:
    """Check if audio is already 16kHz mono."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,channels", "-of", "json", path],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return False  # ffprobe not installed; preprocess_audio will catch this
    if result.returncode != 0:
        return False
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    streams = info.get("streams", [])
    if not streams:
        return False
    return streams[0].get("sample_rate") == "16000" and streams[0].get("channels") == 1


# ──────────────────────────────────────────────
# Phase 1: FunASR transcription
# ──────────────────────────────────────────────

def parse_funasr_results(res: list) -> list:
    """Parse FunASR output into a normalized transcript list.

    Handles all known FunASR result shapes:
    1. sentence_info — composite models with speaker diarization
    2. text + timestamp — models that produce word/segment timestamps without sentence_info
    3. text only — plain text output (SenseVoice, Whisper)
    """
    transcript = []
    for entry in res:
        if "sentence_info" in entry:
            for sent in entry["sentence_info"]:
                transcript.append({
                    "speaker": int(sent.get("spk", 0)),
                    "start_ms": sent["start"],
                    "end_ms": sent["end"],
                    "text": sent.get("text", sent.get("sentence", "")),
                })
        elif "text" in entry:
            timestamps = entry.get("timestamp", [])
            start_ms = timestamps[0][0] if timestamps else 0
            end_ms = timestamps[-1][-1] if timestamps else 0
            transcript.append({
                "speaker": 0,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": entry["text"],
            })
        else:
            print(f"  WARNING: Unrecognized FunASR result shape, "
                  f"keys: {sorted(entry.keys())}")
    return transcript


def transcribe_with_funasr(audio_path: str, lang: str = "zh",
                           num_speakers: Optional[int] = None,
                           device: str = "cuda:0",
                           batch_size_s: int = 300,
                           hotwords: Optional[str] = None) -> list:
    """Run FunASR for ASR and speaker diarization.

    Loads language-specific models and runs the full pipeline:
    VAD -> ASR -> punctuation -> speaker embedding -> clustering.

    For 'zh' mode with hotwords, uses SeACo-Paraformer which biases
    recognition toward provided hotwords (names, terms, etc.).
    """
    from funasr import AutoModel

    preset = MODEL_PRESETS[lang]
    print(f"[Phase 1] Language: {preset['label']}")
    print(f"  Loading models (device={device})...")
    t0 = time.time()

    model_kwargs = {
        "model": preset["asr"],
        "vad_model": preset["vad"],
        "vad_kwargs": {"max_single_segment_time": 60000},
        "spk_model": preset["spk"],
        "device": device,
        "disable_update": True,
    }
    if preset.get("punc"):
        model_kwargs["punc_model"] = preset["punc"]

    # SeACo-Paraformer: pass hotwords at model init
    if hotwords and preset.get("hotword_support"):
        model_kwargs["hotword"] = hotwords
        print(f"  Hotwords: {hotwords[:100]}{'...' if len(hotwords) > 100 else ''}")

    model = AutoModel(**model_kwargs)
    print(f"  Models loaded: {time.time() - t0:.1f}s")

    generate_kwargs = {"input": audio_path, "batch_size_s": batch_size_s}
    if num_speakers:
        generate_kwargs["preset_spk_num"] = num_speakers

    print(f"  Transcribing: {audio_path} (speakers={num_speakers}, batch={batch_size_s}s)")
    t1 = time.time()
    res = model.generate(**generate_kwargs)
    elapsed = time.time() - t1
    print(f"  Transcription done: {elapsed:.1f}s")

    transcript = parse_funasr_results(res)

    speakers = sorted(set(s["speaker"] for s in transcript))
    print(f"  Sentences: {len(transcript)}, speakers detected: {len(speakers)}")
    for spk in speakers:
        count = sum(1 for s in transcript if s["speaker"] == spk)
        print(f"    spk {spk}: {count}")

    return transcript


# ──────────────────────────────────────────────
# Phase 2: Post-processing
# ──────────────────────────────────────────────

def merge_consecutive(transcript: list, gap_ms: int = 2000,
                      max_merge_ms: int = 120000) -> list:
    """Merge consecutive sentences from the same speaker within gap_ms.

    Caps merged segment duration at max_merge_ms so long single-speaker
    stretches (e.g. solo podcasts) retain periodic timestamps instead of
    collapsing into one timestamp-less block.
    """
    if not transcript:
        return []
    merged = []
    cur = dict(transcript[0])
    for sent in transcript[1:]:
        same_speaker = sent["speaker"] == cur["speaker"]
        small_gap = (sent["start_ms"] - cur["end_ms"]) < gap_ms
        under_cap = (sent["end_ms"] - cur["start_ms"]) < max_merge_ms
        if same_speaker and small_gap and under_cap:
            cur["end_ms"] = sent["end_ms"]
            cur["text"] += sent["text"]
        else:
            merged.append(cur)
            cur = dict(sent)
    merged.append(cur)
    return merged


def extract_speaker_names_from_reference(reference_text: Optional[str]) -> list:
    """Best-effort extraction of speaker names from show notes / reference text.

    Recognizes common Chinese/English podcast role labels like
    "主播：李雷", "嘉宾: Alice", "Host: Bob", "Guest — Carol". Host is
    listed first so single-speaker recordings resolve to the host name.
    Returns an empty list if no labels match.
    """
    if not reference_text:
        return []

    hosts, guests = [], []
    # Name class: CJK chars, ASCII letters/digits, and a few name punctuations
    # (·-·). Stops at anything else — whitespace, sentence terminators,
    # parentheses, brackets, quotes, commas. Keeps us safe from
    # "李雷.本期..." or "Alice (senior analyst)" being captured as names.
    name_re = r"[\w一-鿿·\-]{1,30}"
    role_patterns = [
        (r"(?:主播|主持[人员]?|Host)\s*[:：\-—–]\s*(" + name_re + ")", hosts),
        (r"(?:嘉宾|Guest)\s*[:：\-—–]\s*(" + name_re + ")", guests),
    ]
    for pat, bucket in role_patterns:
        for m in re.finditer(pat, reference_text, re.IGNORECASE):
            name = m.group(1).strip()
            if name and name not in bucket:
                bucket.append(name)

    # Dedup across buckets so "主播：Alice\n嘉宾：Alice" doesn't produce
    # duplicate speaker names downstream.
    ordered = []
    for n in hosts + guests:
        if n not in ordered:
            ordered.append(n)
    return ordered


def build_speaker_map(transcript: list, speakers: Optional[list] = None) -> dict:
    """Build speaker ID -> display name mapping.

    If speaker names are provided, assign by first-appearance order.
    Otherwise use generic labels.
    """
    seen_ids = []
    for s in transcript:
        if s["speaker"] not in seen_ids:
            seen_ids.append(s["speaker"])

    if speakers:
        mapping = {}
        for i, spk_id in enumerate(seen_ids):
            mapping[spk_id] = speakers[i] if i < len(speakers) else f"Speaker {spk_id + 1}"
        return mapping

    return {spk_id: f"Speaker {spk_id + 1}" for spk_id in seen_ids}


def _name_variants(name: str) -> list:
    """Generate matching variants for a speaker name.

    For Chinese names (2-4 chars, all CJK): returns [full_name, given_name].
    e.g. "孙冰洁" → [("孙冰洁", "孙冰洁"), ("冰洁", "孙冰洁")].
    For non-Chinese names: returns [full_name] only.
    Each variant is returned as (variant, full_name) so matches map back.
    """
    result = [(name, name)]
    if 2 <= len(name) <= 4 and all('\u4e00' <= c <= '\u9fff' for c in name):
        given = name[1:]
        if given != name:
            result.append((given, name))
    return result


def detect_montage_end(transcript: list, max_scan_ms: int = 180000) -> int:
    """Detect the end of a cold-open montage section.

    Montage = rapid-fire short clips at the start, typically each < 12 seconds,
    followed by a noticeably longer segment (the real show intro). Returns the
    index of the first non-montage segment, or 0 if no montage is detected.

    Heuristic: find the first segment >= 15s within the scan window. If at least
    3 prior segments exist and most (>= 75%) are short (< 12s), that's a montage.
    """
    if len(transcript) < 4:
        return 0

    SHORT_THRESHOLD_MS = 12000
    LONG_THRESHOLD_MS = 15000

    for i, seg in enumerate(transcript):
        if seg["start_ms"] > max_scan_ms:
            break
        duration = seg["end_ms"] - seg["start_ms"]
        if i >= 3 and duration >= LONG_THRESHOLD_MS:
            short_count = sum(
                1 for j in range(i)
                if (transcript[j]["end_ms"] - transcript[j]["start_ms"]) < SHORT_THRESHOLD_MS
            )
            if short_count / i >= 0.75:
                return i
    return 0


def rescore_montage_speakers(transcript: list, montage_end: int,
                             audio_path: str, spk_model_id: str,
                             device: str = "cuda:0",
                             profile_minutes: int = 5) -> list:
    """Re-assign speakers in the montage zone using embedding similarity.

    Extracts speaker embeddings via CAM++ for each segment, builds reference
    profiles from the first N minutes of post-montage content, then re-scores
    montage segments by cosine similarity to each profile.

    Args:
        transcript: raw transcript with start_ms/end_ms/speaker per segment
        montage_end: index of first non-montage segment (from detect_montage_end)
        audio_path: path to the preprocessed audio file
        spk_model_id: FunASR speaker model ID (e.g. iic/speech_campplus_sv_zh-cn_16k-common)
        device: torch device
        profile_minutes: minutes of post-montage audio to build speaker profiles

    Returns:
        transcript with montage segment speakers reassigned
    """
    if montage_end <= 0 or montage_end >= len(transcript):
        return transcript

    try:
        import numpy as np
        import soundfile as sf
        from funasr import AutoModel
    except ImportError as e:
        print(f"  WARNING: Cannot rescore montage speakers (missing dep: {e})")
        return transcript

    print(f"  Montage re-scoring: extracting speaker embeddings...")

    spk_model = AutoModel(model=spk_model_id, device=device, disable_update=True)

    audio_data, sample_rate = sf.read(audio_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]

    def extract_embedding(start_ms: int, end_ms: int):
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)
        segment = audio_data[start_sample:end_sample]
        if len(segment) < sample_rate * 0.3:
            return None
        try:
            result = spk_model.generate(input=segment)
            if result and isinstance(result, list) and len(result) > 0:
                emb = result[0].get("spk_embedding")
                if emb is not None:
                    arr = np.array(emb, dtype=np.float32).flatten()
                    return arr
        except Exception as e:
            print(f"  WARNING: embedding extraction failed at {start_ms}ms: {e}")
        return None

    post_montage = transcript[montage_end:]
    profile_cutoff_ms = post_montage[0]["start_ms"] + profile_minutes * 60 * 1000
    profile_segments = [s for s in post_montage if s["start_ms"] <= profile_cutoff_ms]

    speaker_embeddings = {}
    for seg in profile_segments:
        spk = seg["speaker"]
        emb = extract_embedding(seg["start_ms"], seg["end_ms"])
        if emb is not None:
            speaker_embeddings.setdefault(spk, []).append(emb)

    if len(speaker_embeddings) < 2:
        print(f"  WARNING: Only {len(speaker_embeddings)} speaker profile(s) built, "
              f"need at least 2. Skipping montage re-scoring.")
        return transcript

    speaker_profiles = {}
    for spk, embs in speaker_embeddings.items():
        profile = np.mean(embs, axis=0)
        norm = np.linalg.norm(profile)
        if norm < 1e-8:
            print(f"    WARNING: degenerate embedding profile for speaker {spk}, skipping")
            continue
        profile /= norm
        speaker_profiles[spk] = profile
        print(f"    Speaker {spk}: profile from {len(embs)} segments")

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    changes = 0
    for i in range(montage_end):
        seg = transcript[i]
        emb = extract_embedding(seg["start_ms"], seg["end_ms"])
        if emb is None:
            continue
        scores = {spk: cosine_sim(emb, prof) for spk, prof in speaker_profiles.items()}
        best_spk = max(scores, key=lambda k: scores[k])
        if best_spk != seg["speaker"]:
            old = seg["speaker"]
            seg["speaker"] = best_spk
            changes += 1
            print(f"    [{seg['start_ms']}ms] spk {old} → {best_spk} "
                  f"(scores: {', '.join(f'{k}={v:.3f}' for k, v in sorted(scores.items()))})")

    print(f"  Montage re-scoring: {changes}/{montage_end} segments reassigned")
    return transcript


def verify_speaker_assignment(transcript: list, speaker_map: dict,
                              speaker_names: Optional[list] = None) -> dict:
    """Auto-verify speaker assignment by detecting self-introductions.

    Scans the first 5 minutes of transcript. If a speaker says their own name
    (e.g., "我是张飞" / "I'm Alice") but is currently labeled as someone else,
    swap all speaker assignments globally.

    Returns the (possibly corrected) speaker_map.
    """
    if not transcript or not speaker_names or len(speaker_names) < 2:
        return speaker_map

    # Skip montage/cold-open section — diarization is unreliable there
    montage_end = detect_montage_end(transcript)
    if montage_end > 0:
        print(f"  Montage detected: skipping first {montage_end} segments for self-intro scan")

    # Collect segments from first 5 minutes, starting after montage
    post_montage = transcript[montage_end:]
    if not post_montage:
        return speaker_map
    cutoff_ms = post_montage[0]["start_ms"] + 5 * 60 * 1000
    early_segments = [s for s in post_montage if s["start_ms"] <= cutoff_ms]

    # Build name variants: full name + given name for Chinese names
    all_variants = []
    for name in speaker_names:
        all_variants.extend(_name_variants(name))

    # Patterns: allow optional filler between intro phrase and name
    # "我是冰洁", "我是屠龙之术的主播庄明浩", "I'm Alice", etc.
    # Double braces {{}} escape Python .format(); single {name} is the placeholder.
    intro_patterns = [
        r"我是[^。？！\n]{{0,15}}{name}",
        r"我叫[^。？！\n]{{0,10}}{name}",
        r"I'?\s*m\s+{name}",
        r"I\s+am\s+{name}",
        r"my\s+name\s+is\s+{name}",
        r"this\s+is\s+{name}",
        r"大家好[^。？！\n]{{0,20}}{name}",
    ]

    # Check each early segment for self-introductions
    mismatches = []
    confirmations = []
    for seg in early_segments:
        current_label = speaker_map.get(seg["speaker"], "")
        text = seg["text"]
        for variant, full_name in all_variants:
            for pat_template in intro_patterns:
                pat = pat_template.format(name=re.escape(variant))
                if re.search(pat, text, re.IGNORECASE):
                    entry = {
                        "speaker_id": seg["speaker"],
                        "current_label": current_label,
                        "actual_name": full_name,
                        "evidence": text[:100],
                        "time_ms": seg["start_ms"],
                    }
                    if full_name == current_label:
                        confirmations.append(entry)
                    else:
                        mismatches.append(entry)

    if not mismatches:
        if confirmations:
            c = confirmations[0]
            print(f"  Speaker verification: CONFIRMED at [{format_time_ms(c['time_ms'])}], "
                  f"'{c['current_label']}' correctly says their own name.")
        else:
            print("  WARNING: Could not auto-verify speaker assignment. "
                  "Manual review recommended.")
        return speaker_map

    # If we found a mismatch, swap the two speaker IDs globally
    m = mismatches[0]
    print(f"  Speaker verification: at [{format_time_ms(m['time_ms'])}], "
          f"speaker labeled '{m['current_label']}' said a self-introduction "
          f"matching '{m['actual_name']}'. Swapping labels.")

    # Find the two speaker IDs to swap
    id_a = m["speaker_id"]  # Currently mislabeled
    id_b = None
    for spk_id, label in speaker_map.items():
        if label == m["actual_name"]:
            id_b = spk_id
            break

    if id_b is not None and id_a != id_b:
        speaker_map[id_a], speaker_map[id_b] = speaker_map[id_b], speaker_map[id_a]
        print(f"  Swapped: SPEAKER_{id_a} ↔ SPEAKER_{id_b}")
    else:
        speaker_map[id_a] = m["actual_name"]
        print(f"  Relabeled: SPEAKER_{id_a} → {m['actual_name']}")

    return speaker_map


# ──────────────────────────────────────────────
# Phase 3: LLM cleanup (multi-provider)
# ──────────────────────────────────────────────

def format_time_ms(ms: int) -> str:
    s = ms / 1000
    return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"


def format_chunk(chunk: list, speaker_map: dict) -> str:
    lines = []
    for item in chunk:
        name = speaker_map.get(item["speaker"], f"Unknown{item['speaker']}")
        lines.append(f"[{format_time_ms(item['start_ms'])}] {name}: {item['text']}")
    return "\n".join(lines)


def chunk_by_duration(items: list, duration_ms: int = 900000) -> list:
    """Split items into time-based chunks (default 15 min)."""
    if not items:
        return []
    chunks, cur, start = [], [], items[0]["start_ms"]
    for item in items:
        if item["start_ms"] - start >= duration_ms and cur:
            chunks.append(cur)
            cur, start = [], item["start_ms"]
        cur.append(item)
    if cur:
        chunks.append(cur)
    return chunks


DEFAULT_SYSTEM_PROMPT = """You are a professional meeting transcript editor.

Rules:
1. Remove filler words (um, uh, like, you know, 嗯, 啊, 那个, 就是说, 对对对, etc.)
2. Fix grammar errors; make sentences more readable and polished
3. Merge stuttered/repeated expressions into fluent sentences
4. Fix obvious ASR errors based on context
5. Preserve original meaning — do not add content not in the original
6. Keep timestamps unchanged
7. Preserve technical terms and proper nouns
8. Output cleaned text only, format: [timestamp] Name: content
9. Detect montage/highlight-reel sections at the start or end of the recording \
(rapid-fire short clips edited together, each only a few seconds, often previewing \
topics discussed later). Mark these with a section header: \
"[片头混剪]" at the start or "[片尾混剪]" at the end, placed on its own line \
before the first clip in that section.
10. Within montage sections, speaker diarization is unreliable because clips are \
spliced from different parts of the recording. Fix speaker labels based on CONTENT: \
if someone says "我是X" or introduces themselves, that segment belongs to X; \
if the content clearly matches one speaker's role/expertise, reassign accordingly. \
Outside montage sections, trust the existing speaker labels."""


# ──────────────────────────────────────────────
# LLM cleanup with reference context
# ──────────────────────────────────────────────

def build_system_prompt(speaker_context: Optional[dict] = None,
                        reference_text: Optional[str] = None,
                        speaker_names: Optional[list] = None) -> str:
    """Build the LLM system prompt, enriched with all available context."""
    prompt = DEFAULT_SYSTEM_PROMPT

    # Inject canonical speaker names and common ASR error corrections
    if speaker_names:
        prompt += f"\n\nThe speakers in this recording are: {', '.join(speaker_names)}."
        prompt += ("\nCorrect all ASR misrecognitions of these names. "
                   "If a speaker says their own name in the content, treat that as ground truth. "
                   "Common ASR errors for Chinese names include phonetically similar characters "
                   "(e.g., 关于→关羽, 张非→张飞, 刘备→刘备).")

    # Inject speaker context (roles, background)
    if speaker_context:
        prompt += "\n\nSpeaker context (use to fix ASR errors and identify speakers):\n"
        for name, info in speaker_context.items():
            prompt += f"- {name}: {info}\n"

    # Inject show notes / reference material — this gives the LLM a rich vocabulary
    # of correct proper nouns, terms, topics, and names to draw from
    if reference_text:
        # Truncate to ~4000 chars to stay within prompt budget
        notes_text = reference_text[:4000]
        if len(reference_text) > 4000:
            notes_text += "\n[...truncated]"
        prompt += (
            "\n\nReference material (show notes / meeting agenda). "
            "Use this to correct ASR errors — proper nouns, person names, "
            "organization names, technical terms, and topic keywords in this "
            "document are authoritative spellings:\n\n"
            + notes_text
        )

    return prompt


def _verify_speaker_roles_via_llm(first_chunk_text: str, speaker_map: dict,
                                   speaker_context: dict, model_id: str,
                                   region: str) -> dict:
    """Verify and correct speaker label assignments using LLM content analysis.

    For 2 speakers: binary CORRECT/SWAP detection.
    For 3+ speakers: full reassignment via JSON mapping.
    Returns the (possibly corrected) speaker_map.
    """
    names = list(speaker_map.values())
    speaker_list = "\n".join(f"- {n}" for n in names)
    context_lines = "\n".join(f"- {n}: {info}" for n, info in speaker_context.items())

    if len(speaker_map) == 2:
        return _verify_two_speakers(
            first_chunk_text, speaker_map, speaker_list, context_lines, model_id, region)
    return _verify_multi_speakers(
        first_chunk_text, speaker_map, speaker_list, context_lines, model_id, region)


def _verify_two_speakers(first_chunk_text: str, speaker_map: dict,
                          speaker_list: str, context_lines: str,
                          model_id: str, region: str) -> dict:
    verify_prompt = (
        "You are a speaker identification expert. Analyze the following transcript "
        "excerpt and determine whether the speaker labels are correctly assigned.\n\n"
        f"Current speaker assignments:\n{speaker_list}\n\n"
        f"Expected speaker roles:\n{context_lines}\n\n"
        "Analyze the CONTENT of what each speaker says:\n"
        "- Who introduces the show/topic or the other person? (likely host)\n"
        "- Who asks questions and guides conversation? (likely host)\n"
        "- Who answers questions and shares expertise/stories? (likely guest)\n"
        "- Who does opening/closing remarks? (likely host)\n\n"
        "Respond with EXACTLY one line:\n"
        "- If labels are correct: CORRECT\n"
        "- If labels are swapped: SWAP\n"
        "No explanation, just the single word."
    )
    try:
        result = call_llm(verify_prompt, first_chunk_text, model_id, region)
    except ImportError:
        raise
    except Exception as e:
        print(f"  LLM speaker verification failed ({type(e).__name__}: {e}), "
              f"proceeding with original labels")
        return speaker_map

    verdict = result.strip().upper()
    if verdict in ("SWAP", "SWAPPED"):
        ids = list(speaker_map.keys())
        old_first, old_second = speaker_map[ids[0]], speaker_map[ids[1]]
        speaker_map[ids[0]], speaker_map[ids[1]] = old_second, old_first
        print(f"  LLM speaker verification: SWAPPED labels "
              f"({old_first} ↔ {old_second})")
    elif verdict in ("CORRECT", "OK"):
        print("  LLM speaker verification: labels CONFIRMED correct")
    else:
        print(f"  LLM speaker verification: ambiguous response "
              f"'{result.strip()[:80]}', proceeding with original labels")
    return speaker_map


def _verify_multi_speakers(first_chunk_text: str, speaker_map: dict,
                            speaker_list: str, context_lines: str,
                            model_id: str, region: str) -> dict:
    verify_prompt = (
        "You are a speaker identification expert for multi-speaker recordings. "
        "Analyze the transcript excerpt and determine whether each speaker label "
        "is correctly matched to the right person.\n\n"
        f"Current speaker label assignments:\n{speaker_list}\n\n"
        f"Expected speaker roles/descriptions:\n{context_lines}\n\n"
        "For each speaker, analyze what they talk about and how they speak, "
        "then match them to the correct person from the expected roles.\n\n"
        "Respond in this exact JSON format (no other text):\n"
        "```json\n"
        "{\n"
        '  "correct": true,\n'
        '  "mapping": {"current_label_1": "correct_name_1", ...}\n'
        "}\n"
        "```\n"
        "If labels are already correct, mapping should echo the current assignments."
    )
    try:
        result = call_llm(verify_prompt, first_chunk_text, model_id, region)
    except ImportError:
        raise
    except Exception as e:
        print(f"  LLM speaker verification failed ({type(e).__name__}: {e}), "
              f"proceeding with original labels")
        return speaker_map

    json_match = re.search(r"\{[\s\S]*\}", result)
    if not json_match:
        print(f"  LLM speaker verification: could not parse response, "
              f"proceeding with original labels")
        return speaker_map

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        print(f"  LLM speaker verification: invalid JSON in response, "
              f"proceeding with original labels")
        return speaker_map

    mapping = parsed.get("mapping", {})
    has_changes = any(k != v for k, v in mapping.items())
    if not has_changes:
        print("  LLM speaker verification: labels CONFIRMED correct")
        return speaker_map

    # Validate: all names in mapping must be known speakers
    known_names = set(speaker_map.values())
    targets = [v for k, v in mapping.items() if k != v]
    if len(targets) != len(set(targets)):
        print("  LLM speaker verification: duplicate targets in mapping, skipping")
        return speaker_map
    unknown = [n for n in targets if n not in known_names]
    if unknown:
        print(f"  LLM speaker verification: unknown speakers {unknown}, skipping")
        return speaker_map

    # Build the full permutation atomically (handles 3+ way cycles)
    name_to_id = {v: k for k, v in speaker_map.items()}
    new_map = {}
    changes = []
    for current_label, correct_name in mapping.items():
        sid = name_to_id.get(current_label)
        if sid is not None and current_label != correct_name:
            new_map[sid] = correct_name
            changes.append((current_label, correct_name))
        elif sid is not None:
            new_map[sid] = current_label

    speaker_map.update(new_map)
    for old, new in changes:
        print(f"  LLM speaker verification: {old} → {new}")
    return speaker_map


def run_llm_cleanup(merged: list, speaker_map: dict, model_id: str, region: str,
                    speaker_context: Optional[dict] = None, cache_dir: Optional[Path] = None,
                    reference_text: Optional[str] = None, speaker_names: Optional[list] = None) -> list:
    """Chunk merged transcript and clean each via LLM. Supports resume via cache_dir."""
    chunks = chunk_by_duration(merged)

    # Pre-cleanup: verify speaker roles using LLM analysis of first chunk
    if speaker_context and len(speaker_map) >= 2 and chunks:
        first_chunk_text = format_chunk(chunks[0], speaker_map)
        speaker_map = _verify_speaker_roles_via_llm(
            first_chunk_text, speaker_map, speaker_context, model_id, region)

    system_prompt = build_system_prompt(speaker_context, reference_text, speaker_names)
    cleaned = []
    failed_chunks = []
    if cache_dir:
        cache_dir.mkdir(exist_ok=True)

    provider = detect_llm_provider(model_id)
    print(f"  LLM cleanup: {len(chunks)} chunks, model: {model_id} (provider: {provider})")
    for i, chunk in enumerate(chunks):
        cache_file = cache_dir / f"chunk_{i:03d}.txt" if cache_dir else None

        if cache_file and cache_file.exists():
            cleaned.append(cache_file.read_text(encoding="utf-8"))
            print(f"  chunk {i+1}/{len(chunks)} (cached)")
            continue

        chunk_text = format_chunk(chunk, speaker_map)
        user_msg = (f"Clean the following meeting transcript segment "
                    f"({i+1}/{len(chunks)}):\n\n{chunk_text}")
        try:
            result = call_llm(system_prompt, user_msg, model_id, region)
            cleaned.append(result)
            if cache_file:
                cache_file.write_text(result, encoding="utf-8")
        except Exception as e:
            print(f"  ERROR: chunk {i+1} cleanup failed: {type(e).__name__}: {e}")
            print(f"         Falling back to raw text for this chunk.")
            cleaned.append(chunk_text)
            failed_chunks.append(i + 1)

        print(f"  chunk {i+1}/{len(chunks)} done")
        time.sleep(1)

    if failed_chunks:
        pct = len(failed_chunks) / len(chunks) * 100
        print(f"\n  WARNING: {len(failed_chunks)}/{len(chunks)} chunks ({pct:.0f}%) "
              f"used raw text due to LLM failures: chunks {failed_chunks}")
        print(f"  The output contains uncleaned segments. "
              f"Delete the cached chunks and re-run to retry.")

    return cleaned


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def assemble_markdown(cleaned_parts: list, metadata: dict) -> str:
    speakers_list = "\n".join(f"- {name}" for name in metadata.get("speakers", []))
    duration_s = metadata.get("duration_ms", 0) / 1000
    h, m = int(duration_s // 3600), int((duration_s % 3600) // 60)

    title = metadata.get("title", "Meeting Transcript")
    md = f"""# {title}

## Info

| Field | Value |
|-------|-------|
| **File** | {metadata.get('filename', '')} |
| **Duration** | {h}h{m}m |
| **Speakers** | {metadata.get('num_speakers', '?')} |
| **Language** | {metadata.get('language', 'N/A')} |
| **ASR Engine** | {metadata.get('asr_engine', 'FunASR')} |

## Speaker List

{speakers_list}

---

## Transcript

"""
    md += "\n\n".join(cleaned_parts)
    return md


# ──────────────────────────────────────────────
# Hotword helpers
# ──────────────────────────────────────────────

def resolve_hotwords(hotwords_arg: str) -> str:
    """Resolve --hotwords argument.

    If it's a .txt file path, return the path (FunASR reads it directly).
    Otherwise treat as space-separated hotword string.
    """
    if hotwords_arg and hotwords_arg.endswith(".txt") and Path(hotwords_arg).exists():
        with open(hotwords_arg) as f:
            count = sum(1 for line in f if line.strip())
        print(f"  Hotwords file: {hotwords_arg} ({count} words)")
        return hotwords_arg
    return hotwords_arg


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="FunASR multi-speaker meeting transcription with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported languages: {', '.join(SUPPORTED_LANGS)}")
    p.add_argument("audio_file", help="Audio file (WAV recommended; M4A/MP3 also supported)")
    p.add_argument("--lang", default="zh", choices=SUPPORTED_LANGS,
                   help="Language preset (default: zh). "
                        "zh=SeACo-Paraformer(hotword), zh-basic=Paraformer-large, "
                        "en=Paraformer-en, auto=SenseVoiceSmall, whisper=Whisper-v3-turbo")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Number of speakers in the meeting (improves diarization accuracy)")
    p.add_argument("--speakers", type=str, default=None,
                   help="Comma-separated speaker names, e.g. 'Alice,Bob,Carol'")
    p.add_argument("--hotwords", type=str, default=None,
                   help="Hotwords to bias ASR (space-separated string or .txt file). "
                        "Use for participant names, technical terms, project names. "
                        "Only effective with --lang zh (SeACo-Paraformer)")
    p.add_argument("--reference", type=str, default=None,
                   help="Reference text file (show notes, meeting agenda, attendee list, "
                        "etc.) with authoritative names, terms, and topics. "
                        "Injected into LLM prompt to correct ASR errors.")
    p.add_argument("--device", default=None,
                   help="Device: cuda:0 / cpu (auto-detected by default)")
    p.add_argument("--batch-size", type=int, default=300,
                   help="Batch size in seconds. Use 60 for CPU, 100 if GPU OOM (default: 300)")
    p.add_argument("--audio-format", default="flac", choices=["opus", "flac", "wav"],
                   help="Target format for audio preprocessing (default: flac). "
                        "flac is lossless and avoids truncation issues with opus on long audio.")
    p.add_argument("--output", default=None,
                   help="Output file (default: <stem>-transcript.md)")
    p.add_argument("--model", default=None,
                   help="LLM model ID for cleanup (omit to skip LLM cleanup). "
                        "Auto-detects provider: "
                        "Bedrock ARN/cross-region ID → Bedrock converse API, "
                        "claude-* → Anthropic Messages API, "
                        "gpt-*/deepseek-*/other → OpenAI-compatible API")
    p.add_argument("--bedrock-region", default="us-west-2",
                   help="AWS region for Bedrock (only used when provider is bedrock)")
    p.add_argument("--speaker-context", type=str, default=None,
                   help="JSON file with per-speaker context to help LLM identify speakers")
    p.add_argument("--title", type=str, default="Meeting Transcript",
                   help="Title for the output document (default: 'Meeting Transcript')")
    p.add_argument("--phase1-only", action="store_true",
                   help="Exit after Phase 1 (VAD + ASR + diarization). "
                        "Skips speaker verification and LLM cleanup.")
    p.add_argument("--json-out", type=str, default=None, metavar="PATH",
                   help="Write Phase 1 raw transcript JSON to this path "
                        "(overrides default <stem>_raw_transcript.json naming)")
    p.add_argument("--skip-transcribe", action="store_true",
                   help="Skip ASR, load from *_raw_transcript.json")
    p.add_argument("--skip-llm", action="store_true", help="Skip LLM cleanup")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip audio preprocessing (use input file as-is)")
    p.add_argument("--clean-cache", action="store_true",
                   help="Delete LLM chunk cache after completion")
    p.add_argument("--model-cache-dir", type=str, default=None,
                   help="Directory for ModelScope model cache (~3GB). "
                        "Sets MODELSCOPE_CACHE env var. Default: ~/.cache/modelscope/")
    # Backwards compatibility
    p.add_argument("--bedrock-model", type=str, default=None,
                   help=argparse.SUPPRESS)  # Deprecated, use --model
    args = p.parse_args()
    # Resolve model: --model wins, then --bedrock-model, then skip LLM
    if args.model is None:
        args.model = args.bedrock_model
    if args.model is None:
        args.skip_llm = True

    # Set model cache dir before any FunASR import
    if args.model_cache_dir:
        os.environ["MODELSCOPE_CACHE"] = args.model_cache_dir
        print(f"  Model cache: {args.model_cache_dir}")

    audio_path = Path(args.audio_file)
    raw_json = Path(args.json_out) if args.json_out else Path(f"{audio_path.stem}_raw_transcript.json")
    output_path = Path(args.output) if args.output else Path(f"{audio_path.stem}-transcript.md")

    if args.json_out:
        parent = raw_json.parent or Path(".")
        if not parent.exists():
            print(f"Error: --json-out directory does not exist: {parent}")
            sys.exit(1)

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
        print(f"Device: {args.device}")
        if args.device == "cpu" and args.batch_size > 60:
            print(f"  CPU mode: batch_size adjusted to 60 (was {args.batch_size})")
            args.batch_size = 60

    # Parse speaker names
    speaker_names = [s.strip() for s in args.speakers.split(",")] if args.speakers else None
    num_speakers = args.num_speakers or (len(speaker_names) if speaker_names else None)

    # Validate language preset supports diarization when speakers are requested
    validate_lang_diarization(args.lang, num_speakers)

    # Resolve hotwords
    hotwords = resolve_hotwords(args.hotwords) if args.hotwords else None
    preset = MODEL_PRESETS[args.lang]
    if hotwords and not preset.get("hotword_support"):
        print(f"  Warning: --hotwords ignored for --lang {args.lang} "
              f"(only supported with --lang zh / SeACo-Paraformer)")
        hotwords = None

    # Load reference materials
    reference_text = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            reference_text = ref_path.read_text(encoding="utf-8")
            print(f"  Reference loaded: {ref_path} ({len(reference_text)} chars)")
        else:
            print(f"  Warning: --reference file not found: {args.reference}")

    # Fallback: if --speakers not provided but reference has role labels
    # (主播/嘉宾/Host/Guest), use those so the final output shows real names
    # instead of "Speaker 1". Essential for solo podcasts where the host
    # name only appears in the show notes.
    if not speaker_names and reference_text:
        extracted = extract_speaker_names_from_reference(reference_text)
        if extracted:
            speaker_names = extracted
            if num_speakers is None:
                num_speakers = len(extracted)
                if len(extracted) == 1:
                    print(f"  Note: only 1 speaker label ('{extracted[0]}') found in "
                          f"reference. If the recording has more speakers, pass "
                          f"--num-speakers explicitly.")
            print(f"  Speaker names from reference: {', '.join(extracted)}")

    # ── Phase 0: Audio preprocessing ──
    asr_audio = str(audio_path)
    if not args.skip_transcribe and not args.skip_preprocess:
        if audio_path.exists():
            print("[Phase 0] Audio preprocessing...")
            asr_audio = preprocess_audio(str(audio_path), args.audio_format)
        # else: will be caught below

    # ── Phase 1: Transcribe ──
    if args.skip_transcribe:
        if not raw_json.exists():
            print(f"Error: {raw_json} not found. Run full transcription first.")
            sys.exit(1)
        with open(raw_json, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        print(f"Loaded {len(transcript)} sentences from {raw_json}")
    else:
        if not audio_path.exists():
            print(f"Error: {audio_path} not found")
            sys.exit(1)
        transcript = transcribe_with_funasr(asr_audio, args.lang, num_speakers,
                                            args.device, args.batch_size, hotwords)
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"Raw transcript saved: {raw_json}")

    if not transcript:
        print("Error: empty transcript")
        sys.exit(1)

    if args.phase1_only:
        print(f"--phase1-only: stopping after Phase 1 ({len(transcript)} sentences)")
        sys.exit(0)

    # Runtime check: warn if most segments lack timestamps (diarization degraded)
    no_ts = sum(1 for s in transcript if s["start_ms"] == 0 and s["end_ms"] == 0)
    if no_ts > len(transcript) * 0.5:
        print(f"\n  WARNING: {no_ts}/{len(transcript)} segments lack timestamps. "
              f"Speaker diarization will be degraded. "
              f"Use --lang zh or --lang en for better results.")

    # ── Phase 2: Post-process ──
    # Re-score montage speakers using embedding similarity (before merge/mapping)
    montage_end = detect_montage_end(transcript)
    if montage_end > 0 and audio_path.exists():
        preset = MODEL_PRESETS.get(args.lang, MODEL_PRESETS["zh"])
        transcript = rescore_montage_speakers(
            transcript, montage_end, asr_audio,
            preset["spk"], args.device)

    merged = merge_consecutive(transcript)
    speaker_map = build_speaker_map(transcript, speaker_names)
    # Auto-verify speaker assignment via self-introductions
    speaker_map = verify_speaker_assignment(transcript, speaker_map, speaker_names)
    print(f"[Phase 2] Merged: {len(transcript)} sentences -> {len(merged)} segments")

    # ── Phase 3: LLM cleanup ──
    speaker_context = None
    if args.speaker_context:
        with open(args.speaker_context, "r", encoding="utf-8") as f:
            speaker_context = json.load(f)

    if args.skip_llm:
        chunks = chunk_by_duration(merged)
        cleaned_parts = [format_chunk(chunk, speaker_map) for chunk in chunks]
    else:
        cache_dir = Path(f"{audio_path.stem}_llm_cache")
        print("[Phase 3] LLM cleanup...")
        cleaned_parts = run_llm_cleanup(merged, speaker_map, args.model,
                                        args.bedrock_region, speaker_context,
                                        cache_dir, reference_text, speaker_names)
        if args.clean_cache and cache_dir.exists():
            for f in cache_dir.glob("chunk_*.txt"):
                f.unlink()
            cache_dir.rmdir()
            print("  LLM cache cleaned")

    # ── Output ──
    duration_ms = transcript[-1]["end_ms"] - transcript[0]["start_ms"]
    actual_speakers = sorted(set(s["speaker"] for s in transcript))
    md = assemble_markdown(cleaned_parts, {
        "title": args.title,
        "filename": audio_path.name,
        "duration_ms": duration_ms,
        "num_speakers": len(actual_speakers),
        "language": preset["label"],
        "asr_engine": f"FunASR ({preset['asr'].split('/')[-1]})",
        "speakers": [speaker_map.get(s, f"Speaker {s+1}") for s in actual_speakers],
    })
    output_path.write_text(md, encoding="utf-8")
    print(f"\nDone: {output_path} ({len(merged)} segments, "
          f"{len(actual_speakers)} speakers, {format_time_ms(duration_ms)})")


if __name__ == "__main__":
    main()
