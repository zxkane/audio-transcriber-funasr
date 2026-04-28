#!/usr/bin/env python3
"""Speaker gender classification using 3D-Speaker CAM++ two-class gender model.

Runs on segments from the raw transcript, aggregates per-speaker predictions
with a majority vote, and returns a mapping from speaker ID to "male"/"female".

Also parses reference (show notes / agenda) text for explicit gender hints.
Reference hints override automatic classifier output when both are available.

Typical usage:

    spk_gender = classify_speaker_gender(
        audio_path, transcript, speaker_map,
        model_id="iic/speech_campplus_two_class_gender_16k",
        device="cuda:0",
    )
    # spk_gender = {0: "male", 1: "female"}

    ref_gender = extract_gender_from_reference(reference_text)
    # ref_gender = {"Alice": "female", "Bob": "male"}

    final = merge_gender_sources(spk_gender, ref_gender, speaker_map)
"""

import re
from typing import Dict, Optional

VALID_GENDERS = ("male", "female")


def _select_sample_segments(transcript: list, speaker_id: int,
                            max_samples: int = 3,
                            min_duration_ms: int = 1500) -> list:
    """Pick up to max_samples clearest segments (longest) for a given speaker.

    Short segments (< min_duration_ms) are unreliable for gender classification
    because there may not be enough voicing to extract stable features.
    """
    candidates = [
        s for s in transcript
        if s["speaker"] == speaker_id
        and (s["end_ms"] - s["start_ms"]) >= min_duration_ms
    ]
    candidates.sort(key=lambda s: s["end_ms"] - s["start_ms"], reverse=True)
    return candidates[:max_samples]


def _normalize_gender_label(raw) -> Optional[str]:
    """Normalize classifier output strings to 'male'/'female'/None."""
    if not raw:
        return None
    low = str(raw).strip().lower()
    if low in ("male", "m", "man", "男", "男性"):
        return "male"
    if low in ("female", "f", "woman", "女", "女性"):
        return "female"
    return None


def _majority_vote(labels: list) -> Optional[str]:
    """Return the majority label, or None on tie/empty input."""
    labels = [l for l in labels if l in VALID_GENDERS]
    if not labels:
        return None
    male_count = labels.count("male")
    female_count = labels.count("female")
    if male_count == female_count:
        return None
    return "male" if male_count > female_count else "female"


def classify_speaker_gender(audio_path: str, transcript: list,
                            speaker_ids: Optional[list] = None,
                            model_id: str = "iic/speech_campplus_two_class_gender_16k",
                            device: str = "cpu",
                            max_samples: int = 3,
                            _model_loader=None) -> Dict[int, str]:
    """Classify gender for each speaker by voting across sample segments.

    Uses ModelScope's CAM++ 16k two-class gender model via the 3D-Speaker stack.
    For each speaker, takes up to max_samples of their longest segments,
    runs inference on each, and majority-votes the result.

    Returns a dict {speaker_id: "male"|"female"}. Speakers with ambiguous or
    failed classification are omitted (not mapped to "unknown").

    Args:
        audio_path: path to the preprocessed audio file (16kHz mono)
        transcript: list of segments with speaker/start_ms/end_ms
        speaker_ids: limit classification to these IDs (None = all)
        model_id: ModelScope model ID (default: CAM++ gender 16k)
        device: torch device
        max_samples: max segments per speaker to sample for voting
        _model_loader: test hook — callable returning an object with
            infer(audio_segment_np) -> label. If None, uses ModelScope.
    """
    if not transcript:
        return {}

    ids = speaker_ids if speaker_ids is not None else sorted(
        {s["speaker"] for s in transcript})
    if not ids:
        return {}

    per_speaker_samples = {
        spk: _select_sample_segments(transcript, spk, max_samples=max_samples)
        for spk in ids
    }
    if not any(per_speaker_samples.values()):
        return {}

    try:
        infer_fn = _build_infer_fn(audio_path, model_id, device, _model_loader)
    except Exception as e:  # noqa: BLE001 — degrade gracefully if deps missing
        print(f"  WARNING: gender classifier unavailable ({type(e).__name__}: {e})")
        return {}

    result = {}
    for spk, samples in per_speaker_samples.items():
        if not samples:
            continue
        labels = []
        for seg in samples:
            try:
                raw = infer_fn(seg["start_ms"], seg["end_ms"])
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING: gender inference failed at {seg['start_ms']}ms: {e}")
                continue
            label = _normalize_gender_label(raw)
            if label:
                labels.append(label)
        voted = _majority_vote(labels)
        if voted:
            result[spk] = voted
    return result


def _build_infer_fn(audio_path: str, model_id: str, device: str, _model_loader):
    """Build a function (start_ms, end_ms) -> raw_label.

    Isolates the ModelScope/soundfile dependency so tests can inject a fake
    loader that never touches the filesystem.
    """
    if _model_loader is not None:
        model = _model_loader()
        return lambda start_ms, end_ms: model.infer(start_ms, end_ms)

    import soundfile as sf  # type: ignore
    from modelscope.pipelines import pipeline  # type: ignore
    from modelscope.utils.constant import Tasks  # type: ignore

    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    clf = pipeline(
        task=Tasks.speaker_verification,
        model=model_id,
        device=device,
    )

    def _infer_real(start_ms: int, end_ms: int):
        start = int(start_ms * sr / 1000)
        end = int(end_ms * sr / 1000)
        segment = audio[start:end]
        if len(segment) < sr * 0.3:
            return None
        out = clf(segment)
        if isinstance(out, dict):
            # ModelScope pipeline outputs vary by model; try common keys
            for key in ("text", "label", "result", "gender"):
                if key in out and out[key]:
                    val = out[key]
                    if isinstance(val, (list, tuple)) and val:
                        val = val[0]
                    return val
        return out

    return _infer_real


# ──────────────────────────────────────────────
# Reference-text gender extraction
# ──────────────────────────────────────────────

_NAME_CLASS = r"[\w一-鿿·\-]{1,30}"

# Role words excluded when matching "name (gender)" so "主播（女）" or "Host (female)"
# is not picked up as if "主播"/"Host" were a person's name.
_ROLE_WORDS = {"主播", "主持", "主持人", "主持员", "嘉宾",
               "host", "guest"}


def extract_gender_from_reference(reference_text: Optional[str]) -> Dict[str, str]:
    """Best-effort gender extraction from show notes / reference text.

    Recognizes common patterns:
        "Alice (female)", "Bob (男)", "主播（女）: 韩梅梅"
        "男主播 李雷", "女嘉宾 Carol"
        "Host: Alice [female]"

    Returns a dict {name: "male"|"female"}. Names not found get no entry.
    """
    if not reference_text:
        return {}

    result: Dict[str, str] = {}

    # Pattern 1: role-prefix with inline gender — "主播（女）：韩梅梅", "Host (M): Alice"
    role_gender_name = re.compile(
        r"(?:主播|主持[人员]?|嘉宾|Host|Guest)\s*"
        r"[（(\[]\s*(男|女|male|female|m|f|man|woman)\s*[)）\]]"
        r"\s*[:：\-—–]\s*(" + _NAME_CLASS + ")",
        re.IGNORECASE,
    )
    for m in role_gender_name.finditer(reference_text):
        gender = _normalize_gender_label(m.group(1))
        name = m.group(2).strip()
        if gender and name:
            result.setdefault(name, gender)

    # Pattern 2: gender-prefixed role — "男主播 李雷", "female guest Carol"
    gender_role_name = re.compile(
        r"(男|女|male|female)\s*(?:主播|主持[人员]?|嘉宾|Host|Guest)"
        r"\s*[:：\-—– ]\s*(" + _NAME_CLASS + ")",
        re.IGNORECASE,
    )
    for m in gender_role_name.finditer(reference_text):
        gender = _normalize_gender_label(m.group(1))
        name = m.group(2).strip()
        if gender and name:
            result.setdefault(name, gender)

    # Pattern 3: name followed by inline gender — "Alice (female)", "韩梅梅（女）"
    name_paren_gender = re.compile(
        r"(" + _NAME_CLASS + r")\s*"
        r"[（(\[]\s*(男|女|male|female)\s*[)）\]]",
        re.IGNORECASE,
    )
    for m in name_paren_gender.finditer(reference_text):
        name = m.group(1).strip()
        gender = _normalize_gender_label(m.group(2))
        if gender and name and name.lower() not in _ROLE_WORDS:
            result.setdefault(name, gender)

    return result


def merge_gender_sources(auto: Optional[Dict[int, str]],
                         reference: Optional[Dict[str, str]],
                         speaker_map: Optional[Dict[int, str]]) -> Dict[int, str]:
    """Combine auto-detected and reference-sourced gender into final per-id map.

    Reference (name-based) takes precedence over auto (ID-based) when both
    refer to the same speaker ID via speaker_map.
    """
    auto = auto or {}
    reference = reference or {}
    speaker_map = speaker_map or {}

    name_to_id: Dict[str, int] = {name: sid for sid, name in speaker_map.items() if name}

    merged: Dict[int, str] = dict(auto)
    for name, gender in reference.items():
        sid = name_to_id.get(name)
        if sid is not None and gender in VALID_GENDERS:
            merged[sid] = gender
    return merged


def parse_gender_cli_arg(raw: Optional[str],
                         speaker_map: Dict[int, str]) -> Dict[int, str]:
    """Parse --speaker-genders 'Alice:female,Bob:male' into a per-id map.

    Supports:
      - name:gender comma-separated pairs
      - bare gender list (comma-separated) matched to speaker_map insertion order
    """
    if not raw:
        return {}
    items = [p.strip() for p in raw.split(",") if p.strip()]

    result: Dict[int, str] = {}
    if all(":" in p or "=" in p for p in items):
        name_to_id = {v: k for k, v in speaker_map.items()}
        for item in items:
            sep = ":" if ":" in item else "="
            name, _, g = item.partition(sep)
            name = name.strip()
            gender = _normalize_gender_label(g.strip())
            sid = name_to_id.get(name)
            if sid is not None and gender:
                result[sid] = gender
        return result

    ordered_ids = list(speaker_map.keys())
    for i, item in enumerate(items):
        if i >= len(ordered_ids):
            break
        gender = _normalize_gender_label(item)
        if gender:
            result[ordered_ids[i]] = gender
    return result


def format_gender_label(gender: Optional[str]) -> str:
    """Render gender as a human-readable parenthetical suffix, empty if unknown."""
    if gender == "male":
        return "(male)"
    if gender == "female":
        return "(female)"
    return ""
