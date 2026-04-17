#!/usr/bin/env python3
"""Verify and fix speaker assignments in transcripts using LLM analysis.

Works with both podcast (2 speakers, host/guest roles) and multi-speaker
meeting scenarios. Analyzes the first N minutes of a raw transcript JSON
to detect mismatched speaker labels, then optionally rewrites the JSON
and/or regenerates a corrected markdown transcript.

Inputs:
  - *_raw_transcript.json from Phase 1
  - --speakers "Alice,Bob" (current label order)
  - --speaker-context speaker-context.json (role descriptions)

Usage:
  # Dry-run: just check if labels are swapped
  python3 verify_speakers.py podcast_raw_transcript.json \
      --speakers "孟岩,李继刚" \
      --speaker-context speaker-context.json

  # Fix in place: rewrite the JSON with corrected speaker IDs
  python3 verify_speakers.py podcast_raw_transcript.json \
      --speakers "孟岩,李继刚" \
      --speaker-context speaker-context.json --fix

  # Multi-speaker meeting: full reassignment
  python3 verify_speakers.py meeting_raw_transcript.json \
      --speakers "Alice,Bob,Carol,Dave" \
      --speaker-context speaker-context.json --fix

  # Use different LLM / analysis window
  python3 verify_speakers.py podcast_raw_transcript.json \
      --speakers "Host,Guest" \
      --speaker-context ctx.json \
      --minutes 5 --model claude-sonnet-4-6
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from llm_utils import call_llm, detect_llm_provider


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def format_time_ms(ms: int) -> str:
    s = ms / 1000
    return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"


def build_speaker_map(transcript: list, speakers: list) -> dict:
    seen_ids = []
    for s in transcript:
        if s["speaker"] not in seen_ids:
            seen_ids.append(s["speaker"])
    mapping = {}
    for i, spk_id in enumerate(seen_ids):
        mapping[spk_id] = speakers[i] if i < len(speakers) else f"Speaker {spk_id + 1}"
    return mapping


def extract_early_transcript(transcript: list, minutes: int,
                             speaker_map: dict) -> str:
    if not transcript:
        return ""
    cutoff_ms = transcript[0]["start_ms"] + minutes * 60 * 1000
    lines = []
    for s in transcript:
        if s["start_ms"] > cutoff_ms:
            break
        name = speaker_map.get(s["speaker"], f"Speaker {s['speaker']}")
        lines.append(f"[{format_time_ms(s['start_ms'])}] {name}: {s['text']}")
    return "\n".join(lines)


def compute_speaker_stats(transcript: list, speaker_map: dict,
                          minutes: Optional[int] = None) -> dict:
    """Per-speaker character count and segment count within the analysis window."""
    cutoff_ms = None
    if minutes and transcript:
        cutoff_ms = transcript[0]["start_ms"] + minutes * 60 * 1000
    stats = {}
    for s in transcript:
        if cutoff_ms and s["start_ms"] > cutoff_ms:
            break
        name = speaker_map.get(s["speaker"], f"Speaker {s['speaker']}")
        if name not in stats:
            stats[name] = {"segments": 0, "chars": 0}
        stats[name]["segments"] += 1
        stats[name]["chars"] += len(s["text"])
    return stats


# ──────────────────────────────────────────────
# Podcast verification (2 speakers)
# ──────────────────────────────────────────────

def verify_podcast(early_text: str, speaker_map: dict,
                   speaker_context: dict, model_id: str, region: str) -> dict:
    """Binary host/guest swap detection for 2-speaker podcasts."""
    names = list(speaker_map.values())
    speaker_list = "\n".join(f"- {n}" for n in names)
    context_lines = "\n".join(f"- {n}: {info}" for n, info in speaker_context.items())

    system_prompt = (
        "You are a speaker identification expert. Analyze the transcript excerpt "
        "and determine whether the speaker labels are correctly assigned.\n\n"
        f"Current speaker label assignments:\n{speaker_list}\n\n"
        f"Expected speaker roles:\n{context_lines}\n\n"
        "Analyze the CONTENT of what each speaker says:\n"
        "- Who introduces the show/topic or the other person? → host\n"
        "- Who asks questions and guides conversation? → host\n"
        "- Who answers questions and shares expertise/stories? → guest\n"
        "- Who does opening/closing remarks? → host\n\n"
        "Respond in this exact format (3 lines):\n"
        "VERDICT: CORRECT or SWAP\n"
        "CONFIDENCE: HIGH, MEDIUM, or LOW\n"
        "EVIDENCE: one sentence explaining your reasoning"
    )

    result = call_llm(system_prompt, early_text, model_id, region)
    verdict, confidence, evidence = "UNKNOWN", "LOW", ""
    for line in result.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("EVIDENCE:"):
            evidence = line.split(":", 1)[1].strip()

    return {"verdict": verdict, "confidence": confidence, "evidence": evidence}


# ──────────────────────────────────────────────
# Multi-speaker meeting verification
# ──────────────────────────────────────────────

def verify_meeting(early_text: str, speaker_map: dict,
                   speaker_context: dict, model_id: str, region: str) -> dict:
    """Full reassignment analysis for multi-speaker meetings."""
    names = list(speaker_map.values())
    speaker_list = "\n".join(f"- {n}" for n in names)
    context_lines = "\n".join(f"- {n}: {info}" for n, info in speaker_context.items())

    system_prompt = (
        "You are a speaker identification expert for multi-speaker meetings. "
        "Analyze the transcript excerpt and determine whether each speaker label "
        "is correctly matched to the right person.\n\n"
        f"Current speaker label assignments:\n{speaker_list}\n\n"
        f"Expected speaker roles/descriptions:\n{context_lines}\n\n"
        "For each speaker in the transcript, analyze what they talk about and how "
        "they speak, then match them to the correct person from the expected roles.\n\n"
        "Respond in this exact JSON format:\n"
        "```json\n"
        "{\n"
        '  "correct": true/false,\n'
        '  "confidence": "HIGH/MEDIUM/LOW",\n'
        '  "mapping": {\n'
        '    "current_label_1": "correct_name_1",\n'
        '    "current_label_2": "correct_name_2"\n'
        "  },\n"
        '  "evidence": {\n'
        '    "current_label_1": "why this person is actually correct_name_1",\n'
        '    "current_label_2": "why this person is actually correct_name_2"\n'
        "  }\n"
        "}\n"
        "```\n"
        "If labels are already correct, mapping should echo the current assignments."
    )

    result = call_llm(system_prompt, early_text, model_id, region)

    # Extract JSON from response
    json_match = re.search(r"\{[\s\S]*\}", result)
    if not json_match:
        return {"correct": None, "confidence": "LOW",
                "mapping": {}, "evidence": {}, "raw": result}

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"correct": None, "confidence": "LOW",
                "mapping": {}, "evidence": {}, "raw": result}

    return parsed


# ──────────────────────────────────────────────
# Apply fixes
# ──────────────────────────────────────────────

def apply_swap(transcript: list, speaker_map: dict, id_a: int, id_b: int) -> list:
    """Swap two speaker IDs in the raw transcript."""
    for seg in transcript:
        if seg["speaker"] == id_a:
            seg["speaker"] = id_b
        elif seg["speaker"] == id_b:
            seg["speaker"] = id_a
    speaker_map[id_a], speaker_map[id_b] = speaker_map[id_b], speaker_map[id_a]
    return transcript


def apply_meeting_mapping(transcript: list, speaker_map: dict,
                          mapping: dict) -> list:
    """Apply a full name remapping from meeting verification result.

    mapping: {"current_label": "correct_name", ...}
    """
    # Build reverse map: name -> speaker_id
    name_to_id = {v: k for k, v in speaker_map.items()}

    # Build the permutation: speaker_id -> new_speaker_id
    id_remap = {}
    for current_label, correct_name in mapping.items():
        if current_label == correct_name:
            continue
        src_id = name_to_id.get(current_label)
        dst_id = name_to_id.get(correct_name)
        if src_id is not None and dst_id is not None:
            id_remap[src_id] = dst_id

    if not id_remap:
        return transcript

    # Apply permutation via temporary offset to avoid collisions.
    # Cyclic remaps (e.g. A->B, B->A) would overwrite in a single pass,
    # so we first shift remapped IDs into a high range, then subtract.
    offset = max(s["speaker"] for s in transcript) + 1000
    for seg in transcript:
        if seg["speaker"] in id_remap:
            seg["speaker"] = id_remap[seg["speaker"]] + offset
    for seg in transcript:
        if seg["speaker"] >= offset:
            seg["speaker"] -= offset

    # Update speaker_map
    for current_label, correct_name in mapping.items():
        sid = name_to_id.get(current_label)
        if sid is not None:
            speaker_map[sid] = correct_name

    return transcript


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Verify and fix speaker assignments in raw transcript JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("transcript_json",
                   help="Path to *_raw_transcript.json from Phase 1")
    p.add_argument("--speakers", required=True,
                   help="Comma-separated speaker names in current label order")
    p.add_argument("--speaker-context", required=True,
                   help="JSON file with per-speaker role descriptions")
    p.add_argument("--minutes", type=int, default=5,
                   help="Minutes of transcript to analyze (default: 5)")
    p.add_argument("--model", default="us.anthropic.claude-sonnet-4-6",
                   help="LLM model ID (auto-detects Bedrock/Anthropic/OpenAI)")
    p.add_argument("--bedrock-region", default="us-west-2",
                   help="AWS region for Bedrock")
    p.add_argument("--fix", action="store_true",
                   help="Apply corrections to the JSON file (otherwise dry-run)")
    p.add_argument("--output", default=None,
                   help="Write corrected JSON to a different file instead of overwriting")
    args = p.parse_args()

    # Load transcript
    json_path = Path(args.transcript_json)
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    print(f"Loaded {len(transcript)} segments from {json_path}")

    # Parse speakers
    speaker_names = [s.strip() for s in args.speakers.split(",")]
    speaker_map = build_speaker_map(transcript, speaker_names)
    num_speakers = len(speaker_map)
    print(f"Speakers ({num_speakers}): {', '.join(speaker_map.values())}")

    # Load speaker context
    ctx_path = Path(args.speaker_context)
    if not ctx_path.exists():
        print(f"Error: speaker context file not found: {ctx_path}")
        sys.exit(1)
    try:
        with open(ctx_path, "r", encoding="utf-8") as f:
            speaker_context = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: {ctx_path} is not valid JSON: {e}")
        sys.exit(1)

    # Extract early transcript
    early_text = extract_early_transcript(transcript, args.minutes, speaker_map)
    if not early_text:
        print("Error: no segments within analysis window")
        sys.exit(1)

    early_lines = early_text.count("\n") + 1
    print(f"Analyzing first {args.minutes} minutes ({early_lines} segments)...\n")

    # Show speaking stats
    stats = compute_speaker_stats(transcript, speaker_map, args.minutes)
    print("Speaking stats (analysis window):")
    for name, s in stats.items():
        print(f"  {name}: {s['segments']} segments, {s['chars']} chars")
    print()

    # Run verification
    provider = detect_llm_provider(args.model)
    print(f"Model: {args.model} (provider: {provider})")

    if num_speakers == 2:
        print("Mode: podcast (2-speaker swap detection)\n")
        result = verify_podcast(early_text, speaker_map, speaker_context,
                                args.model, args.bedrock_region)
        verdict = result["verdict"]
        confidence = result["confidence"]
        evidence = result["evidence"]

        print(f"Verdict:    {verdict}")
        print(f"Confidence: {confidence}")
        print(f"Evidence:   {evidence}")

        if verdict not in ("CORRECT", "SWAP"):
            print(f"\nWARNING: Unrecognized verdict '{verdict}'. "
                  f"Verification inconclusive — manual review recommended.")
            sys.exit(2)

        needs_fix = verdict == "SWAP"
        if needs_fix and args.fix:
            ids = list(speaker_map.keys())
            print(f"\nApplying swap: {speaker_map[ids[0]]} ↔ {speaker_map[ids[1]]}")
            transcript = apply_swap(transcript, speaker_map, ids[0], ids[1])
        elif needs_fix:
            ids = list(speaker_map.keys())
            print(f"\nSwap needed: {speaker_map[ids[0]]} ↔ {speaker_map[ids[1]]}")
            print("Run with --fix to apply correction")

    else:
        print(f"Mode: meeting ({num_speakers}-speaker reassignment)\n")
        result = verify_meeting(early_text, speaker_map, speaker_context,
                                args.model, args.bedrock_region)

        correct = result.get("correct", None)
        confidence = result.get("confidence", "UNKNOWN")
        mapping = result.get("mapping", {})
        evidence = result.get("evidence", {})

        print(f"Correct:    {correct}")
        print(f"Confidence: {confidence}")

        if correct is None:
            raw = result.get("raw", "")
            if raw:
                print(f"\nRaw LLM response (first 300 chars):\n  {raw[:300]}")
            print("\nWARNING: Verification inconclusive — could not parse LLM "
                  "response. Manual review recommended.")
            sys.exit(2)

        if evidence:
            print("\nEvidence:")
            for name, reason in evidence.items():
                print(f"  {name}: {reason}")

        # Check if mapping differs from current
        has_changes = any(k != v for k, v in mapping.items())
        if mapping:
            print("\nProposed mapping:")
            for current, correct_name in mapping.items():
                marker = " ← CHANGE" if current != correct_name else ""
                print(f"  {current} → {correct_name}{marker}")

        # Validate mapping before applying
        known_names = set(speaker_map.values())
        if has_changes:
            targets = [v for k, v in mapping.items() if k != v]
            if len(targets) != len(set(targets)):
                dupes = {t for t in targets if targets.count(t) > 1}
                print(f"\nERROR: LLM mapping has duplicate targets: {dupes}")
                print("Skipping reassignment to avoid data corruption.")
                sys.exit(2)
            unknown = [n for n in targets if n not in known_names]
            if unknown:
                print(f"\nERROR: Mapping references unknown speakers: {unknown}")
                print("Skipping reassignment.")
                sys.exit(2)

        needs_fix = has_changes
        if needs_fix and args.fix:
            print("\nApplying reassignment...")
            transcript = apply_meeting_mapping(transcript, speaker_map, mapping)
        elif needs_fix:
            print("\nRun with --fix to apply corrections")

    # Write output
    if needs_fix and args.fix:
        out_path = Path(args.output) if args.output else json_path
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print(f"\nCorrected transcript saved: {out_path}")
        print(f"Re-run the main pipeline with --skip-transcribe to regenerate markdown")
    elif not needs_fix:
        print("\nNo corrections needed — speaker labels appear correct.")


if __name__ == "__main__":
    main()
