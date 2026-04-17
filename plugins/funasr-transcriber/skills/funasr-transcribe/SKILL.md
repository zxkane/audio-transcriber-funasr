---
name: funasr-transcribe
version: 1.2.0
description: >
  This skill should be used when the user asks to "transcribe a meeting",
  "transcribe audio", "transcribe a meeting recording",
  "convert audio to text", "generate meeting minutes from audio",
  "do speech-to-text", "transcribe with speaker diarization",
  "identify speakers in audio", "transcribe Chinese audio",
  "transcribe English audio", "transcribe Japanese audio",
  "multi-speaker transcription", "transcribe a podcast",
  "transcribe podcast episode", "transcribe an interview",
  "convert podcast to text", "podcast to transcript",
  or mentions FunASR, Paraformer, SenseVoice, Whisper, meeting
  transcription, podcast transcription, or speaker diarization.
  Supports multi-speaker meeting and podcast transcription in Chinese,
  English, Japanese, Korean, Cantonese, and 99 languages (via Whisper)
  with automatic speaker diarization and hotword biasing.
  Works on both GPU and CPU. Use this skill even when the user doesn't
  say "transcribe" explicitly — e.g., "I have a podcast episode I need
  turned into text" or "convert this interview recording" should trigger it.
---

# FunASR Meeting Transcription

Transcribe multi-speaker meetings, podcasts, and interviews into
structured Markdown with automatic speaker diarization, hotword biasing,
and optional LLM cleanup, using the open-source FunASR pipeline.

**Optimized for long-form audio**: handles arbitrarily long recordings
(4+ hours tested), separates speakers via CAM++ diarization,
merges consecutive utterances, and maps speaker IDs to real names.
Works for both large meetings (10+ speakers) and podcasts (2–3 speakers).

## Supported Languages

| `--lang` | Model | Languages | Hotword |
|----------|-------|-----------|---------|
| `zh` (default) | SeACo-Paraformer | Chinese (CER 1.95%) | Yes |
| `zh-basic` | Paraformer-large | Chinese | No |
| `en` | Paraformer-en | English | No |
| `auto` | SenseVoiceSmall | Auto-detect: zh/en/ja/ko/yue | No |
| `whisper` | Whisper-large-v3-turbo | 99 languages | No |

All presets include **speaker diarization** (CAM++) and **VAD** (FSMN).

## Workflow

Before starting transcription, **always ask the user** the following:

1. **Audio file** — path to the recording (required)
2. **Type** — meeting, podcast, or interview? (affects defaults)
3. **Language** — what language is the recording in? (default: Chinese)
4. **Number of speakers** — how many participants? (improves diarization)
5. **Speaker names** — for podcasts: host + guest names; for meetings: attendee list
6. **Supporting files** — ask:
   > "Do you have any of the following to improve transcription accuracy?"
   > - **Attendee / guest list** — used for hotwords and speaker mapping
   > - **Meeting agenda or episode topic** — used for hotwords (terms, names)
   > - **Reference documents** (prior notes, show notes, etc.) — used to identify speakers via keyword matching after transcription
   >
   > These are optional but significantly improve speaker identification
   > and domain-specific term recognition.

**Adapt defaults based on recording type:**
- **Meeting**: ask about supporting files, default `--lang zh`
- **Podcast / interview**: default `--num-speakers 2`, always ask for
  host + guest names, suggest `--speaker-context` for host/guest roles,
  use `--lang auto` or `--lang whisper` for multilingual shows

If the user provides supporting materials:
- Extract participant names and key terms → create `hotwords.txt`
- Extract per-person context → create `speaker-context.json`
- Pass the original reference document (show notes, meeting agenda, attendee
  list, etc.) directly with `--reference` — the LLM uses all proper nouns,
  terms, and names from it to correct ASR errors
- Use all three with `--hotwords`, `--speaker-context`, and `--reference`

If no supporting files are available, proceed with `--num-speakers` only.

## Quick Start

### 1. Environment Setup

```bash
bash ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/setup_env.sh
# Or force CPU:  bash setup_env.sh cpu
```

**Critical for long meetings**: The setup script patches FunASR's spectral
clustering for O(N^2*k) performance. Without this, recordings over ~1 hour
hang for hours during speaker clustering.

### 2. Audio Preprocessing

The script automatically converts input audio to 16kHz mono and validates
that no audio is lost during conversion (detects silent truncation).

```bash
# Automatic (default: FLAC, lossless, safest for long recordings)
# Just pass the original file — the script handles conversion:
python3 transcribe_funasr.py recording.m4a --audio-format flac

# Or manually convert beforehand:
ffmpeg -i recording.m4a -ar 16000 -ac 1 -sample_fmt s16 meeting.flac
```

**Important**: Use `--audio-format flac` (the default) for recordings over
2 hours. Opus encoding can silently truncate long M4A files. The script
will abort with a clear error if truncation is detected.

| Format | 4h14m meeting | Quality | ASR impact |
|--------|--------------|---------|-----------|
| **FLAC 16-bit** | **219MB** | Lossless | **Baseline (recommended)** |
| Opus 32kbps | 55MB | Lossy | -3% sentences, risk of truncation on long files |
| WAV | 465MB | Lossless | Same as FLAC |
| Original M4A | 173MB | Source | Also works directly |

FunASR natively reads FLAC, WAV, Opus, M4A, and MP3.
**Do NOT split long recordings** — splitting breaks speaker ID consistency.

### 3. Prepare Supporting Files (Recommended)

Prepare these from the meeting invite / attendee list:

- **`hotwords.txt`** — One term per line: participant names, project names,
  domain jargon (Chinese terms work best; English terms may regress)
- **`speaker-context.json`** — Per-person keywords for LLM speaker identification

See `references/pipeline-details.md` for format details and effectiveness data.

### 4. Run Transcription

Copy the scripts to the working directory (output files written to CWD):

```bash
cp ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/{llm_utils,transcribe_funasr}.py .
```

**Prerequisites for Phase 3 (LLM cleanup):** AWS credentials with Bedrock
`InvokeModel` permission. Skip with `--skip-llm` if unavailable.

```bash
# Chinese meeting with hotwords (recommended)
python3 transcribe_funasr.py meeting.wav --lang zh --num-speakers 9 \
    --hotwords hotwords.txt

# English meeting with speaker names
python3 transcribe_funasr.py meeting.wav --lang en \
    --speakers "Alice,Bob,Carol,Dave"

# Auto-detect language (zh/en/ja/ko/yue)
python3 transcribe_funasr.py meeting.wav --lang auto --num-speakers 6

# Whisper for any language
python3 transcribe_funasr.py meeting.wav --lang whisper --num-speakers 4

# Full pipeline with all supporting files (best quality)
python3 transcribe_funasr.py episode.m4a --lang zh --num-speakers 2 \
    --hotwords hotwords.txt \
    --speakers "孟岩,李继刚" \
    --speaker-context speaker-context.json \
    --reference show-notes.md    # any supporting text: agenda, attendee list, etc.

# Use different LLM providers for cleanup (auto-detected from model ID)
# Bedrock (ARN or cross-region ID)
python3 transcribe_funasr.py meeting.wav \
    --model arn:aws:bedrock:us-west-2:123456:application-inference-profile/abc
# Anthropic Messages API
python3 transcribe_funasr.py meeting.wav --model claude-sonnet-4-6
# OpenAI-compatible API (also works with DeepSeek, vLLM, etc.)
python3 transcribe_funasr.py meeting.wav --model gpt-4o

# Raw transcription only (no LLM)
python3 transcribe_funasr.py meeting.wav --skip-llm

# Resume interrupted LLM cleanup
python3 transcribe_funasr.py meeting.wav --skip-transcribe
```

### 5. Verify Speaker Labels (Post-Processing)

If the transcript has swapped speaker labels (host↔guest, or wrong names
on meeting participants), use the standalone verification script:

```bash
cp ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/{llm_utils,verify_speakers}.py .

# Podcast: check if host/guest are swapped (dry-run)
python3 verify_speakers.py podcast_raw_transcript.json \
    --speakers "孟岩,李继刚" \
    --speaker-context speaker-context.json

# Apply the fix
python3 verify_speakers.py podcast_raw_transcript.json \
    --speakers "孟岩,李继刚" \
    --speaker-context speaker-context.json --fix

# Multi-speaker meeting: full reassignment
python3 verify_speakers.py meeting_raw_transcript.json \
    --speakers "Alice,Bob,Carol,Dave" \
    --speaker-context speaker-context.json --fix

# Then re-run LLM cleanup with corrected labels
python3 transcribe_funasr.py original.m4a --skip-transcribe --clean-cache
```

The script analyzes the first 5 minutes (configurable with `--minutes`)
using LLM to match content patterns to speaker roles. It auto-detects
podcast (2 speakers → swap detection) vs meeting (N speakers → full
reassignment).

### 6. Speaker Diarization Tips

FunASR's CAM++ may merge acoustically similar speakers. To improve:

1. **`--num-speakers N`** — Hint expected count
2. **`--hotwords`** — Include participant names (Chinese names work best)
3. **Keyword matching** — Search `*_raw_transcript.json` for unique phrases
4. **`--speaker-context`** — Provide per-person keywords for LLM splitting

## Key Flags

| Flag | Purpose |
|------|---------|
| `--lang` | `zh` (default), `zh-basic`, `en`, `auto`, `whisper` |
| `--hotwords` | Hotword file or string — biases ASR toward terms (zh only) |
| `--reference F` | Reference file (show notes, agenda, attendee list) — injected into LLM prompt for ASR correction |
| `--num-speakers N` | Expected speaker count (improves diarization) |
| `--speakers "A,B,C"` | Assign real names by first-appearance order |
| `--speaker-context F` | JSON with per-speaker keywords for LLM |
| `--audio-format` | Target conversion format: `flac` (default, lossless), `opus`, `wav` |
| `--device cpu` | Force CPU mode |
| `--batch-size N` | Adjust for memory (60 for CPU, 100 if GPU OOM) |
| `--skip-transcribe` | Resume from saved `*_raw_transcript.json` |
| `--skip-llm` | Skip LLM cleanup |
| `--skip-preprocess` | Skip audio conversion (use input file as-is) |
| `--model ID` | LLM model for cleanup (auto-detects Bedrock/Anthropic/OpenAI) |
| `--title "..."` | Output document title (default: "Meeting Transcript") |
| `--clean-cache` | Delete LLM chunk cache after completion |

## Outputs

- `<stem>-transcript.md` — Final Markdown with speaker labels
- `<stem>_raw_transcript.json` — Raw Phase 1 output (for resume/analysis)

## Additional Resources

- **`references/pipeline-details.md`** — Architecture, model specs, benchmarks,
  hotword effectiveness data, clustering patch, diarization limitations,
  supporting file preparation guide
- **`scripts/transcribe_funasr.py`** — Main transcription pipeline
- **`scripts/verify_speakers.py`** — Standalone speaker label verification & fix
- **`scripts/llm_utils.py`** — Shared LLM call infrastructure (Bedrock/Anthropic/OpenAI)
- **`scripts/setup_env.sh`** — Environment setup (venv + deps + patch)
- **`scripts/patch_clustering.py`** — Sparse eigsh patch for long meetings

## CPU-only / Low-Memory Machines

Long recordings (2+ hours) on resource-constrained machines may hit two silent
failure modes: **exec timeouts** (agent kills the process) and **OOM kills**
(kernel kills the process when RAM is exhausted).

See `references/pipeline-details.md § Running on CPU-only / Low-Memory Machines`
for detailed workarounds:
- Detach from agent timeouts with `systemd-run` or `nohup`
- Prevent OOM via swap and/or `--lang zh-basic` (lighter model)

## Podcast / Interview Transcription

The pipeline also handles podcasts and interviews. Key differences from meetings:
- Fewer speakers (2–3), usually known upfront — always provide `--num-speakers`
  and `--speakers`
- Use `--lang auto` for bilingual shows, `--lang whisper` for other languages
- Provide `--speaker-context` describing host/guest roles for better LLM cleanup

See `references/pipeline-details.md § Podcast Transcription` for recommended
settings and examples.
