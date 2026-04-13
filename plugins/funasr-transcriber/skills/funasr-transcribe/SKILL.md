---
name: funasr-transcribe
version: 1.1.0
description: >
  This skill should be used when the user asks to "transcribe a meeting",
  "transcribe audio", "transcribe a meeting recording",
  "convert audio to text", "generate meeting minutes from audio",
  "do speech-to-text", "transcribe with speaker diarization",
  "identify speakers in audio", "transcribe Chinese audio",
  "transcribe English audio", "transcribe Japanese audio",
  "multi-speaker transcription", or mentions FunASR, Paraformer,
  SenseVoice, meeting transcription, or speaker diarization.
  Supports multi-speaker meeting transcription in Chinese, English,
  Japanese, Korean, and Cantonese with automatic speaker diarization.
  Works on both GPU and CPU.
---

# FunASR Meeting Transcription

Transcribe multi-speaker meeting recordings into structured Markdown
with automatic speaker diarization and optional LLM cleanup, using the
open-source FunASR pipeline.

**Optimized for meetings**: handles arbitrarily long recordings
(4+ hours tested), separates speakers via CAM++ diarization,
merges consecutive utterances, and maps speaker IDs to real names.

## Supported Languages

| `--lang` | Model | Languages |
|----------|-------|-----------|
| `zh` (default) | Paraformer-large | Chinese (CER 1.95% SOTA) |
| `en` | Paraformer-en | English |
| `auto` | SenseVoiceSmall | Auto-detect: Chinese, English, Japanese, Korean, Cantonese |

All presets include **speaker diarization** (CAM++) and **VAD** (FSMN).

## Quick Start

### 1. Environment Setup

Run the setup script. It auto-detects GPU/CUDA; falls back to CPU-only.

```bash
bash ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/setup_env.sh
# Or force CPU:  bash setup_env.sh cpu
```

**Critical for long meetings**: The setup script automatically patches
FunASR's spectral clustering for O(N^2*k) performance. Without this,
recordings over ~1 hour hang for hours during speaker clustering.

### 2. Audio Preprocessing

Convert to 16kHz mono WAV:

```bash
ffmpeg -i recording.m4a -ar 16000 -ac 1 meeting.wav
```

**Do NOT split long recordings** — splitting breaks speaker ID
consistency across segments. FunASR handles arbitrary length.

### 3. Run Transcription

Copy the script to the working directory (output files are written
relative to CWD):

```bash
cp ${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts/transcribe_funasr.py .
```

**Prerequisites for LLM cleanup (Phase 3):** AWS credentials with
Bedrock `InvokeModel` permission. Skip with `--skip-llm` if unavailable.

```bash
# Chinese meeting, 9 speakers
python3 transcribe_funasr.py meeting.wav --lang zh --num-speakers 9

# English meeting, with real names
python3 transcribe_funasr.py meeting.wav --lang en --speakers "Alice,Bob,Carol,Dave"

# Auto-detect language (zh/en/ja/ko/yue)
python3 transcribe_funasr.py meeting.wav --lang auto --num-speakers 6

# CPU mode (auto-detected or forced)
python3 transcribe_funasr.py meeting.wav --lang zh --device cpu

# Raw transcription only (no LLM cleanup)
python3 transcribe_funasr.py meeting.wav --skip-llm

# Resume interrupted LLM cleanup
python3 transcribe_funasr.py meeting.wav --skip-transcribe
```

### 4. Speaker Identification (Optional)

FunASR's CAM++ may merge acoustically similar speakers. To improve:

1. **Provide `--num-speakers N`** to hint the expected count
2. **Keyword matching**: Search `*_raw_transcript.json` for unique
   phrases from reference docs (agendas, attendee notes) to map
   speaker IDs to real names
3. **LLM-assisted**: Provide `--speaker-context context.json` with
   per-person keywords for LLM to split merged speakers (~73% success)

Speaker context JSON format:

```json
{
  "Alice": "Discussed Q1 revenue, mentioned Chicago office",
  "Bob": "Presented CI/CD pipeline, uses Terraform"
}
```

## Pipeline Phases

| Phase | What | Skippable |
|-------|------|-----------|
| **Phase 1**: FunASR ASR | Language-specific model + CAM++ diarization | `--skip-transcribe` |
| **Phase 2**: Post-process | Merge consecutive utterances, map speaker names | No |
| **Phase 3**: LLM cleanup | Bedrock Claude cleans fillers, fixes ASR errors | `--skip-llm` |

## GPU vs CPU

| | GPU | CPU |
|--|-----|-----|
| Batch size | 300 (default) | 60 (auto-adjusted) |
| 4h meeting transcription | ~3 min | ~30-60 min |
| Speaker clustering (patched) | ~10s | ~2-5 min |

CPU mode is fully functional with no quality difference.
Device is auto-detected; override with `--device cpu` or `--device cuda:0`.

## Key Flags

| Flag | Purpose |
|------|---------|
| `--lang {zh,en,auto}` | Language preset (default: zh) |
| `--num-speakers N` | Expected speaker count (improves diarization) |
| `--speakers "A,B,C"` | Assign real names by first-appearance order |
| `--device cpu` | Force CPU mode |
| `--batch-size N` | Adjust for memory (lower = less RAM/VRAM) |
| `--skip-transcribe` | Resume from saved `*_raw_transcript.json` |
| `--skip-llm` | Output raw transcript without LLM polishing |
| `--speaker-context F` | JSON with per-speaker keywords for LLM |
| `--bedrock-model ID` | Override LLM model (default: `us.anthropic.claude-sonnet-4-6`) |
| `--bedrock-region R` | Override Bedrock region (default: `us-west-2`) |
| `--clean-cache` | Delete LLM chunk cache after completion |

## Outputs

- `<stem>-transcript.md` — Final Markdown transcript with speaker labels
- `<stem>_raw_transcript.json` — Raw Phase 1 output (for resume/analysis)

## Additional Resources

- **`references/pipeline-details.md`** — Architecture, model specs,
  benchmarks, clustering patch, diarization limitations
- **`scripts/transcribe_funasr.py`** — Main transcription pipeline
- **`scripts/setup_env.sh`** — Environment setup (venv + deps + patch)
- **`scripts/patch_clustering.py`** — Sparse eigsh patch for long meetings
