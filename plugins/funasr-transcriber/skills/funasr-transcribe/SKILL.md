---
name: funasr-transcribe
version: 1.4.0
description: >
  This skill should be used when the user explicitly asks to "transcribe a meeting",
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
  Works on both GPU and CPU.
metadata:
  openclaw:
    requires:
      bins: ["python3", "ffmpeg"]
    env_vars:
      - name: AWS_REGION
        required: false
        description: "AWS region for Bedrock LLM cleanup (default: us-west-2). Bedrock uses the standard AWS credential chain (IAM role, SSO, ~/.aws/credentials, env vars) — no explicit keys needed."
      - name: ANTHROPIC_API_KEY
        required: false
        description: "API key for Anthropic Claude LLM cleanup"
      - name: OPENAI_API_KEY
        required: false
        description: "API key for OpenAI-compatible LLM cleanup"
      - name: OPENAI_BASE_URL
        required: false
        description: "Base URL for OpenAI-compatible API (vLLM, Ollama, etc.)"
    emoji: "🎙️"
    homepage: "https://github.com/zxkane/audio-transcriber-funasr"
---

# FunASR Meeting & Podcast Transcription

Transcribe multi-speaker audio into structured Markdown with automatic
speaker diarization, hotword biasing, and optional LLM cleanup.

All scripts run directly from the plugin directory — no copying needed.
Define this shorthand at the start of every session:

```bash
SCRIPTS=${CLAUDE_PLUGIN_ROOT}/skills/funasr-transcribe/scripts
```

## Supported Languages

| `--lang` | Model | Languages | Hotword |
|----------|-------|-----------|---------|
| `zh` (default) | SeACo-Paraformer | Chinese (CER 1.95%) | Yes |
| `zh-basic` | Paraformer-large | Chinese | No |
| `en` | Paraformer-en | English | No |
| `auto` | SenseVoiceSmall | Auto-detect: zh/en/ja/ko/yue | No |
| `whisper` | Whisper-large-v3-turbo | 99 languages | No |

All presets include **speaker diarization** (CAM++) and **VAD** (FSMN).

> **Diarization caveat:** `auto` and `whisper` do not output per-sentence timestamps,
> so speaker diarization does not work with these presets. Use `zh`, `zh-basic`, or
> `en` when speaker identification is needed (e.g., podcasts, meetings).

## Workflow

Before starting transcription, **always ask the user**:

1. **Audio file** — path to the recording (required)
2. **Type** — meeting, podcast, or interview? (affects defaults)
3. **Language** — what language is spoken? (default: Chinese)
4. **Number of speakers** — how many participants? (improves diarization)
5. **Speaker names** — for podcasts: host + guest names; for meetings: attendee list
6. **Supporting files** — ask:
   > "Do you have any of the following to improve accuracy?"
   > - **Attendee / guest list** — for hotwords and speaker mapping
   > - **Meeting agenda or episode topic** — for hotwords (terms, names)
   > - **Reference documents** (show notes, prior notes) — for speaker identification and ASR correction

**Adapt defaults by recording type:**
- **Meeting**: default `--lang zh`, ask about supporting files
- **Podcast / interview**: default `--lang zh`, `--num-speakers 2`, always ask for
  host + guest names, suggest `--speaker-context` for roles
  (do NOT use `--lang auto` — it lacks timestamps for speaker diarization)

If the user provides supporting materials:
- Extract participant names and key terms → create `hotwords.txt`
- Extract per-person context → create `speaker-context.json`
- Pass original reference document with `--reference`
- Use all three together for best results

## Quick Start

### 1. Environment Setup

```bash
AUTO_YES=1 bash $SCRIPTS/setup_env.sh
# Or force CPU:  AUTO_YES=1 bash $SCRIPTS/setup_env.sh cpu
```

The setup script patches FunASR's spectral clustering for O(N²·k) performance.
Without this, recordings over ~1 hour hang for hours during speaker clustering.

### 2. Run Transcription

Output files are written to the current working directory.

**LLM cleanup (Phase 3) is opt-in.** By default, transcription runs locally
without contacting any external service. To enable LLM-powered ASR correction
and speaker name refinement, pass `--model <model-id>`. Use LLM cleanup when:
- The raw transcript has many ASR errors (names, technical terms)
- You need polished, publication-ready output
- Speaker names need to be refined from context

> **⚠️ Data Privacy:** When LLM cleanup is enabled via `--model`, transcript
> excerpts are sent to external LLM providers (AWS Bedrock, Anthropic, or
> OpenAI depending on the model ID). Use `--skip-llm` or omit `--model` to
> keep all data local. For Bedrock, boto3 uses the standard AWS credential
> chain (IAM role, SSO, `~/.aws/credentials`, env vars).

```bash
# Chinese meeting with hotwords (local-only, no LLM)
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --lang zh --num-speakers 9 --hotwords hotwords.txt

# English meeting with speaker names
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --lang en --speakers "Alice,Bob,Carol,Dave"

# Auto-detect language (zh/en/ja/ko/yue)
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --lang auto --num-speakers 6

# Whisper for any language
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --lang whisper --num-speakers 4

# Enable LLM cleanup for polished output (requires --model)
# Bedrock (uses AWS credential chain: IAM role, SSO, ~/.aws/credentials)
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --lang zh --num-speakers 9 --hotwords hotwords.txt \
    --model us.anthropic.claude-sonnet-4-6

# Anthropic API (requires ANTHROPIC_API_KEY env var)
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --model claude-sonnet-4-6

# OpenAI-compatible API (requires OPENAI_API_KEY env var)
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --model gpt-4o

# Full pipeline with all supporting files + LLM (best quality)
python3 $SCRIPTS/transcribe_funasr.py episode.m4a \
    --lang zh --num-speakers 2 \
    --hotwords hotwords.txt \
    --speakers "关羽,张飞" \
    --speaker-context speaker-context.json \
    --reference show-notes.md \
    --model us.anthropic.claude-sonnet-4-6

# Resume interrupted LLM cleanup
python3 $SCRIPTS/transcribe_funasr.py meeting.wav \
    --skip-transcribe --model us.anthropic.claude-sonnet-4-6
```

### 3. Verify Speaker Labels

If the transcript has swapped speaker labels (common with podcasts),
the verification script can detect and fix mismatches using LLM analysis:

```bash
# Dry-run: check if host/guest are swapped
python3 $SCRIPTS/verify_speakers.py podcast_raw_transcript.json \
    --speakers "关羽,张飞" \
    --speaker-context speaker-context.json

# Apply the fix
python3 $SCRIPTS/verify_speakers.py podcast_raw_transcript.json \
    --speakers "关羽,张飞" \
    --speaker-context speaker-context.json --fix

# Multi-speaker meeting: full reassignment
python3 $SCRIPTS/verify_speakers.py meeting_raw_transcript.json \
    --speakers "Alice,Bob,Carol,Dave" \
    --speaker-context speaker-context.json --fix

# Then regenerate the markdown with corrected labels
python3 $SCRIPTS/transcribe_funasr.py original.m4a \
    --skip-transcribe --clean-cache
```

The script analyzes the first 5 minutes (configurable with `--minutes`)
and auto-detects podcast (2 speakers, swap detection) vs meeting
(N speakers, full reassignment).

## Audio Preprocessing

The script automatically converts input audio to 16kHz mono FLAC and
validates that no audio is lost (detects silent truncation).

| Format | 4h14m meeting | Quality | Recommendation |
|--------|--------------|---------|----------------|
| **FLAC** | **219MB** | Lossless | **Default, safest** |
| Opus | 55MB | Lossy | Risk of truncation on long files |
| WAV | 465MB | Lossless | Works but larger |
| Original M4A | 173MB | Source | Also works directly |

**Do NOT split long recordings** — splitting breaks speaker ID consistency.

## Key Flags

| Flag | Purpose |
|------|---------|
| `--lang` | `zh` (default), `zh-basic`, `en`, `auto`, `whisper` |
| `--hotwords` | Hotword file or string — biases ASR (zh only) |
| `--reference F` | Reference file for LLM ASR correction |
| `--num-speakers N` | Expected speaker count (improves diarization) |
| `--speakers "A,B,C"` | Assign real names by first-appearance order |
| `--speaker-context F` | JSON with per-speaker roles for LLM |
| `--audio-format` | `flac` (default), `opus`, `wav` |
| `--device cpu` | Force CPU mode |
| `--batch-size N` | Adjust for memory (60 for CPU, 100 if GPU OOM) |
| `--skip-transcribe` | Resume from saved `*_raw_transcript.json` |
| `--skip-llm` | Skip LLM cleanup (default when `--model` is omitted) |
| `--model ID` | Enable LLM cleanup with this model (auto-detects Bedrock/Anthropic/OpenAI) |
| `--title "..."` | Output document title |
| `--clean-cache` | Delete LLM chunk cache after completion |
| `--output PATH` | Custom output file path |
| `--model-cache-dir` | ModelScope model cache directory (~3GB, default: `~/.cache/modelscope/`) |

## Outputs

- `<stem>-transcript.md` — Final Markdown with speaker labels and timestamps
- `<stem>_raw_transcript.json` — Raw Phase 1 output (for resume/analysis)

## Speaker Diarization Tips

FunASR's CAM++ may merge acoustically similar speakers. To improve:

1. **`--num-speakers N`** — Hint expected count
2. **`--hotwords`** — Include participant names (Chinese names work best)
3. **`--speaker-context`** — Provide per-person keywords for LLM splitting
4. **Keyword matching** — Search `*_raw_transcript.json` for unique phrases

## CPU-only / Low-Memory Machines

Long recordings on resource-constrained machines may hit exec timeouts
or OOM kills. See `references/pipeline-details.md` for workarounds:
- Detach from agent timeouts with `systemd-run` or `nohup`
- Prevent OOM via swap and/or `--lang zh-basic` (lighter model)

## Additional Resources

- **`references/pipeline-details.md`** — Architecture, model specs, benchmarks,
  speaker role verification, hotword effectiveness, clustering patch
- **`scripts/transcribe_funasr.py`** — Main transcription pipeline
- **`scripts/verify_speakers.py`** — Speaker label verification & fix
- **`scripts/llm_utils.py`** — Shared LLM infrastructure (Bedrock/Anthropic/OpenAI)
- **`scripts/setup_env.sh`** — Environment setup (venv + deps + patch)
