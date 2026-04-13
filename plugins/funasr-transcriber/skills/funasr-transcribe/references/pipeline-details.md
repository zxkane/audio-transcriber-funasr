# FunASR Meeting Transcription Pipeline — Technical Details

## Architecture

```
Audio File (.m4a/.mp3/.wav)
  │
  ├─ [ffmpeg] ──► 16kHz mono WAV
  │
  ├─ [Phase 1: FunASR] ──► raw_transcript.json
  │   ├─ FSMN-VAD: segment speech vs silence
  │   ├─ ASR model (language-dependent, see below)
  │   ├─ Punctuation restoration (model-dependent)
  │   └─ CAM++: speaker embeddings → spectral clustering
  │
  ├─ [Phase 2: Post-process]
  │   ├─ Merge consecutive same-speaker utterances (<2s gap)
  │   └─ Map speaker IDs to names (if provided)
  │
  └─ [Phase 3: LLM cleanup] ──► transcript.md
      ├─ Remove fillers (um, uh, 嗯, 啊, etc.)
      ├─ Fix ASR errors (homophones, context-based)
      ├─ Polish grammar while preserving meaning
      └─ (Optional) Identify merged speakers via context
```

## Language Presets & Models

### `--lang zh` (Chinese, default)

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 220M | Chinese ASR (CER 1.95%) |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 0.4M | Voice activity detection |
| Punctuation | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 290M | Punctuation restoration |
| Speaker | `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2M | Speaker diarization |

### `--lang en` (English)

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020` | 220M | English ASR |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 0.4M | Voice activity detection |
| Punctuation | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 290M | Punctuation restoration |
| Speaker | `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2M | Speaker diarization |

### `--lang auto` (Auto-detect: zh/en/ja/ko/yue)

| Component | Model ID | Params | Role |
|-----------|----------|--------|------|
| ASR | `iic/SenseVoiceSmall` | 234M | Multi-language ASR with auto language detection |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 0.4M | Voice activity detection |
| Speaker | `iic/speech_campplus_sv_zh-cn_16k-common` | 7.2M | Speaker diarization |

SenseVoiceSmall includes built-in punctuation and supports emotion detection.

All models are auto-downloaded from ModelScope on first run (~2GB total).

## Performance Benchmarks

Tested on a 4h14m, 9-speaker Chinese meeting recording:

| Metric | GPU (L40S 46GB) | CPU (estimated) |
|--------|-----------------|-----------------|
| Model load | 14s | ~30s |
| Transcription | 169s | ~30-60 min |
| Speaker clustering | ~10s (patched) | ~2-5 min (patched) |
| LLM cleanup (17 chunks) | ~35 min | ~35 min (network-bound) |
| Total | ~38 min | ~70-100 min |

**Without the clustering patch**, the original `scipy.linalg.eigh()` on the full Laplacian
matrix was O(N^3) and took **10+ hours** on this recording. The patch reduces it to O(N^2*k)
via `scipy.sparse.linalg.eigsh()`.

## Clustering Patch (Critical for Long Audio)

FunASR's `SpectralCluster.get_spec_embs()` uses `scipy.linalg.eigh(L)` which computes
ALL eigenvalues of the NxN Laplacian. For a 4-hour recording, N can be 6000+, making
this O(N^3) operation take hours.

The patch (`scripts/patch_clustering.py`) replaces this with:
- `scipy.sparse.linalg.eigsh(L_sparse, k=num_speakers, which='SM')` — only computes
  the k smallest eigenvalues needed, reducing complexity to O(N^2 * k)
- Vectorized `p_pruning()` — replaces Python loop with numpy broadcasting

**Always run the patch before processing audio longer than ~1 hour.**

## Speaker Diarization Limitations

FunASR's CAM++ speaker diarization may merge acoustically similar speakers into one ID.
In the tested 9-person meeting, only 7 unique IDs were detected (two pairs merged).

Workarounds:
1. **Provide `--num-speakers N`** to hint expected count (uses `preset_spk_num`)
2. **Post-hoc keyword matching**: use reference documents (meeting agendas, attendee notes)
   to identify which speaker ID maps to which person
3. **LLM-assisted splitting**: provide `--speaker-context` with per-person keywords;
   the LLM can then split merged speakers when context is clear (~73% success rate)

## Audio Preprocessing

FunASR works best with:
- **Format**: WAV (PCM 16-bit)
- **Sample rate**: 16kHz
- **Channels**: mono

Convert with ffmpeg:
```bash
ffmpeg -i recording.m4a -ar 16000 -ac 1 meeting.wav
```

For long recordings, do NOT split the audio — FunASR handles arbitrarily long files
and splitting breaks speaker consistency across segments.

## Resume / Checkpoint Support

The pipeline supports resuming interrupted runs:
- **Phase 1 output**: `<stem>_raw_transcript.json` — use `--skip-transcribe` to skip ASR
- **Phase 3 cache**: `<stem>_llm_cache/chunk_NNN.txt` — already-cleaned chunks are reused

## Speaker Context JSON Format

The `--speaker-context` file helps the LLM identify speakers and fix ASR errors:

```json
{
  "Alice": "Discussed Q1 revenue targets, mentioned Chicago office relocation",
  "Bob": "Presented the new CI/CD pipeline, uses Terraform and ArgoCD",
  "Carol": "HR updates, mentioned hiring freeze and new PTO policy"
}
```

The context is injected into the LLM system prompt for each cleanup chunk.
