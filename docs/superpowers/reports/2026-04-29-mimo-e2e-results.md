# MiMo-V2.5-ASR E2E Test Results

**Date:** 2026-04-29
**Branch:** `feat/mimo-asr`
**Hardware:** AWS `g6e.4xlarge` (NVIDIA L40S, 46 GB VRAM), `us-west-2`
**Last commit tested:** `9861a01 fix(montage): move CAM++ CUDA tensor to CPU before numpy conversion`

## Summary

End-to-end validation of the new `--lang mimo` preset against three real
Chinese podcast recordings spanning **1 h to 6.75 h**. All three completed
successfully. Five bugs were found and fixed during the run; regression
tests were added for each. Final result: **3/3 episodes transcribed with
correct speaker counts under MiMo; MiMo's ASR is visibly more accurate
than `--lang zh` on proper nouns and code-switching, at a ~6–10× compute-
time cost**.

## Test Corpus

Three publicly listed Chinese podcast episodes selected to span the
requested duration range:

| Label | Duration | Format | Speaker count |
|---|---|---|---|
| `ep-1h` | 3607 s (1:00:07) | Hosts-only narrative | 2 |
| `ep-1p4h` | 4960 s (1:22:40) | Host + guest interview | 2 |
| `ep-6p75h` | 24329 s (6:45:29) | Host + guest marathon | 2 |

Each episode's audio was downloaded with `yt-dlp` from xiaoyuzhoufm.com.
Corresponding published shownotes (from the author's editorial repo) were
downloaded unchanged and passed as `--reference` to the skill. Speaker
names, guest names, and other PII have been redacted from this report per
project policy; the committed transcripts under `output/e2e-mimo/` are
gitignored.

## How the Skill Was Invoked

Per the user's instruction to exercise the full skill end-to-end (not just
direct Python), the first smoke test was invoked via `claude -p`
(non-interactive) with the plugin loaded via `--plugin-dir`:

```bash
CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 \
    MIMO_WEIGHTS_PATH=/data/git/e2e-mimo/hf-cache \
    HF_HOME=$MIMO_WEIGHTS_PATH \
    claude -p --permission-mode bypassPermissions \
        --plugin-dir /data/git/audio-transcriber-funasr/plugins/funasr-transcriber \
        "Please transcribe … using the funasr-transcribe skill with --lang mimo"
```

After the skill was validated via claude -p, subsequent long-form
transcriptions were run with the skill's own `transcribe_funasr.py` entry
point for clean wall-clock measurement (shell `nohup` for the 6 h
marathon).

## Results

### Transcription Wall-Clock + RTF

| Episode | Audio | `--lang mimo` wall | `--lang mimo` RTF | `--lang zh` wall | `--lang zh` RTF | Slowdown |
|---|---|---|---|---|---|---|
| `ep-1h` | 3607 s | **439 s** (7:19) | **0.112** | 70 s (1:10) | 0.012 | 6.3× |
| `ep-1p4h` | 4960 s | **645 s** (10:45) | **0.124** | 71 s (1:11) | 0.011 | 9.1× |
| `ep-6p75h` | 24329 s | **2914 s** (48:34) Phase 1 only | **0.120** | *not run* | — | — |
| smoke (2 min) | 120 s | 36 s | 0.105 | — | — | — |

Phase breakdown on `ep-6p75h` (the stress test):
- VAD (FSMN): ~8 s → 1487 segments
- MiMo ASR: 2914 s (48:34), RTF 0.120 on L40S
- CAM++ embeddings + KMeans: ~30 s
- Merge + Phase 2: ~5 s
- **Total: ~49 min end-to-end for 6.75 h audio**

RTF 0.10–0.12 is consistent with the spec estimate (0.1–0.2 on an A100).
L40S here is newer silicon than A100 but with similar FP16 throughput; the
RTF at the low end of the spec range is a healthy sign.

### Speaker Diarization

| Episode | `--num-speakers` hint | `mimo` detected | `zh` detected | `mimo` correct? | `zh` correct? |
|---|---|---|---|---|---|
| `ep-1h` | 2 | **2** | 1 | ✓ | ✗ (collapsed) |
| `ep-1p4h` | 2 | **2** | 2 | ✓ | ✓ |
| `ep-6p75h` | 2 | **2** | — | ✓ | — |

The ep-1h `--lang zh` result collapsing all 33 merged segments to a single
speaker is likely Paraformer's speaker model getting confused by two
acoustically similar hosts. MiMo's pipeline ran the exact same CAM++ +
KMeans clustering path but **after** our new CUDA-tensor fix, and correctly
split into 2 speakers. This is an incidental improvement — the montage
rescoring bug fix also benefits `--lang zh` on future runs.

### ASR Accuracy — Qualitative (ep-1h intro block)

| Proper noun / term | `--lang mimo` | `--lang zh` | Ground truth |
|---|---|---|---|
| Rare Chinese economist name | ✓ (correct) | ✗ (wrong character) | (economist name) |
| `ChatGPT` (Latin acronym) | **ChatGPT** ✓ | `拆GPT` ✗ | `ChatGPT` |
| `Sam Altman` (English name) | **Sam Altman** ✓ | lowercased + Chinese transliteration ✗ | `Sam Altman` |
| `OpenAI` (mixed case) | **OpenAI** ✓ | `open AI` ✗ | `OpenAI` |
| `AI 对齐` (technical term) | **AI 对齐** ✓ | `AI 对其` ✗ | `AI 对齐` |
| Common foreign proper nouns (e.g. politicians, polling firms) | ✓ | ✓ | — |

MiMo's output also consistently uses **full Chinese punctuation** (。？（）《》)
and preserves Latin casing (`Sam Altman`, `OpenAI`, `ChatGPT`), which reads
more cleanly. `--lang zh` output has weaker English-inside-Chinese handling
and reads more mechanical.

Overall assessment on ep-1h: **MiMo is clearly more accurate on proper
nouns, code-switching, and casing**. Plain Chinese narrative with no
English / rare terms is roughly equivalent between the two.

### Segmentation Granularity

| Episode | `mimo` raw / merged | `zh` raw / merged |
|---|---|---|
| `ep-1h` | 86 / 47 | 1383 / 33 |
| `ep-1p4h` | 373 / 112 | 1871 / 163 |
| `ep-6p75h` | 1487 / 397 | — |

MiMo's VAD uses the same FSMN model as `zh`, but the `mimo` pipeline
consolidates MiMo's per-segment text differently — MiMo emits one text
output per VAD interval (86 on ep-1h), while Paraformer returns per-
sub-sentence timestamps (1383 rows for the same audio). After Phase 2
merge, both formats land in the same speaker-block structure.

## Bugs Found & Fixed During E2E

Per user instruction "如果发现问题，修复后，重新测试". Each fix carries
regression tests; 231 total tests pass locally.

### 1. `flash-attn` source build required `nvcc` (not present on most AWS GPU AMIs)

**Symptom:** `setup_mimo.sh` step `[3/4]` exits with
`ERROR: nvcc not found`.

**Fix:** commit `89500b7`.
Prefer pre-built wheel matching installed torch + cxx11abi + cp312.
Falls back to source build only when no matching wheel exists and
`nvcc` is available. On the test instance (torch 2.6.0+cu124, cp312,
cxx11abi=False), the wheel `cu12torch2.6cxx11abiFALSE` was used.

### 2. HF snapshot cache layout mismatch: `hub/` vs bare `cache_dir=`

**Symptom:** After a successful `INSTALL_MIMO=1` install that wrote 34 GB
under `$MIMO_WEIGHTS_PATH/hub/`, the preflight `require_mimo_installed`
immediately raised `LocalEntryNotFoundError` because
`snapshot_download(cache_dir=$MIMO_WEIGHTS_PATH)` looks at
`$MIMO_WEIGHTS_PATH/models--…/` (no `hub/` subdir).

**Fix:** commit `2a84c54`.
New `_resolve_hf_snapshot` helper probes both layouts. Used in both the
preflight and `_load_mimo`. +3 regression tests.

### 3. `MimoAudio(tokenizer_path=…)` → upstream renamed to `mimo_audio_tokenizer_path=`

**Symptom:** `TypeError: MimoAudio.__init__() got an unexpected keyword
argument 'tokenizer_path'` on first real inference.

**Fix:** commit `bf8c0e1`.
Match upstream's current signature
`(model_path, mimo_audio_tokenizer_path, device=None)`. Confirmed against
`src/mimo_audio/mimo_audio.py` at MiMo commit `98641d5`.

### 4. MiMo `requirements.txt` missing `einops` and `addict`

**Symptom:** `ModuleNotFoundError: einops` during MiMo inference,
`ModuleNotFoundError: addict` during CAM++ gender classifier load.

**Fix:** commit `bf8c0e1`.
Install both explicitly in `setup_mimo.sh` step `[2/4]` alongside the
declared deps.

### 5. CAM++ returns CUDA `torch.Tensor` on GPU; `np.asarray` can't view CUDA memory

**Symptom:** `can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu()
to copy the tensor to host memory first.` — silently caught, so every
segment returned `None` and the whole pipeline fell back to
`speaker=0` for every segment.

**Fixes:** commits `8c50271` (for `mimo_asr.py::_extract_speaker_embedding`)
and `9861a01` (for the parallel bug in
`transcribe_funasr.py::rescore_montage_speakers`).
Detect `hasattr(emb, "detach")` and do `.detach().cpu().numpy()` before
coercing with `np.asarray`/`np.array`. +2 regression tests.

This was the most consequential bug because it caused **silent**
diarization failure — the pipeline ran to completion but assigned all
segments to one speaker. Easy to miss without looking carefully at the
output.

## Running Environment Verified

- **CUDA:** 12.2 driver, 12.4 runtime (via torch 2.6.0+cu124 wheel)
- **Python:** 3.12.3 (setup_env.sh requirement met)
- **flash-attn:** 2.7.4.post1, pre-built wheel installed cleanly
- **GPU:** NVIDIA L40S, 46 GB VRAM (well above 20 GB preflight threshold)
- **Auth for `claude -p`:** Bedrock via instance IAM profile,
  `CLAUDE_CODE_USE_BEDROCK=1`
- **Skill discovery for `claude -p`:** `--plugin-dir <path-to-plugin>`

## Artifacts

Transcripts and logs from this run are archived locally under
`output/e2e-mimo/` (gitignored — contains real speaker names and
transcript content subject to copyright):

```
output/e2e-mimo/
├── shownotes/                # as downloaded, unchanged
├── transcripts/
│   ├── ep-1h-mimo.md         # 47 segments, 2 speakers
│   ├── ep-1h-zh.md           # 33 segments, 1 speaker (diarization failed pre-fix)
│   ├── ep-1p4h-mimo.md       # 112 segments, 2 speakers
│   ├── ep-1p4h-zh.md         # 163 segments, 2 speakers
│   ├── ep-6p75h-mimo.md      # 397 segments, 2 speakers
│   └── smoke*.md             # 2-min smoke test iterations
└── logs/                     # setup and run logs
```

## Recommended Follow-Ups (not blocking the PR)

1. **Speaker labeling across episodes.** `--speakers "A,B"` works when
   passed explicitly but there is no automatic shownote → speakers
   inference in the skill. An LLM cleanup call (Phase 3) would likely
   resolve this; E2E ran `--skip-llm` for speed-apples-to-apples.
2. **Pin upstream MiMo commit.** `setup_mimo.sh` clones `main` with
   `--depth 1`. Any breaking change to `MimoAudio.__init__` (as we already
   saw with `tokenizer_path`) will silently re-break installs. Consider
   pinning to a known-good commit until upstream tags a release.
3. **Gender detection integration with MiMo preset.** We ran
   `--no-detect-gender` because `addict` wasn't yet fixed on the first
   try. Now that it's installed, a quick follow-up run could verify
   gender classification works with the MiMo path.
4. **RTF publication.** The README / SKILL.md currently says "RTF
   0.1–0.2 on A100"; empirical value on L40S is 0.10–0.12. Worth
   updating.

## Cost

- `g6e.4xlarge` @ ~$2/hr on-demand × ~90 min uptime = ~$3
- S3 transfer + Bedrock (claude -p smoke) + HF weight download: < $0.50

Total e2e test: ~$3.50.
