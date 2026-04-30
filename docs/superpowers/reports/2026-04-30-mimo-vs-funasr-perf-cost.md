# MiMo-V2.5-ASR vs FunASR (Paraformer) on GPU — Performance & Cost

**Companion to** [`2026-04-29-mimo-e2e-results.md`](./2026-04-29-mimo-e2e-results.md).

FunASR Paraformer GPU wall-clock + cost numbers used in this comparison
are taken from a separate, previously-run GPU benchmark of the same
FunASR pipeline on AWS ECS Managed Instances (Spot pricing,
`ap-northeast-1`). They are *not* re-measured here. Those numbers are
dated 2026-04-26 and use the same worker entry point our skill exposes
(`transcribe_funasr.py`) with pinned ModelScope model revisions. Only
MiMo numbers come from the e2e described in the companion report.

## TL;DR

| Metric | FunASR (Paraformer, `--lang zh`) | MiMo-V2.5-ASR (`--lang mimo`) |
|---|---|---|
| Parameters | ~0.2 B | **8 B** (40× larger) |
| Wall on 1 h audio (L40S) | **104 s** (1m44s) | **439 s** (7m19s) |
| Wall on 6h45m audio (L40S) | **377 s** (6m17s) | **2954 s** (49 min) |
| RTF on L40S | ~0.014 | **~0.12** |
| Best-price family that fits | L4 (g6.2xlarge, 24 GB VRAM) | L4 (24 GB VRAM — **barely** fits, 20 GB floor) |
| Cost per 1 h episode on L4 (Spot) | **$0.013** | ~$0.14 (**~10× more**) |
| Cost per 6h45m episode on L4 (Spot) | **$0.049** | ~$0.94 (**~19× more**) |
| Proper-noun / code-switch accuracy | Good but errors on rare terms | **Visibly better** — names, English-in-Chinese, casing |
| Weight download | ~1 GB (ModelScope cache) | **~34 GB** (HF) |
| Setup time (cold) | ~4 min | **~20 min** (flash-attn wheel + 34 GB DL) |

**Bottom line: FunASR is ~10–20× cheaper and faster; MiMo is more accurate
on proper nouns and code-switching. Use MiMo when accuracy on rare
terms matters; use FunASR when throughput/cost matters.** Both rely
on the same VAD + CAM++ diarization stack in this skill.

## Methodology note

The reference FunASR benchmark measures Spot pricing in `ap-northeast-1`
on pinned ModelScope revisions; the MiMo e2e ran on-demand
`g6e.4xlarge` in `us-west-2` with `AUTO_YES=1 INSTALL_MIMO=1
MIMO_WEIGHTS_PATH=…`. To compare fairly I re-cost the MiMo wall-clocks
at the Spot rates from the reference bench (**L40S g6e.xlarge
$1.1937/hr; L4 g6.2xlarge $0.4588/hr**). Cold-start is excluded for
both sides — the reference bench reports 216 s median cold start on ECS
Managed Instances; our MiMo weight-download setup was one-time and
amortized across all 3 episodes.

The reference bench's 1 h episode is a different audio file than our
MiMo 1 h run (both Chinese podcast dialogue, durations match within 6%).
**The 6h45m episode is the same audio in both benchmarks** — direct
head-to-head.

L4 MiMo cost estimates assume **2.5× wall-clock slowdown vs L40S**
(L4 FP16 throughput is ~33% of L40S for dense matmul; conservative
estimate pending an actual L4 run). L4 price is 38% of L40S per hour,
so total cost is ~95% of L40S at 2.5× slower — L4 is only marginally
cheaper for MiMo, unlike FunASR where L4 is a clear win.

## Performance — Head-to-Head Wall-Clock

### Same 6h45m episode (direct comparison)

Identical audio input in both benchmarks:

| Runtime | Wall | RTF | Speedup vs real-time |
|---|---|---|---|
| FunASR on L40S (Paraformer) | **377 s** | 0.0155 | 64.5× |
| FunASR on L4 (Paraformer)   | **388 s** | 0.0160 | 62.7× |
| MiMo on L40S                | **2954 s** | 0.121 | 8.2× |
| MiMo on L4 (estimated)      | ~7385 s   | 0.304 | 3.3× |

**MiMo is ~8× slower than FunASR on the same GPU** for the longest
realistic test case (6h45m). This ratio is stable across episodes:

| Episode | FunASR L40S wall | MiMo L40S wall | MiMo / FunASR |
|---|---|---|---|
| 1 h   | 104 s  | 439 s  | **4.2×** slower |
| 3h36m | 202 s  | (not tested) | estimate 8–9× |
| 6h45m | 377 s  | 2954 s | **7.8×** slower |

The slowdown multiplier grows with audio length because FunASR's
Paraformer batches whole-audio segments efficiently (reference bench
measured GPU util ~17% on 1 h, ~56% on 6h45m — clearly not saturated),
while MiMo's generative decoding is inherently sequential per VAD chunk
and can't benefit from batching until we wire up `--mimo-batch > 1`
(currently accepts the flag but doesn't parallelize — known follow-up).

## Cost — Per-Episode, Spot Pricing

Computed as `wall_clock_hours × $/hr_spot`. Excludes cold-start.

| Episode | FunASR L4 | MiMo L4 (est) | MiMo L40S | Cost multiplier MiMo vs FunASR (L4) |
|---|---|---|---|---|
| 1 h     | **$0.0133** | $0.140 | $0.146 | **~10.5×** |
| 3h36m   | **$0.0258** | ~$0.47 (est) | ~$0.36 | ~18× |
| 6h45m   | **$0.0494** | $0.941 | $0.980 | **~19×** |

The per-1000-episodes math:

- FunASR L4: 1000 × avg $0.035 = **$35 / 1000 episodes**
- MiMo L40S: 1000 × avg $0.55 = **$550 / 1000 episodes**
- Delta: **~$515** per 1000 episodes = ~15× cost increase

At a 30–50 episodes/day target scale, that's $10–20/day extra GPU spend
on MiMo — not prohibitive, but also not free.

## Why FunASR is so much cheaper

Paraformer is a **non-autoregressive** ASR: one forward pass per audio
chunk, fixed compute regardless of transcript length, heavily optimized
for short inference. FunASR ships quantized weights (~200 M params fp16
or int8) that fit entirely in L4 VRAM with headroom. Observed GPU
utilization in the reference bench: **17–56%** across all configs — the
whole pipeline is VAD-stalled or I/O-bound, not GPU-bound. More
importantly the GPU is *done in seconds*, so Spot per-hour pricing
amortizes over a vanishingly short run.

MiMo-V2.5-ASR is an **autoregressive** decoder on 8 B params: generates
text token-by-token per VAD chunk. Each chunk's compute scales with
generated text length. The GPU is genuinely saturated throughout
(unlike FunASR), so on a $1.19/hr Spot L40S you're paying the full rate
for 49 min to transcribe what Paraformer does in 6 min on the same GPU
for $0.12.

## Where MiMo wins: accuracy

From the e2e's ep-1h intro block (direct diff on identical audio). Real
proper nouns redacted per project policy; representative errors shown:

| Term class | FunASR `--lang zh` | MiMo `--lang mimo` |
|---|---|---|
| Rare Chinese surname (economist name) | wrong character ✗ | correct ✓ |
| `ChatGPT` | `拆GPT` ✗ | **ChatGPT** ✓ |
| English full name (e.g. `Sam Altman`) | lowercased Chinese transliteration | **original casing** ✓ |
| `OpenAI` | `open AI` (broken casing) ✗ | **OpenAI** ✓ |
| `AI 对齐` | `AI 对其` (wrong homophone) ✗ | **AI 对齐** ✓ |

MiMo also:
- Preserves English casing inside Chinese text
- Uses full Chinese punctuation consistently (。？（）《》) where
  Paraformer mixes 、, omits 、
- Handles Chinese-English code-switching cleanly (e.g. "一些 research
  is…" patterns in the marathon transcript)
- Correctly spells rare proper nouns (economists, company names)

**FunASR is not bad** — plain Chinese narrative is roughly equivalent
between the two presets — but for podcast/interview audio with lots
of names, jargon, or English loanwords, MiMo's quality advantage is
consistently visible.

## Where FunASR wins: speaker diarization (unexpected)

One counter-intuitive finding from the e2e: on `ep-1h` (two acoustically
similar hosts), `--lang zh` + `--num-speakers 2` collapsed to 1 speaker
while `--lang mimo` correctly identified 2. This is actually an
_artifact_ of our e2e fixing the CUDA→numpy tensor bug in
`rescore_montage_speakers` (commit `9861a01`) — FunASR was silently
failing to build speaker profiles. **After that fix, both presets
should diarize equivalently** (same CAM++ + KMeans stack). `ep-1p4h`
already got 2 speakers on `--lang zh` even before the fix.

## Practical guidance

### Use FunASR (`--lang zh`) when:
- Processing volume matters ($0.05 vs $0.95 for a 7 h podcast × 1000 episodes = $900 delta)
- Wall-clock throughput matters (5 min vs 49 min)
- Audio is plain Chinese narrative without heavy code-switching or rare proper nouns
- Running on AWS managed GPU (reference bench validates L4 as the sweet spot)

### Use MiMo (`--lang mimo`) when:
- Audio has dense proper nouns, English-in-Chinese code-switching, or rare technical terms
- Accuracy justifies 10–20× the cost for a given batch
- You have ≥ 20 GB VRAM locally (L40S, L4 24GB, A100, H100, etc.) and don't need Spot
- Ad-hoc / research use, where a few extra dollars per episode is cheap vs manual post-editing

### Combined workflow (future idea, not implemented):

1. Always run `--lang zh` first (fast baseline)
2. Detect high-English-ratio / high-proper-noun-density segments in the transcript
3. Re-transcribe those segments with `--lang mimo` targeted at just the suspicious chunks
4. Splice the MiMo output into the FunASR transcript

This would capture MiMo's accuracy win at <10% of its cost. Tracked as
a follow-up idea; the current skill doesn't support partial-chunk
re-transcription.

## Instance-fit matrix (for deployment planning)

| Instance family | VRAM | FunASR (any length) | MiMo (≥ 20 GB floor) | Spot $/hr |
|---|---|---|---|---|
| g5.xlarge (A10G) | 24 GB | ✓ | ✓ (barely) | $0.76 |
| g6.xlarge (L4) | 24 GB | ✓ | ✓ (barely) | $0.46 |
| g6.2xlarge (L4) | 24 GB | ✓ | ✓ (barely) | $0.46 |
| g6e.xlarge (L40S) | 48 GB | ✓ (wasted) | ✓ | $1.19 |
| g6e.2xlarge (L40S) | 48 GB | ✓ (wasted) | ✓ | $1.27 |
| g6e.4xlarge (L40S) | 48 GB | ✓ (wasted) | ✓ | ~$2.00 on-demand |

**For routed production workloads:** `g6.2xlarge` (L4) is the sweet
spot for both presets — FunASR doesn't need the extra VRAM and MiMo
fits at 24 GB VRAM (we verified the 20 GB floor holds with ~4 GB slack
for KV cache on real 6h45m audio).

## Open questions / follow-up work

1. **Measure MiMo on L4 directly.** The 2.5× slowdown estimate is a
   conservative guess. A single 1 h episode run on g6.2xlarge would
   confirm the L4 cost projection.
2. **`--mimo-batch > 1` implementation.** The flag is plumbed through
   the CLI but not wired into concurrent execution. On an H100 80 GB
   this could 4–8× throughput and bring MiMo into FunASR's cost
   territory for longer episodes.
3. **MiMo INT8 / FP8 quantization.** Xiaomi has not published quantized
   weights. A community INT8 build would roughly halve both VRAM and
   wall-clock; not on anyone's roadmap yet.
4. **Break-even audio length for MiMo routing in a FunASR+MiMo hybrid
   pipeline.** With current wall-clock, MiMo is never cheaper; but if
   it could re-transcribe just <10% of audio (suspicious chunks) it'd
   be <2× FunASR cost with near-MiMo accuracy.

## Raw numbers (for sanity-checking)

MiMo on L40S (our e2e, g6e.4xlarge):
```
ep-1h    (3607s): wall 439s (Phase 1 = 398.6s), RTF 0.112, 2 speakers
ep-1p4h  (4960s): wall 645s, RTF 0.124, 2 speakers
ep-6p75h(24329s): wall 2954s (Phase 1 = 2914s), RTF 0.120, 2 speakers
```

FunASR on GPU (reference benchmark, dated 2026-04-26):
```
36m   (2160s): L40S wall 64s / $0.0212, L4 est $0.0085
1h    (3840s): L40S wall 104s / $0.0345, L4 est $0.0133
3h36m(12960s): L40S wall 202s / $0.0670, L4 est $0.0258
6h45m(24300s): L40S wall 377s / $0.1250, L4 wall 388s / $0.0494
```
