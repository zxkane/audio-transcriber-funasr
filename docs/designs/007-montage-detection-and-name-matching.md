# Design: Montage Detection & Chinese Given-Name Matching

**Issue:** #7
**Status:** Implemented

## Problem

Phase 2 speaker verification fails on two common podcast patterns:

1. **Cold-open montages** — rapid-fire highlight clips (< 12s each) at the
   start of an episode. CAM++ diarization assigns wrong speaker IDs to these
   short clips, and Phase 2 incorrectly picks them up as self-introductions.

2. **Chinese given-name intros** — hosts say "我是丽华" (given name only)
   instead of "我是王丽华" (full name), or use filler words between the intro
   phrase and name ("我是某某频道的主播赵大明"). Strict `\s*` regex patterns
   miss these.

## Design

### Montage Detection (`detect_montage_end`)

A time-based heuristic scans the first 3 minutes of transcript:

- Walk segments sequentially
- When a segment >= 15s is found at index `i >= 3`, check if >= 75% of
  prior segments are short (< 12s)
- If yes, return `i` as the montage boundary; otherwise return 0

Thresholds were tuned on real Chinese podcast cold opens (typically 4-10
clips of 2-10s each, followed by a 20-40s real intro).

### Embedding Re-scoring (`rescore_montage_speakers`)

For detected montage zones, diarization clusters are unreliable. Instead:

1. Extract CAM++ speaker embeddings for each segment
2. Build reference profiles by averaging embeddings from the first 5 minutes
   of post-montage content (where diarization is reliable)
3. For each montage segment, compute cosine similarity against all profiles
4. Reassign to the best-matching speaker

This leverages the same speaker model used for diarization but with a
per-segment comparison approach instead of clustering.

### Chinese Name Variants (`_name_variants`)

For Chinese names (2-4 CJK characters):
- Generate `(variant, full_name)` tuples for matching
- e.g. "王丽华" → `[("王丽华", "王丽华"), ("丽华", "王丽华")]`

Non-Chinese names pass through unchanged as `[(name, name)]`.

### Filler-Tolerant Intro Patterns

Updated regex patterns allow variable-length filler between the intro
phrase and speaker name:

| Pattern | Max filler | Example |
|---------|-----------|---------|
| `我是...{name}` | 15 chars | 我是某某频道的主播赵大明 |
| `我叫...{name}` | 10 chars | 我叫那个赵大明 |
| `大家好...{name}` | 20 chars | 大家好欢迎收听本期节目我是赵大明 |

Filler is constrained to exclude sentence-ending punctuation (`。？！\n`)
to prevent cross-sentence false matches.

### LLM Montage Awareness

System prompt rules 9-10 instruct the LLM cleanup phase to:
- Detect and label montage sections with `[片头混剪]` / `[片尾混剪]` headers
- Fix speaker labels within montage sections based on content semantics
- Trust existing speaker labels outside montage sections

## Integration

1. `detect_montage_end()` runs before Phase 2 self-intro scan
2. `rescore_montage_speakers()` runs in `main()` after Phase 1, before merge
3. `verify_speaker_assignment()` skips montage segments for self-intro matching
4. LLM cleanup (Phase 3) receives montage-aware system prompt
