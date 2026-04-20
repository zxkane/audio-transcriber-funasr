# Test Cases: Montage Detection & Chinese Given-Name Matching

**Issue:** #7

## Montage Detection (`TestDetectMontageEnd`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| MD-01 | Few segments (< 4) | 1 long segment | Return 0 (< 4 segments, early exit) |
| MD-02 | All long segments | 2 segments, both > 15s | Return 0 (< 4 segments, early exit) |
| MD-03 | Classic cold open | 4 short clips (3s each) + 1 long (18s) | Return 4 (montage ends at index 4) |
| MD-04 | Mixed lengths early | 1 short + 1 long + 1 short (3 total) | Return 0 (< 4 segments, early exit) |
| MD-05 | Many clips then intro | 8 highlight clips (2-10s) + 1 long intro (41s) | Return 8 (montage ends at index 8) |

## Montage Detection Boundary (`TestDetectMontageEndBoundary`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| MD-06 | Minimum montage (4 segments) | 3 short (3s each) + 1 long (21s) | Return 3 (montage ends at index 3) |
| MD-07 | 4 segments, long not enough | 3 short (3s each) + 1 at 14s (< 15s threshold) | Return 0 (no montage) |

## Rescore Montage Speakers (`TestRescoreMontageSpakers`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| RS-01 | montage_end is 0 | montage_end=0 | Return transcript unchanged |
| RS-02 | montage_end exceeds length | montage_end=5, len=1 | Return transcript unchanged |
| RS-03 | montage_end is negative | montage_end=-1 | Return transcript unchanged |

## Chinese Name Variant Matching (`TestVerifySpeakerAssignment`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| NM-01 | 3-char given name | "我是丽华" with speaker_names=["赵大明", "王丽华"] | Matches "王丽华", swaps labels |
| NM-02 | Filler between intro and name | "我是某某频道的主播赵大明" | Matches "赵大明" despite filler |
| NM-03 | 2-char name given name | "我是磊" with speaker_names=["林峰", "陈磊"] | Matches "陈磊" via given name "磊" |

## Filler Regex Boundary (`TestVerifySpeakerAssignment`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| FR-01 | Filler exceeds 15 chars | "我是" + 16 filler chars + name | No match (unchanged) |
| FR-02 | Punctuation cutoff | "我是主持人。赵大明你好" | No match (sentence boundary blocks) |

## Name Variants Helper (`test_name_variants_helper`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| NV-01 | 3-char Chinese name | "王丽华" | [("王丽华", "王丽华"), ("丽华", "王丽华")] |
| NV-02 | 3-char Chinese name | "赵大明" | [("赵大明", "赵大明"), ("大明", "赵大明")] |
| NV-03 | Non-Chinese name | "Alice" | [("Alice", "Alice")] |
| NV-04 | 2-char Chinese name | "陈磊" | [("陈磊", "陈磊"), ("磊", "陈磊")] |
| NV-05 | 4-char Chinese name | "欧阳明月" | [("欧阳明月", "欧阳明月"), ("阳明月", "欧阳明月")] |
