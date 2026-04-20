# Test Cases: Montage Detection & Chinese Given-Name Matching

**Issue:** #7

## Montage Detection (`TestDetectMontageEnd`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| MD-01 | Few segments (< 4) | 1 long segment | Return 0 (no montage) |
| MD-02 | All long segments | 2 segments, both > 15s | Return 0 (no montage) |
| MD-03 | Classic cold open | 4 short clips (3s each) + 1 long (18s) | Return 4 (montage ends at index 4) |
| MD-04 | Mixed lengths early | 1 short + 1 long + 1 short | Return 0 (long segment too early breaks pattern) |
| MD-05 | Many clips then intro | 8 highlight clips (2-10s) + 1 long intro (41s) | Return 8 (montage ends at index 8) |

## Chinese Name Variant Matching (`TestVerifySpeakerAssignment`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| NM-01 | 3-char given name | "我是丽华" with speaker_names=["赵大明", "王丽华"] | Matches "王丽华", swaps labels |
| NM-02 | Filler between intro and name | "我是某某频道的主播赵大明" | Matches "赵大明" despite filler |
| NM-03 | 2-char name given name | "我是磊" with speaker_names=["林峰", "陈磊"] | Matches "陈磊" via given name "磊" |

## Name Variants Helper (`test_name_variants_helper`)

| ID | Scenario | Input | Expected |
|----|----------|-------|----------|
| NV-01 | 3-char Chinese name | "王丽华" | [("王丽华", "王丽华"), ("丽华", "王丽华")] |
| NV-02 | 3-char Chinese name | "赵大明" | [("赵大明", "赵大明"), ("大明", "赵大明")] |
| NV-03 | Non-Chinese name | "Alice" | [("Alice", "Alice")] |
| NV-04 | 2-char Chinese name | "陈磊" | [("陈磊", "陈磊"), ("磊", "陈磊")] |
