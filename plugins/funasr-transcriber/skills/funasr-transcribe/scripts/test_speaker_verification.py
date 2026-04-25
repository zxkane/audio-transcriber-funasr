#!/usr/bin/env python3
"""Tests for speaker verification pipeline.

Covers: llm_utils, verify_speakers, and speaker-related functions
in transcribe_funasr. All LLM calls are mocked.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from llm_utils import detect_llm_provider, is_retryable, call_llm
import transcribe_funasr as tf
import verify_speakers as vs


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def make_segment(speaker, start_ms, end_ms, text):
    return {"speaker": speaker, "start_ms": start_ms, "end_ms": end_ms, "text": text}


@pytest.fixture
def two_speaker_transcript():
    return [
        make_segment(0, 0, 5000, "大家好，欢迎来到我们的节目"),
        make_segment(0, 5000, 15000, "今天我们请到了一位特别的嘉宾"),
        make_segment(1, 15000, 30000, "谢谢主持人，很高兴来到这里"),
        make_segment(0, 30000, 45000, "能不能先给大家介绍一下你自己"),
        make_segment(1, 45000, 90000, "好的，我是做人工智能研究的"),
        make_segment(0, 90000, 120000, "那我们今天就来聊聊AI的发展"),
    ]


@pytest.fixture
def four_speaker_transcript():
    return [
        make_segment(0, 0, 10000, "Let's start the meeting"),
        make_segment(1, 10000, 25000, "I'll present the engineering update"),
        make_segment(2, 25000, 40000, "Design review is ready for Q3"),
        make_segment(3, 40000, 55000, "Budget looks good this quarter"),
        make_segment(0, 55000, 70000, "Great, any questions?"),
        make_segment(1, 70000, 90000, "We need more time for the migration"),
    ]


@pytest.fixture
def speaker_context_podcast():
    return {
        "关羽": "Host, asks questions, introduces topics",
        "张飞": "Guest, AI researcher, shares expertise",
    }


@pytest.fixture
def speaker_context_meeting():
    return {
        "Alice": "Team lead, runs the meeting",
        "Bob": "Engineer, gives technical updates",
        "Carol": "Designer, presents design work",
        "Dave": "Finance, reports on budget",
    }


# ──────────────────────────────────────────────
# llm_utils: detect_llm_provider
# ──────────────────────────────────────────────

class TestDetectLLMProvider:
    def test_bedrock_arn(self):
        assert detect_llm_provider("arn:aws:bedrock:us-west-2:123:inference-profile/abc") == "bedrock"

    def test_bedrock_cross_region(self):
        assert detect_llm_provider("us.anthropic.claude-sonnet-4-6") == "bedrock"

    def test_anthropic_bare_claude(self):
        assert detect_llm_provider("claude-sonnet-4-6") == "anthropic"

    def test_anthropic_claude_opus(self):
        assert detect_llm_provider("claude-opus-4-7") == "anthropic"

    def test_openai_gpt(self):
        assert detect_llm_provider("gpt-4o") == "openai"

    def test_openai_deepseek(self):
        assert detect_llm_provider("deepseek-chat") == "openai"

    def test_openai_default(self):
        assert detect_llm_provider("some-custom-model") == "openai"


# ──────────────────────────────────────────────
# llm_utils: is_retryable
# ──────────────────────────────────────────────

class TestIsRetryable:
    def test_rate_limit(self):
        assert is_retryable(Exception("rate_limit_exceeded"))

    def test_throttle(self):
        assert is_retryable(Exception("throttling exception"))

    def test_429(self):
        assert is_retryable(Exception("HTTP 429 Too Many Requests"))

    def test_529(self):
        assert is_retryable(Exception("529 overloaded"))

    def test_not_retryable(self):
        assert not is_retryable(Exception("invalid api key"))

    def test_not_retryable_generic(self):
        assert not is_retryable(Exception("something went wrong"))

    def test_read_timeout_retryable(self):
        assert is_retryable(Exception("Read timeout on endpoint URL: https://bedrock..."))

    def test_connect_timeout_retryable(self):
        assert is_retryable(Exception("Connect timeout on endpoint URL"))

    def test_timed_out_retryable(self):
        assert is_retryable(Exception("Request timed out after 300s"))


# ──────────────────────────────────────────────
# llm_utils: call_llm
# ──────────────────────────────────────────────

class TestCallLLM:
    @patch("llm_utils._call_anthropic", return_value="test response")
    def test_anthropic_routing(self, mock_call):
        result = call_llm("sys", "user", "claude-sonnet-4-6")
        assert result == "test response"
        mock_call.assert_called_once_with("sys", "user", "claude-sonnet-4-6")

    @patch("llm_utils._call_bedrock", return_value="bedrock response")
    def test_bedrock_routing(self, mock_call):
        result = call_llm("sys", "user", "us.anthropic.claude-sonnet-4-6", region="us-east-1")
        assert result == "bedrock response"
        mock_call.assert_called_once_with("sys", "user", "us.anthropic.claude-sonnet-4-6", "us-east-1")

    @patch("llm_utils._call_openai", return_value="openai response")
    def test_openai_routing(self, mock_call):
        result = call_llm("sys", "user", "gpt-4o")
        assert result == "openai response"
        mock_call.assert_called_once_with("sys", "user", "gpt-4o")

    @patch("llm_utils._call_anthropic")
    def test_retry_on_rate_limit(self, mock_call):
        mock_call.side_effect = [Exception("rate_limit_exceeded"), "ok"]
        result = call_llm("sys", "user", "claude-sonnet-4-6", max_retries=2)
        assert result == "ok"
        assert mock_call.call_count == 2

    @patch("llm_utils._call_anthropic")
    def test_no_retry_on_auth_error(self, mock_call):
        mock_call.side_effect = Exception("invalid api key")
        with pytest.raises(Exception, match="invalid api key"):
            call_llm("sys", "user", "claude-sonnet-4-6", max_retries=3)
        assert mock_call.call_count == 1

    @patch("llm_utils._call_bedrock", return_value="response")
    def test_default_region(self, mock_call):
        call_llm("sys", "user", "us.anthropic.claude-sonnet-4-6")
        mock_call.assert_called_once_with("sys", "user", "us.anthropic.claude-sonnet-4-6", "us-west-2")


# ──────────────────────────────────────────────
# verify_speakers: helpers
# ──────────────────────────────────────────────

class TestFormatTimeMs:
    def test_zero(self):
        assert vs.format_time_ms(0) == "00:00:00"

    def test_seconds(self):
        assert vs.format_time_ms(5000) == "00:00:05"

    def test_minutes(self):
        assert vs.format_time_ms(90000) == "00:01:30"

    def test_hours(self):
        assert vs.format_time_ms(3661000) == "01:01:01"


class TestBuildSpeakerMap:
    def test_two_speakers(self, two_speaker_transcript):
        m = vs.build_speaker_map(two_speaker_transcript, ["Host", "Guest"])
        assert m == {0: "Host", 1: "Guest"}

    def test_more_names_than_speakers(self, two_speaker_transcript):
        m = vs.build_speaker_map(two_speaker_transcript, ["A", "B", "C"])
        assert m == {0: "A", 1: "B"}

    def test_fewer_names_than_speakers(self, two_speaker_transcript):
        m = vs.build_speaker_map(two_speaker_transcript, ["OnlyOne"])
        assert m[0] == "OnlyOne"
        assert m[1] == "Speaker 2"

    def test_preserves_appearance_order(self):
        transcript = [
            make_segment(2, 0, 1000, "first"),
            make_segment(0, 1000, 2000, "second"),
        ]
        m = vs.build_speaker_map(transcript, ["Alpha", "Beta"])
        assert m == {2: "Alpha", 0: "Beta"}


class TestExtractEarlyTranscript:
    def test_respects_cutoff(self, two_speaker_transcript):
        speaker_map = {0: "Host", 1: "Guest"}
        result = vs.extract_early_transcript(two_speaker_transcript, 1, speaker_map)
        lines = result.strip().split("\n")
        for line in lines:
            assert "Host:" in line or "Guest:" in line

    def test_empty_transcript(self):
        assert vs.extract_early_transcript([], 5, {}) == ""

    def test_includes_timestamp(self, two_speaker_transcript):
        speaker_map = {0: "H", 1: "G"}
        result = vs.extract_early_transcript(two_speaker_transcript, 5, speaker_map)
        assert "[00:00:00]" in result


class TestComputeSpeakerStats:
    def test_counts_segments_and_chars(self, two_speaker_transcript):
        speaker_map = {0: "Host", 1: "Guest"}
        stats = vs.compute_speaker_stats(two_speaker_transcript, speaker_map)
        assert stats["Host"]["segments"] == 4
        assert stats["Guest"]["segments"] == 2
        assert stats["Host"]["chars"] > 0
        assert stats["Guest"]["chars"] > 0

    def test_with_minutes_cutoff(self, two_speaker_transcript):
        speaker_map = {0: "Host", 1: "Guest"}
        stats_full = vs.compute_speaker_stats(two_speaker_transcript, speaker_map)
        stats_limited = vs.compute_speaker_stats(two_speaker_transcript, speaker_map, minutes=1)
        total_full = sum(s["segments"] for s in stats_full.values())
        total_limited = sum(s["segments"] for s in stats_limited.values())
        assert total_limited <= total_full


# ──────────────────────────────────────────────
# verify_speakers: apply_swap
# ──────────────────────────────────────────────

class TestApplySwap:
    def test_swaps_two_speakers(self):
        transcript = [
            make_segment(0, 0, 1000, "hello"),
            make_segment(1, 1000, 2000, "world"),
            make_segment(0, 2000, 3000, "foo"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        vs.apply_swap(transcript, speaker_map, 0, 1)
        assert transcript[0]["speaker"] == 1
        assert transcript[1]["speaker"] == 0
        assert transcript[2]["speaker"] == 1
        assert speaker_map[0] == "Bob"
        assert speaker_map[1] == "Alice"

    def test_swap_is_reversible(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(1, 1000, 2000, "b")]
        speaker_map = {0: "X", 1: "Y"}
        vs.apply_swap(transcript, speaker_map, 0, 1)
        vs.apply_swap(transcript, speaker_map, 0, 1)
        assert transcript[0]["speaker"] == 0
        assert transcript[1]["speaker"] == 1
        assert speaker_map == {0: "X", 1: "Y"}


# ──────────────────────────────────────────────
# verify_speakers: apply_meeting_mapping
# ──────────────────────────────────────────────

class TestApplyMeetingMapping:
    def test_no_changes(self):
        transcript = [make_segment(0, 0, 1000, "hello")]
        speaker_map = {0: "Alice", 1: "Bob"}
        mapping = {"Alice": "Alice", "Bob": "Bob"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 0

    def test_simple_swap(self):
        transcript = [
            make_segment(0, 0, 1000, "a"),
            make_segment(1, 1000, 2000, "b"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        mapping = {"Alice": "Bob", "Bob": "Alice"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 1
        assert transcript[1]["speaker"] == 0

    def test_partial_mapping_ignored(self):
        transcript = [
            make_segment(0, 0, 1000, "a"),
            make_segment(1, 1000, 2000, "b"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        mapping = {"Alice": "Alice"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 0
        assert transcript[1]["speaker"] == 1

    def test_three_way_rotation(self):
        transcript = [
            make_segment(0, 0, 1000, "a"),
            make_segment(1, 1000, 2000, "b"),
            make_segment(2, 2000, 3000, "c"),
        ]
        speaker_map = {0: "A", 1: "B", 2: "C"}
        mapping = {"A": "B", "B": "A", "C": "C"}
        vs.apply_meeting_mapping(transcript, speaker_map, mapping)
        assert transcript[0]["speaker"] == 1
        assert transcript[1]["speaker"] == 0
        assert transcript[2]["speaker"] == 2


# ──────────────────────────────────────────────
# verify_speakers: verify_podcast (mocked LLM)
# ──────────────────────────────────────────────

class TestVerifyPodcast:
    @patch("verify_speakers.call_llm")
    def test_correct_verdict(self, mock_llm):
        mock_llm.return_value = "VERDICT: CORRECT\nCONFIDENCE: HIGH\nEVIDENCE: labels match roles"
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = vs.verify_podcast("transcript text", speaker_map, ctx, "model", "us-west-2")
        assert result["verdict"] == "CORRECT"
        assert result["confidence"] == "HIGH"

    @patch("verify_speakers.call_llm")
    def test_swap_verdict(self, mock_llm):
        mock_llm.return_value = "VERDICT: SWAP\nCONFIDENCE: HIGH\nEVIDENCE: host is answering"
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = vs.verify_podcast("transcript text", speaker_map, ctx, "model", "us-west-2")
        assert result["verdict"] == "SWAP"

    @patch("verify_speakers.call_llm")
    def test_malformed_response(self, mock_llm):
        mock_llm.return_value = "I cannot determine the speaker roles from this text."
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = vs.verify_podcast("transcript text", speaker_map, ctx, "model", "us-west-2")
        assert result["verdict"] == "UNKNOWN"
        assert result["confidence"] == "LOW"


# ──────────────────────────────────────────────
# verify_speakers: verify_meeting (mocked LLM)
# ──────────────────────────────────────────────

class TestVerifyMeeting:
    @patch("verify_speakers.call_llm")
    def test_correct_mapping(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": True,
            "confidence": "HIGH",
            "mapping": {"Alice": "Alice", "Bob": "Bob"},
            "evidence": {"Alice": "leads meeting", "Bob": "gives updates"},
        })
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "team lead", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is True
        assert result["mapping"]["Alice"] == "Alice"

    @patch("verify_speakers.call_llm")
    def test_swapped_mapping(self, mock_llm):
        mock_llm.return_value = '```json\n{"correct": false, "confidence": "HIGH", "mapping": {"Alice": "Bob", "Bob": "Alice"}, "evidence": {}}\n```'
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "team lead", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is False
        assert result["mapping"]["Alice"] == "Bob"

    @patch("verify_speakers.call_llm")
    def test_no_json_in_response(self, mock_llm):
        mock_llm.return_value = "I'm not sure about the speaker assignments."
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "leader", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is None
        assert "raw" in result

    @patch("verify_speakers.call_llm")
    def test_invalid_json_in_response(self, mock_llm):
        mock_llm.return_value = '{"mapping": invalid json here}'
        speaker_map = {0: "Alice", 1: "Bob"}
        ctx = {"Alice": "leader", "Bob": "engineer"}
        result = vs.verify_meeting("text", speaker_map, ctx, "model", "us-west-2")
        assert result["correct"] is None


# ──────────────────────────────────────────────
# transcribe_funasr: speaker verification functions
# ──────────────────────────────────────────────


class TestTranscribeMergeConsecutive:
    def test_merges_same_speaker(self):
        transcript = [
            make_segment(0, 0, 1000, "hello "),
            make_segment(0, 1500, 2500, "world"),
        ]
        merged = tf.merge_consecutive(transcript)
        assert len(merged) == 1
        assert merged[0]["text"] == "hello world"

    def test_no_merge_different_speakers(self):
        transcript = [
            make_segment(0, 0, 1000, "hello"),
            make_segment(1, 1500, 2500, "world"),
        ]
        merged = tf.merge_consecutive(transcript)
        assert len(merged) == 2

    def test_no_merge_large_gap(self):
        transcript = [
            make_segment(0, 0, 1000, "hello"),
            make_segment(0, 5000, 6000, "world"),
        ]
        merged = tf.merge_consecutive(transcript)
        assert len(merged) == 2

    def test_empty_transcript(self):
        assert tf.merge_consecutive([]) == []

    def test_solo_podcast_keeps_timestamps(self):
        # Long single-speaker run (solo podcast) with small gaps should NOT
        # collapse into a single segment — we cap merged duration so
        # periodic timestamps remain in the output.
        transcript = [
            make_segment(0, i * 10000, i * 10000 + 8000, f"sentence {i} ")
            for i in range(60)  # 10 minutes of tightly-packed speech
        ]
        merged = tf.merge_consecutive(transcript, max_merge_ms=120000)
        assert len(merged) >= 5, (
            f"Expected periodic timestamps for solo podcast, got {len(merged)} segment(s)"
        )
        for seg in merged:
            assert seg["end_ms"] - seg["start_ms"] <= 120000 + 8000


class TestTranscribeBuildSpeakerMap:
    def test_with_names(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(1, 1000, 2000, "b")]
        m = tf.build_speaker_map(transcript, ["Alice", "Bob"])
        assert m == {0: "Alice", 1: "Bob"}

    def test_without_names(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(1, 1000, 2000, "b")]
        m = tf.build_speaker_map(transcript)
        assert m == {0: "Speaker 1", 1: "Speaker 2"}

    def test_solo_podcast_with_host_name(self):
        transcript = [make_segment(0, 0, 1000, "a"), make_segment(0, 2000, 3000, "b")]
        m = tf.build_speaker_map(transcript, ["李雷"])
        assert m == {0: "李雷"}


class TestExtractSpeakerNamesFromReference:
    def test_chinese_host_label(self):
        text = "节目信息\n主播：李雷\n更多内容..."
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_host_and_guest(self):
        text = "主播：关羽\n嘉宾：张飞"
        assert tf.extract_speaker_names_from_reference(text) == ["关羽", "张飞"]

    def test_english_labels(self):
        text = "Host: Alice\nGuest - Bob"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice", "Bob"]

    def test_no_labels_returns_empty(self):
        assert tf.extract_speaker_names_from_reference("random show notes text") == []

    def test_empty_input(self):
        assert tf.extract_speaker_names_from_reference("") == []
        assert tf.extract_speaker_names_from_reference(None) == []

    def test_stops_at_punctuation(self):
        text = "主播：李雷，资深宏观研究员"
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_stops_at_chinese_period(self):
        text = "主播：李雷。本期节目我们将..."
        assert tf.extract_speaker_names_from_reference(text) == ["李雷"]

    def test_stops_at_ascii_period(self):
        text = "Host: Alice.senior analyst"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice"]

    def test_stops_at_parenthesis(self):
        text = "Host: Alice (senior analyst)\nGuest: Bob"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice", "Bob"]

    def test_dedup_across_host_and_guest(self):
        text = "主播：Alice\n嘉宾：Alice"
        assert tf.extract_speaker_names_from_reference(text) == ["Alice"]


class TestDetectAliasInSpeakers:
    def test_alias_detected_chinese(self):
        ref = "主播：张三（张三的播客）"
        result = tf.detect_alias_in_speakers(["张三的播客"], ref)
        assert result == [("张三的播客", "张三")]

    def test_real_name_accepted(self):
        ref = "主播：张三（张三的播客）"
        assert tf.detect_alias_in_speakers(["张三"], ref) == []

    def test_english_alias_detected(self):
        ref = "Host: Alice (AlicePodcast)"
        result = tf.detect_alias_in_speakers(["AlicePodcast"], ref)
        assert result == [("AlicePodcast", "Alice")]

    def test_ascii_parens_detected(self):
        ref = "Host: 张三(张三的播客)\n嘉宾: 李四"
        result = tf.detect_alias_in_speakers(["张三的播客", "李四"], ref)
        assert result == [("张三的播客", "张三")]

    def test_no_parens_in_reference(self):
        ref = "主播：张三\n嘉宾：李四"
        assert tf.detect_alias_in_speakers(["张三"], ref) == []

    def test_multiple_mismatches(self):
        ref = "主播：张三（张三的播客）\n嘉宾：李四（四哥）"
        result = tf.detect_alias_in_speakers(["张三的播客", "四哥"], ref)
        assert ("张三的播客", "张三") in result
        assert ("四哥", "李四") in result

    def test_partial_mismatch(self):
        ref = "主播：张三（张三的播客）\n嘉宾：李四"
        result = tf.detect_alias_in_speakers(["张三的播客", "李四"], ref)
        assert result == [("张三的播客", "张三")]

    def test_unknown_name_not_flagged(self):
        ref = "主播：张三（张三的播客）"
        assert tf.detect_alias_in_speakers(["王五"], ref) == []

    def test_empty_inputs(self):
        assert tf.detect_alias_in_speakers([], "主播：张三（张三的播客）") == []
        assert tf.detect_alias_in_speakers(["张三"], None) == []
        assert tf.detect_alias_in_speakers(["张三"], "") == []

    def test_reference_without_role_label(self):
        ref = "张三（张三的播客）是一位资深投资人"
        assert tf.detect_alias_in_speakers(["张三的播客"], ref) == []


class TestDetectMontageEnd:
    def test_no_montage_few_segments(self):
        transcript = [make_segment(0, 0, 30000, "long intro")]
        assert tf.detect_montage_end(transcript) == 0

    def test_no_montage_all_long(self):
        transcript = [
            make_segment(0, 0, 20000, "first long"),
            make_segment(1, 20000, 45000, "second long"),
        ]
        assert tf.detect_montage_end(transcript) == 0

    def test_detects_cold_open(self):
        """Short clips followed by a long segment = montage."""
        transcript = [
            make_segment(0, 0, 3000, "clip one"),
            make_segment(1, 3000, 6000, "clip two"),
            make_segment(0, 6000, 9000, "clip three"),
            make_segment(1, 9000, 12000, "clip four"),
            make_segment(0, 12000, 30000, "welcome to the show, real intro starts"),
        ]
        assert tf.detect_montage_end(transcript) == 4

    def test_no_montage_mixed_lengths(self):
        """One long segment among early ones breaks the montage pattern."""
        transcript = [
            make_segment(0, 0, 3000, "short"),
            make_segment(1, 3000, 20000, "long early on"),
            make_segment(0, 20000, 23000, "short again"),
        ]
        assert tf.detect_montage_end(transcript) == 0

    def test_many_short_clips_then_long_intro(self):
        """Typical podcast cold open: 8 highlight clips then a long real intro."""
        transcript = [
            make_segment(0, 7000, 12000, "highlight clip one"),
            make_segment(1, 17000, 21000, "highlight clip two"),
            make_segment(0, 21000, 27000, "highlight clip three"),
            make_segment(0, 28000, 38000, "highlight clip four"),
            make_segment(0, 39000, 47000, "highlight clip five"),
            make_segment(1, 48000, 50000, "highlight clip six"),
            make_segment(0, 50000, 53000, "highlight clip seven"),
            make_segment(0, 55000, 60000, "highlight clip eight"),
            make_segment(1, 62000, 103000, "Hi everyone welcome to the show, this is the real intro that goes on for a while"),
        ]
        assert tf.detect_montage_end(transcript) == 8


class TestRescoreMontageSpakers:
    def test_noop_when_montage_end_zero(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        result = tf.rescore_montage_speakers(transcript, 0, "/fake.wav", "model")
        assert result is transcript

    def test_noop_when_montage_end_exceeds_length(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        result = tf.rescore_montage_speakers(transcript, 5, "/fake.wav", "model")
        assert result is transcript

    def test_noop_when_montage_end_negative(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        result = tf.rescore_montage_speakers(transcript, -1, "/fake.wav", "model")
        assert result is transcript


class TestDetectMontageEndBoundary:
    def test_exactly_four_segments_montage(self):
        """Minimum montage: 3 short clips + 1 long at index 3."""
        transcript = [
            make_segment(0, 0, 3000, "clip one"),
            make_segment(1, 3000, 6000, "clip two"),
            make_segment(0, 6000, 9000, "clip three"),
            make_segment(1, 9000, 30000, "real intro starts here"),
        ]
        assert tf.detect_montage_end(transcript) == 3

    def test_exactly_four_segments_no_montage_long_not_enough(self):
        """4 segments but the 4th is only 14s — below 15s threshold."""
        transcript = [
            make_segment(0, 0, 3000, "clip one"),
            make_segment(1, 3000, 6000, "clip two"),
            make_segment(0, 6000, 9000, "clip three"),
            make_segment(1, 9000, 23000, "almost long enough"),
        ]
        assert tf.detect_montage_end(transcript) == 0


class TestVerifySpeakerAssignment:
    def test_detects_chinese_self_intro_mismatch(self):
        transcript = [
            make_segment(0, 0, 5000, "大家好我是张飞"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "关羽", 1: "张飞"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["关羽", "张飞"])
        assert result[0] == "张飞"
        assert result[1] == "关羽"

    def test_confirms_correct_assignment(self):
        transcript = [
            make_segment(0, 0, 5000, "大家好我是关羽"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "关羽", 1: "张飞"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["关羽", "张飞"])
        assert result[0] == "关羽"

    def test_english_self_intro(self):
        transcript = [
            make_segment(0, 0, 5000, "Hi everyone I'm Bob"),
            make_segment(1, 5000, 10000, "Welcome"),
        ]
        speaker_map = {0: "Alice", 1: "Bob"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["Alice", "Bob"])
        assert result[0] == "Bob"
        assert result[1] == "Alice"

    def test_no_intro_returns_unchanged(self):
        transcript = [
            make_segment(0, 0, 5000, "今天天气不错"),
            make_segment(1, 5000, 10000, "是的"),
        ]
        speaker_map = {0: "A", 1: "B"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["A", "B"])
        assert result == {0: "A", 1: "B"}

    def test_given_name_match_chinese(self):
        """'我是丽华' should match full name '王丽华'."""
        transcript = [
            make_segment(0, 0, 5000, "嗨，朋友们好，欢迎收听我们的节目，我是丽华"),
            make_segment(1, 5000, 10000, "好，我是某某频道的主播赵大明"),
        ]
        speaker_map = {0: "赵大明", 1: "王丽华"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["赵大明", "王丽华"])
        assert result[0] == "王丽华"
        assert result[1] == "赵大明"

    def test_filler_between_intro_and_name(self):
        """'我是某某频道的主播赵大明' should match despite filler words."""
        transcript = [
            make_segment(0, 0, 5000, "好，感谢节目的邀请，我是某某频道的主播赵大明"),
            make_segment(1, 5000, 10000, "欢迎"),
        ]
        speaker_map = {0: "赵大明", 1: "王丽华"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["赵大明", "王丽华"])
        assert result[0] == "赵大明"

    def test_two_char_name_given_name_match(self):
        """Two-char Chinese name: '我是磊' should match '陈磊'."""
        transcript = [
            make_segment(0, 0, 5000, "大家好我是磊"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "林峰", 1: "陈磊"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["林峰", "陈磊"])
        assert result[0] == "陈磊"
        assert result[1] == "林峰"

    def test_name_variants_helper(self):
        """_name_variants produces correct variants."""
        assert tf._name_variants("王丽华") == [("王丽华", "王丽华"), ("丽华", "王丽华")]
        assert tf._name_variants("赵大明") == [("赵大明", "赵大明"), ("大明", "赵大明")]
        assert tf._name_variants("Alice") == [("Alice", "Alice")]
        assert tf._name_variants("陈磊") == [("陈磊", "陈磊"), ("磊", "陈磊")]

    def test_name_variants_four_char_name(self):
        """4-char Chinese name (compound surname)."""
        assert tf._name_variants("欧阳明月") == [("欧阳明月", "欧阳明月"), ("阳明月", "欧阳明月")]

    def test_filler_exceeding_limit_no_match(self):
        """16 chars of filler between '我是' and name should NOT match."""
        transcript = [
            make_segment(0, 0, 5000, "我是这个节目的特别邀请来的嘉宾主持人赵大明"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "王丽华", 1: "赵大明"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["王丽华", "赵大明"])
        assert result == {0: "王丽华", 1: "赵大明"}

    def test_punctuation_cutoff_prevents_cross_sentence_match(self):
        """Punctuation between '我是' and name should block matching."""
        transcript = [
            make_segment(0, 0, 5000, "我是主持人。赵大明你好"),
            make_segment(1, 5000, 10000, "你好"),
        ]
        speaker_map = {0: "王丽华", 1: "赵大明"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, ["王丽华", "赵大明"])
        assert result == {0: "王丽华", 1: "赵大明"}

    def test_no_speaker_names_returns_unchanged(self):
        transcript = [make_segment(0, 0, 5000, "hello")]
        speaker_map = {0: "Speaker 1"}
        result = tf.verify_speaker_assignment(transcript, speaker_map, None)
        assert result == {0: "Speaker 1"}


class TestChunkByDuration:
    def test_single_chunk(self):
        items = [make_segment(0, 0, 1000, "a"), make_segment(0, 1000, 2000, "b")]
        chunks = tf.chunk_by_duration(items, duration_ms=900000)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        items = [
            make_segment(0, 0, 1000, "first"),
            make_segment(0, 1000000, 1001000, "second"),
        ]
        chunks = tf.chunk_by_duration(items, duration_ms=900000)
        assert len(chunks) == 2

    def test_empty(self):
        assert tf.chunk_by_duration([]) == []


class TestFormatChunk:
    def test_formats_with_names(self):
        chunk = [make_segment(0, 0, 1000, "hello"), make_segment(1, 1000, 2000, "world")]
        speaker_map = {0: "Alice", 1: "Bob"}
        result = tf.format_chunk(chunk, speaker_map)
        assert "[00:00:00] Alice: hello" in result
        assert "[00:00:01] Bob: world" in result


class TestBuildSystemPrompt:
    def test_base_prompt(self):
        prompt = tf.build_system_prompt()
        assert "transcript editor" in prompt
        assert "Keep timestamps unchanged" in prompt

    def test_with_speaker_names(self):
        prompt = tf.build_system_prompt(speaker_names=["Alice", "Bob"])
        assert "Alice" in prompt
        assert "Bob" in prompt

    def test_with_speaker_context(self):
        ctx = {"Alice": "team lead", "Bob": "engineer"}
        prompt = tf.build_system_prompt(speaker_context=ctx)
        assert "team lead" in prompt
        assert "engineer" in prompt

    def test_with_reference_text(self):
        prompt = tf.build_system_prompt(reference_text="Meeting about Q3 roadmap")
        assert "Q3 roadmap" in prompt

    def test_reference_truncation(self):
        long_ref = "x" * 5000
        prompt = tf.build_system_prompt(reference_text=long_ref)
        assert "[...truncated]" in prompt


# ──────────────────────────────────────────────
# transcribe_funasr: validate_lang_diarization
# ──────────────────────────────────────────────

class TestValidateLangDiarization:
    def test_auto_with_speakers_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            tf.validate_lang_diarization("auto", 2)
        assert exc_info.value.code == 1

    def test_whisper_with_speakers_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            tf.validate_lang_diarization("whisper", 3)
        assert exc_info.value.code == 1

    def test_zh_with_speakers_ok(self):
        tf.validate_lang_diarization("zh", 2)

    def test_auto_without_speakers_ok(self):
        tf.validate_lang_diarization("auto", None)

    def test_zh_basic_with_speakers_ok(self):
        tf.validate_lang_diarization("zh-basic", 5)

    def test_en_with_speakers_ok(self):
        tf.validate_lang_diarization("en", 4)


# ──────────────────────────────────────────────
# transcribe_funasr: --model-cache-dir
# ──────────────────────────────────────────────

class TestModelCacheDir:
    def test_argparse_accepts_model_cache_dir(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("audio_file")
        parser.add_argument("--model-cache-dir", type=str, default=None)
        args = parser.parse_args(["test.wav", "--model-cache-dir", "/tmp/models"])
        assert args.model_cache_dir == "/tmp/models"

    def test_argparse_default_is_none(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("audio_file")
        parser.add_argument("--model-cache-dir", type=str, default=None)
        args = parser.parse_args(["test.wav"])
        assert args.model_cache_dir is None


# ──────────────────────────────────────────────
# transcribe_funasr: parse_funasr_results
# ──────────────────────────────────────────────

class TestParseFunasrResults:
    def test_sentence_info_shape(self):
        """Standard Paraformer output with sentence_info (speaker diarization)."""
        res = [{"sentence_info": [
            {"spk": 0, "start": 0, "end": 5000, "text": "hello"},
            {"spk": 1, "start": 5000, "end": 10000, "text": "world"},
        ]}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 2
        assert transcript[0] == {"speaker": 0, "start_ms": 0, "end_ms": 5000, "text": "hello"}
        assert transcript[1] == {"speaker": 1, "start_ms": 5000, "end_ms": 10000, "text": "world"}

    def test_sentence_info_uses_sentence_key(self):
        """Some models use 'sentence' instead of 'text' in sentence_info."""
        res = [{"sentence_info": [
            {"spk": 0, "start": 0, "end": 1000, "sentence": "fallback text"},
        ]}]
        transcript = tf.parse_funasr_results(res)
        assert transcript[0]["text"] == "fallback text"

    def test_text_only_no_timestamp(self):
        """SenseVoice/Whisper output: text only, no timestamps."""
        res = [{"text": "auto detected speech"}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "auto detected speech"
        assert transcript[0]["speaker"] == 0
        assert transcript[0]["start_ms"] == 0
        assert transcript[0]["end_ms"] == 0

    def test_text_with_timestamp_no_sentence_info(self):
        """Result with text + timestamp but no sentence_info — previously dropped."""
        res = [{"text": "detected speech", "timestamp": [[0, 500], [500, 1200]]}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "detected speech"
        assert transcript[0]["speaker"] == 0
        assert transcript[0]["start_ms"] == 0
        assert transcript[0]["end_ms"] == 1200

    def test_text_with_empty_timestamp(self):
        """Result with text + empty timestamp list."""
        res = [{"text": "speech with empty ts", "timestamp": []}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 1
        assert transcript[0]["text"] == "speech with empty ts"
        assert transcript[0]["start_ms"] == 0
        assert transcript[0]["end_ms"] == 0

    def test_unknown_shape_warns(self, capsys):
        """Result with no recognized keys should warn, not silently drop."""
        res = [{"unknown_key": "value"}]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "unrecognized" in captured.out.lower() or "unknown" in captured.out.lower()

    def test_multiple_results_mixed(self):
        """Multiple results with different shapes in one response."""
        res = [
            {"sentence_info": [{"spk": 0, "start": 0, "end": 5000, "text": "first"}]},
            {"text": "second", "timestamp": [[5000, 6000], [6000, 8000]]},
            {"text": "third"},
        ]
        transcript = tf.parse_funasr_results(res)
        assert len(transcript) == 3
        assert transcript[0]["text"] == "first"
        assert transcript[1]["text"] == "second"
        assert transcript[1]["end_ms"] == 8000
        assert transcript[2]["text"] == "third"
        assert transcript[2]["start_ms"] == 0

    def test_empty_results(self):
        """Empty results list."""
        assert tf.parse_funasr_results([]) == []

    def test_sentence_info_missing_spk_defaults_to_zero(self):
        """sentence_info entries without 'spk' key default to speaker 0."""
        res = [{"sentence_info": [{"start": 0, "end": 1000, "text": "no spk"}]}]
        transcript = tf.parse_funasr_results(res)
        assert transcript[0]["speaker"] == 0


# ──────────────────────────────────────────────
# transcribe_funasr: _verify_speaker_roles_via_llm
# ──────────────────────────────────────────────

class TestVerifySpeakerRolesViaLLM:
    @patch("transcribe_funasr.call_llm", return_value="CORRECT")
    def test_correct_keeps_map(self, mock_llm):
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Host"
        assert result[1] == "Guest"

    @patch("transcribe_funasr.call_llm", return_value="SWAP")
    def test_swap_two_speakers(self, mock_llm):
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Guest"
        assert result[1] == "Host"

    @patch("transcribe_funasr.call_llm", return_value="I'm not sure about this")
    def test_ambiguous_keeps_map(self, mock_llm):
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Host"
        assert result[1] == "Guest"

    @patch("transcribe_funasr.call_llm")
    def test_llm_failure_keeps_map(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API error")
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Host"
        assert result[1] == "Guest"

    @patch("transcribe_funasr.call_llm")
    def test_import_error_propagates(self, mock_llm):
        mock_llm.side_effect = ImportError("No module named 'boto3'")
        speaker_map = {0: "Host", 1: "Guest"}
        ctx = {"Host": "asks questions", "Guest": "answers"}
        with pytest.raises(ImportError):
            tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")

    @patch("transcribe_funasr.call_llm")
    def test_multi_speaker_json_swap(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": False,
            "mapping": {"Alice": "Bob", "Bob": "Alice", "Carol": "Carol"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Bob"
        assert result[1] == "Alice"
        assert result[2] == "Carol"

    @patch("transcribe_funasr.call_llm")
    def test_multi_speaker_correct(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": True,
            "mapping": {"Alice": "Alice", "Bob": "Bob", "Carol": "Carol"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Alice"

    @patch("transcribe_funasr.call_llm")
    def test_multi_speaker_invalid_json_keeps_map(self, mock_llm):
        mock_llm.return_value = "not valid json at all"
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result == {0: "Alice", 1: "Bob", 2: "Carol"}

    @patch("transcribe_funasr.call_llm")
    def test_multi_speaker_three_way_cycle(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": False,
            "mapping": {"Alice": "Bob", "Bob": "Carol", "Carol": "Alice"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result[0] == "Bob"
        assert result[1] == "Carol"
        assert result[2] == "Alice"

    @patch("transcribe_funasr.call_llm")
    def test_multi_speaker_duplicate_targets_rejected(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "correct": False,
            "mapping": {"Alice": "Carol", "Bob": "Carol", "Carol": "Alice"},
        })
        speaker_map = {0: "Alice", 1: "Bob", 2: "Carol"}
        ctx = {"Alice": "leader", "Bob": "eng", "Carol": "design"}
        result = tf._verify_speaker_roles_via_llm("text", speaker_map, ctx, "model", "us-west-2")
        assert result == {0: "Alice", 1: "Bob", 2: "Carol"}


# ──────────────────────────────────────────────
# Integration: main() with mock LLM
# ──────────────────────────────────────────────

class TestVerifySpeakersMain:
    def _write_fixtures(self, tmpdir, transcript, context):
        json_path = tmpdir / "transcript.json"
        ctx_path = tmpdir / "context.json"
        json_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        ctx_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
        return json_path, ctx_path

    @patch("verify_speakers.call_llm")
    def test_podcast_swap_dryrun(self, mock_llm, two_speaker_transcript,
                                  speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: SWAP\nCONFIDENCE: HIGH\nEVIDENCE: labels are swapped"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path)]
        vs.main()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["speaker"] == 0  # unchanged — dry-run doesn't modify

    @patch("verify_speakers.call_llm")
    def test_podcast_swap_fix(self, mock_llm, two_speaker_transcript,
                               speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: SWAP\nCONFIDENCE: HIGH\nEVIDENCE: labels are swapped"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path), "--fix"]
        vs.main()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["speaker"] == 1  # swapped

    @patch("verify_speakers.call_llm")
    def test_podcast_correct_no_changes(self, mock_llm, two_speaker_transcript,
                                         speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: CORRECT\nCONFIDENCE: HIGH\nEVIDENCE: looks good"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path), "--fix"]
        vs.main()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["speaker"] == 0  # unchanged

    @patch("verify_speakers.call_llm")
    def test_unknown_verdict_exits_2(self, mock_llm, two_speaker_transcript,
                                      speaker_context_podcast, tmp_path):
        mock_llm.return_value = "VERDICT: MAYBE\nCONFIDENCE: LOW\nEVIDENCE: unclear"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, two_speaker_transcript, speaker_context_podcast)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "关羽,张飞",
                     "--speaker-context", str(ctx_path)]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 2

    @patch("verify_speakers.call_llm")
    def test_meeting_inconclusive_exits_2(self, mock_llm, four_speaker_transcript,
                                           speaker_context_meeting, tmp_path):
        mock_llm.return_value = "I don't know"
        json_path, ctx_path = self._write_fixtures(
            tmp_path, four_speaker_transcript, speaker_context_meeting)
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "Alice,Bob,Carol,Dave",
                     "--speaker-context", str(ctx_path)]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 2

    def test_missing_context_file_exits_1(self, two_speaker_transcript, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps(two_speaker_transcript), encoding="utf-8")
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "A,B",
                     "--speaker-context", str(tmp_path / "nonexistent.json")]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 1

    def test_invalid_json_context_exits_1(self, two_speaker_transcript, tmp_path):
        json_path = tmp_path / "transcript.json"
        json_path.write_text(json.dumps(two_speaker_transcript), encoding="utf-8")
        ctx_path = tmp_path / "bad.json"
        ctx_path.write_text("not valid json {{{", encoding="utf-8")
        sys.argv = ["verify_speakers.py", str(json_path),
                     "--speakers", "A,B",
                     "--speaker-context", str(ctx_path)]
        with pytest.raises(SystemExit) as exc_info:
            vs.main()
        assert exc_info.value.code == 1


# ──────────────────────────────────────────────
# Phase1-only and json-out flag tests
# ──────────────────────────────────────────────

class TestPhase1Flags:
    """Tests for --phase1-only and --json-out CLI flags."""

    def _parse(self, extra_args):
        """Parse CLI args with defaults suitable for testing."""
        base = ["test.wav"]
        with patch("sys.argv", ["transcribe_funasr.py"] + base + extra_args):
            p = argparse.ArgumentParser()
            p.add_argument("audio_file")
            p.add_argument("--phase1-only", action="store_true")
            p.add_argument("--json-out", type=str, default=None)
            p.add_argument("--skip-transcribe", action="store_true")
            p.add_argument("--skip-llm", action="store_true")
            p.add_argument("--model", default=None)
            p.add_argument("--output", default=None)
            return p.parse_args(base + extra_args)

    def test_phase1_only_flag_parsed(self):
        args = self._parse(["--phase1-only"])
        assert args.phase1_only is True

    def test_phase1_only_default_false(self):
        args = self._parse([])
        assert args.phase1_only is False

    def test_json_out_flag_parsed(self):
        args = self._parse(["--json-out", "/tmp/out.json"])
        assert args.json_out == "/tmp/out.json"

    def test_json_out_default_none(self):
        args = self._parse([])
        assert args.json_out is None

    def test_json_out_overrides_default_path(self):
        from pathlib import Path
        args = self._parse(["--json-out", "/tmp/custom.json"])
        raw_json = Path(args.json_out) if args.json_out else Path(f"{Path(args.audio_file).stem}_raw_transcript.json")
        assert raw_json == Path("/tmp/custom.json")

    def test_default_raw_json_path(self):
        from pathlib import Path
        args = self._parse([])
        raw_json = Path(args.json_out) if args.json_out else Path(f"{Path(args.audio_file).stem}_raw_transcript.json")
        assert raw_json == Path("test_raw_transcript.json")

    def test_phase1_only_with_json_out(self):
        args = self._parse(["--phase1-only", "--json-out", "/tmp/out.json"])
        assert args.phase1_only is True
        assert args.json_out == "/tmp/out.json"

    def test_phase1_only_early_exit(self, tmp_path):
        """--phase1-only exits after Phase 1 without producing .md output."""
        transcript = [
            make_segment(0, 0, 5000, "Hello world"),
            make_segment(1, 5000, 10000, "Hi there"),
        ]
        raw_json = tmp_path / "test_raw_transcript.json"
        with open(raw_json, "w") as f:
            json.dump(transcript, f)

        md_path = tmp_path / "test-transcript.md"
        test_args = [
            "transcribe_funasr.py",
            str(tmp_path / "test.wav"),
            "--phase1-only",
            "--skip-transcribe",
            "--json-out", str(raw_json),
            "--device", "cpu",
        ]
        with patch("sys.argv", test_args), \
             patch.object(sys, "exit") as mock_exit:
            mock_exit.side_effect = SystemExit(0)
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 0
        assert not md_path.exists()

    def test_json_out_writes_to_custom_path(self, tmp_path):
        """--json-out writes transcript JSON to the specified path."""
        custom_json = tmp_path / "custom_output.json"
        transcript = [
            make_segment(0, 0, 5000, "Hello"),
            make_segment(1, 5000, 10000, "World"),
        ]
        with patch("sys.argv", [
                "transcribe_funasr.py", str(tmp_path / "test.wav"),
                "--json-out", str(custom_json),
                "--skip-llm", "--device", "cpu",
            ]), \
             patch.object(tf, "transcribe_with_funasr", return_value=transcript), \
             patch.object(tf, "preprocess_audio", return_value=str(tmp_path / "test.wav")), \
             patch.object(tf, "detect_montage_end", return_value=0), \
             patch("pathlib.Path.exists", return_value=True):
            tf.main()
        assert custom_json.exists()
        saved = json.loads(custom_json.read_text())
        assert len(saved) == 2
        assert saved[0]["text"] == "Hello"

    def test_without_new_flags_runs_all_phases(self, tmp_path):
        """Without new flags, all phases run (backward compatibility)."""
        transcript = [
            make_segment(0, 0, 5000, "Hello"),
            make_segment(1, 5000, 10000, "World"),
        ]
        output_md = tmp_path / "test-transcript.md"
        raw_json = tmp_path / "test_raw_transcript.json"

        def fake_exists(self_path=None):
            return True

        with patch("sys.argv", [
                "transcribe_funasr.py", str(tmp_path / "test.wav"),
                "--skip-llm", "--device", "cpu",
                "--output", str(output_md),
                "--json-out", str(raw_json),
            ]), \
             patch.object(tf, "transcribe_with_funasr", return_value=transcript), \
             patch.object(tf, "preprocess_audio", return_value=str(tmp_path / "test.wav")), \
             patch.object(tf, "detect_montage_end", return_value=0), \
             patch("pathlib.Path.exists", return_value=True):
            tf.main()
        assert output_md.exists()
        assert raw_json.exists()
        content = output_md.read_text()
        assert "Transcript" in content

    def test_json_out_nonexistent_parent_exits(self, tmp_path):
        """--json-out to a nonexistent directory exits with code 1."""
        bad_path = tmp_path / "nonexistent" / "dir" / "out.json"
        test_args = [
            "transcribe_funasr.py",
            str(tmp_path / "test.wav"),
            "--json-out", str(bad_path),
            "--device", "cpu",
        ]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 1

    def test_phase1_only_empty_transcript_exits_error(self, tmp_path):
        """--phase1-only with empty transcript exits with error, not success."""
        raw_json = tmp_path / "empty_raw_transcript.json"
        with open(raw_json, "w") as f:
            json.dump([], f)
        test_args = [
            "transcribe_funasr.py",
            str(tmp_path / "test.wav"),
            "--phase1-only",
            "--skip-transcribe",
            "--json-out", str(raw_json),
            "--device", "cpu",
        ]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 1

    def test_phase1_only_writes_json_then_exits(self, tmp_path):
        """--phase1-only writes raw JSON and exits 0 without --skip-transcribe."""
        transcript = [
            make_segment(0, 0, 5000, "Hello"),
            make_segment(1, 5000, 10000, "World"),
        ]
        custom_json = tmp_path / "phase1_out.json"
        with patch("sys.argv", [
                "transcribe_funasr.py", str(tmp_path / "test.wav"),
                "--phase1-only",
                "--json-out", str(custom_json),
                "--device", "cpu",
            ]), \
             patch.object(tf, "transcribe_with_funasr", return_value=transcript), \
             patch.object(tf, "preprocess_audio", return_value=str(tmp_path / "test.wav")), \
             patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                tf.main()
            assert exc_info.value.code == 0
        assert custom_json.exists()
        saved = json.loads(custom_json.read_text())
        assert len(saved) == 2
        assert saved[0]["text"] == "Hello"
        md_path = tmp_path / "test-transcript.md"
        assert not md_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
