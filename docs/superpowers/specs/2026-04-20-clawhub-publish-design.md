# ClawHub Package Publishing via GitHub Actions

## Goal

Publish the `funasr-transcriber` plugin to ClawHub (OpenClaw skill marketplace)
as package `zxkane-audio-transcriber-funasr`, automated via GitHub Actions.

## Components

### 1. `openclaw.plugin.json` (repo root)

ClawHub package manifest. Fields:

| Field | Value |
|-------|-------|
| name | `zxkane-audio-transcriber-funasr` |
| version | `1.3.0` (synced with SKILL.md) |
| author | `zxkane` |
| license | `MIT` |
| homepage | `https://github.com/zxkane/audio-transcriber-funasr` |
| keywords | audio, transcription, funasr, chinese, speech-to-text, speaker-diarization, meeting-minutes, podcast |

### 2. `.github/workflows/clawhub-publish.yml`

Triggers:

- **pull_request**: dry-run (validates manifest, no mutation)
- **push tags `v*`**: real publish
- **workflow_dispatch**: real publish (manual escape hatch)

Uses official reusable workflow:
`openclaw/clawhub/.github/workflows/package-publish.yml@main`

Version sourced from SKILL.md frontmatter (not git tag).

Secrets: `CLAWHUB_TOKEN` (API token from clawhub.ai Settings > API tokens).

### 3. SKILL.md frontmatter addition

Add `metadata.openclaw` block:

```yaml
metadata:
  openclaw:
    requires:
      bins: ["python3", "ffmpeg"]
    emoji: "🎙️"
    homepage: "https://github.com/zxkane/audio-transcriber-funasr"
```

### Not changed

- `test.yml` workflow (unrelated)
- `.claude-plugin/marketplace.json` (Claude Code ecosystem, separate)
- No script changes

## Setup requirement

User must create a `CLAWHUB_TOKEN` secret in the GitHub repo:
1. Log in to clawhub.ai via GitHub OAuth
2. Settings > API tokens > create token
3. Add as repo secret `CLAWHUB_TOKEN`
