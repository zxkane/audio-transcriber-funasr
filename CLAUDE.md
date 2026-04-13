# Audio Transcriber — FunASR

This project is a Claude Code plugin providing Chinese audio transcription
using FunASR with speaker diarization and LLM cleanup.

## Project Structure

- `plugins/funasr-transcriber/` — The plugin source code
- `.claude-plugin/marketplace.json` — skills.sh marketplace registration
- `.claude/skills/` — Symlinks for Claude Code skill discovery
- `output/` — Generated transcription outputs (gitignored)

## Development

- All skill source lives under `plugins/funasr-transcriber/skills/funasr-transcribe/`
- Scripts are in `scripts/`, references in `references/`
- The main entry point is `SKILL.md`
- Test changes by running transcription on sample audio

## Conventions

- English for code, comments, commit messages, and documentation
- Chinese audio content is the primary use case but the pipeline is language-agnostic
- Keep SKILL.md lean (<2000 words); move details to references/
