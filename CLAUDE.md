# Audio Transcriber — FunASR

This project is a Claude Code plugin providing Chinese audio transcription
using FunASR with speaker diarization and LLM cleanup.

## Project Structure

- `plugins/funasr-transcriber/` — The plugin source code
- `.agents/skills/` — Autonomous dev team skills (installed via skills.sh)
- `.claude/skills/` — Symlinks for Claude Code skill discovery
- `.claude-plugin/marketplace.json` — skills.sh marketplace registration
- `output/` — Generated transcription outputs (gitignored)

## Development Workflow

- Use `/autonomous-dev` skill for code changes (TDD, worktree isolation, CI verification)
- Use `/autonomous-review` skill for PR code review before merging
- Use `/create-issue` skill for structured GitHub issue creation
- Use `document-skills:skill-creator` skill to verify/review SKILL.md changes
- All skill source lives under `plugins/funasr-transcriber/skills/funasr-transcribe/`
- Scripts are in `scripts/`, references in `references/`
- The main entry point is `SKILL.md`
- Run tests: `cd plugins/funasr-transcriber/skills/funasr-transcribe/scripts && python3 -m pytest test_speaker_verification.py -v`

## Conventions

- English for code, comments, commit messages, and documentation
- Chinese audio content is the primary use case but the pipeline is language-agnostic
- Keep SKILL.md lean (<500 lines); move details to references/
- Scripts run directly from plugin directory via `$SCRIPTS` — never copy to CWD
