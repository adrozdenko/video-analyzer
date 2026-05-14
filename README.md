# video-analyzer

Analyze videos using local Whisper transcription + Claude vision on keyframes, producing structured markdown, JSON, or plain text summaries.

## Requirements

- Python ≥ 3.12
- [ffmpeg](https://ffmpeg.org/download.html) on your PATH
- `claude` CLI (for keyframe vision) — `npm install -g @anthropic-ai/claude-code`

## Install

```bash
uv pip install -e .
```

Or with dev dependencies:

```bash
uv pip install -e ".[dev]"
```

## Usage

```bash
video-analyzer <video_file> [OPTIONS]
```

### Basic examples

```bash
# Analyze a video, print markdown to stdout
video-analyzer lecture.mp4

# Save output to file
video-analyzer lecture.mp4 -o summary.md

# Audio-only — skip keyframe extraction (faster, cheaper)
video-analyzer podcast.mp3 --audio-only

# Deep knowledge extraction instead of high-level summary
video-analyzer tutorial.mp4 --detail

# Preview cost estimate without running
video-analyzer big_video.mp4 --dry-run
```

### CLI flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output PATH` | `-o` | stdout | Write output to file |
| `--format` | `-f` | `md` | Output format: `md`, `json`, `txt` |
| `--audio-only` | `-a` | off | Skip keyframe extraction; transcribe only |
| `--detail` | `-d` | off | Exhaustive knowledge extraction (longer output) |
| `--dry-run` | | off | Show cost estimate, don't run |
| `--whisper-model` | | `medium` | Whisper size: `tiny`, `base`, `small`, `medium`, `large` |
| `--scene-threshold` | | `0.3` | Scene-change sensitivity (0.05–0.9) |
| `--max-keyframes` | | `20` | Cap on extracted keyframes |
| `--verbose` | `-v` | off | Log debug output |

### Output formats

| Format | Flag | Best for |
|--------|------|----------|
| Markdown | `-f md` | Human reading, notes, Obsidian |
| JSON | `-f json` | Downstream processing, agents |
| Plain text | `-f txt` | Quick reads, pasting into other tools |

### Input formats

Accepts any file ffmpeg can decode: `.mp4`, `.mov`, `.mkv`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.ogg`, etc.

Audio-only inputs (MP3, WAV, M4A) are detected automatically and skip keyframe extraction — no need to pass `--audio-only` explicitly.

## Pipeline

```
probe_video → extract_audio → transcribe (Whisper)
                           → extract_keyframes → analyze_keyframes (Claude)
                                              → merge_timeline → summarize
```

Each stage produces a `StageResult[T]` — failures in optional stages (vision, transcription) are surfaced as warnings rather than fatal errors, so you still get a partial result.

## Agent usage

From within Claude Code, the `/analyze-video` skill wraps this tool with smart defaults:

```
/analyze-video path/to/file.mp4
/analyze-video podcast.mp3 transcribe
/analyze-video lecture.mp4 extract knowledge
```

For raw CLI use from an agent, JSON output is most ergonomic:

```bash
video-analyzer input.mp4 -f json -o /tmp/result.json
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/
```
