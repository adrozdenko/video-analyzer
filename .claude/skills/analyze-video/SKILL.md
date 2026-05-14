---
name: analyze-video
description: Extract knowledge, transcripts, and insights from video or audio files using the local video-analyzer pipeline
argument-hint: [file path or description of what to analyze]
---

# /analyze-video

Run the video-analyzer pipeline on a local file and return structured knowledge.

## What This Skill Does

1. Determines the right mode based on the file type and user intent
2. Runs `uv run video-analyzer` with the correct flags
3. Saves the output and presents it clearly

## File Type Detection

| Extension | Behavior |
|-----------|----------|
| `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` | Audio-only auto-detected — no `--audio-only` flag needed |
| `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` | Full pipeline (keyframes + audio) unless user says "audio only" |

## Mode Selection

| User says | Flags to use |
|-----------|-------------|
| "transcribe", "text", "what was said" | `--audio-only --whisper-model large --format txt` |
| "knowledge", "extract", "insights", "everything" | `--detail` (auto-uses claude-sonnet for vision) |
| "summary", "overview", "quick" | (default, no extra flags — uses claude-haiku for vision) |
| "precise text", "word by word", "exact words" | `--audio-only --whisper-model large` |
| Long video > 30 min, audio-only | `--audio-only --whisper-model large` |
| Multiple files | Use `batch` subcommand |

**Default whisper model:** `medium` (fast, good enough)
**For important content:** `large` (slower but significantly more accurate, especially for non-English)

**Vision model defaults:**
- `--detail` → claude-sonnet (auto-elevated, reads dense text/slides/code accurately)
- default → claude-haiku (cheap, good for general scene description)
- Override with `--vision-model haiku/sonnet/opus`

## Instructions

### Step 1 — Parse Input
- `$ARGUMENTS` is a file path, a description containing a path, or multiple paths
- Detect file extension to determine mode
- Read user intent to select the right flags

### Step 2 — Build the Command

Single file:
```bash
uv run video-analyzer analyze <path> [flags] --output <output_path>
```

Multiple files or glob:
```bash
uv run video-analyzer batch <path1> <path2> --output-dir /tmp/batch_results/
```

Always use `--output` (or `--output-dir` for batch) to save results. Name single-file outputs:
`/tmp/<filename_stem>_analysis.md` (or `.txt` for transcription-only)

### Step 3 — Run and Report

Run from the project directory: `/Users/adrozdenko/projects/video-analyzer`

```bash
cd /Users/adrozdenko/projects/video-analyzer && uv run video-analyzer analyze <path> <flags> --output <output_path>
```

After running, read the output file and present it to the user.

### Step 4 — Offer Next Steps

After extraction, suggest what makes sense:
- Transcript → offer `/heygen-script` to adapt it for avatar video
- Knowledge extraction → offer to summarize key points or create a HeyGen script
- Long transcript → offer to identify the most quotable moments

## Common Command Examples

**Transcribe an MP3 word-by-word (precise):**
```bash
uv run video-analyzer analyze /path/to/audio.mp3 --audio-only --whisper-model large --format txt --output /tmp/audio_transcript.txt
```

**Full knowledge extraction from a video (uses sonnet automatically):**
```bash
uv run video-analyzer analyze /path/to/video.mp4 --detail --output /tmp/video_analysis.md
```

**Quick summary of a video (haiku, cheap):**
```bash
uv run video-analyzer analyze /path/to/video.mp4 --output /tmp/video_summary.md
```

**Audio-only track from video (skip keyframe cost):**
```bash
uv run video-analyzer analyze /path/to/video.mp4 --audio-only --whisper-model large --output /tmp/video_transcript.md
```

**Force a specific vision model:**
```bash
uv run video-analyzer analyze /path/to/video.mp4 --detail --vision-model opus --output /tmp/video_analysis.md
```

**Batch process a folder:**
```bash
uv run video-analyzer batch /path/to/*.mp4 --output-dir /tmp/batch/ --detail
```

## Error Handling

| Error | Fix |
|-------|-----|
| `Whisper OOM` | Switch to `--whisper-model medium` or `small` |
| `ffmpeg not found` | User needs to install ffmpeg |
| `Pipeline failed: Probe failed` | Check file path exists and is readable |
| `Claude CLI not found` | Run `npm install -g @anthropic-ai/claude-code` |

## Output Location

Always save output to `/tmp/<stem>_<mode>.md` and tell the user the path so they can reference it later.
