"""CLI entry point for video-analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from video_analyzer.config.settings import DetailMode, WhisperModel, load_settings
from video_analyzer.pipeline import Pipeline

_FORMAT_EXT = {"md": ".md", "json": ".json", "txt": ".txt"}

app = typer.Typer(
    name="video-analyzer",
    help="Analyze videos using hybrid audio transcription + keyframe vision AI.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def analyze(
    video_path: Annotated[
        Path,
        typer.Argument(
            help="Path to video file", exists=True, file_okay=True, dir_okay=False
        ),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path")
    ] = None,
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format: md, json, txt")
    ] = "md",
    scene_threshold: Annotated[
        float, typer.Option(help="Scene detection threshold (0.05-0.9)")
    ] = 0.3,
    max_keyframes: Annotated[
        int, typer.Option(help="Maximum keyframes to extract")
    ] = 20,
    whisper_model: Annotated[
        str, typer.Option(help="Whisper model: tiny, base, small, medium, large")
    ] = "medium",
    audio_only: Annotated[
        bool,
        typer.Option("--audio-only", "-a", help="Skip video keyframes, analyze audio/transcript only"),
    ] = False,
    detail: Annotated[
        bool,
        typer.Option("--detail", "-d", help="Exhaustive knowledge extraction instead of summary"),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show cost estimate without running")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Analyze a video file using audio transcription + keyframe vision AI."""
    settings = load_settings(
        scene_threshold=scene_threshold,
        max_keyframes=max_keyframes,
        whisper_model=WhisperModel(whisper_model),
        audio_only=audio_only,
        output_format=format,
        detail_mode=DetailMode.DETAILED if detail else DetailMode.SUMMARY,
        verbose=verbose,
    )

    pipeline = Pipeline(settings)

    if dry_run:
        pipeline.dry_run(video_path)
        return

    console.print(f"\n[bold]Analyzing:[/bold] {video_path.name}\n")

    result = pipeline.run(video_path)

    if not result.ok:
        console.print(f"\n[red]Pipeline failed: {result.error}[/red]")
        raise typer.Exit(1)

    summary = result.data

    if output:
        summary.save(output)
        console.print(f"\n[green]Summary written to {output}[/green]")
    else:
        console.print("\n" + summary.summary_text)

    console.print(f"\n[dim]Transcript segments: {summary.transcript_segments}[/dim]")
    console.print(f"[dim]Keyframes analyzed: {summary.keyframes_analyzed}[/dim]")
    console.print(f"[dim]Pipeline duration: {result.duration_ms:.0f}ms[/dim]")


@app.command()
def batch(
    patterns: Annotated[
        list[str],
        typer.Argument(help="File paths or glob patterns (e.g. '*.mp4' 'clips/*.mov')"),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-O", help="Directory to save output files"),
    ] = None,
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format: md, json, txt")
    ] = "md",
    whisper_model: Annotated[
        str, typer.Option(help="Whisper model: tiny, base, small, medium, large")
    ] = "medium",
    audio_only: Annotated[
        bool, typer.Option("--audio-only", "-a", help="Skip keyframe extraction")
    ] = False,
    detail: Annotated[
        bool, typer.Option("--detail", "-d", help="Exhaustive knowledge extraction")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Analyze multiple video files by path or glob pattern."""
    files: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.exists() and p.is_file():
            files.append(p)
        else:
            matched = sorted(Path().glob(pattern))
            if not matched:
                console.print(f"[yellow]No files matched: {pattern}[/yellow]")
            files.extend(f for f in matched if f.is_file())

    if not files:
        console.print("[red]No files to process.[/red]")
        raise typer.Exit(1)

    settings = load_settings(
        whisper_model=WhisperModel(whisper_model),
        audio_only=audio_only,
        output_format=format,
        detail_mode=DetailMode.DETAILED if detail else DetailMode.SUMMARY,
        verbose=verbose,
    )
    ext = _FORMAT_EXT.get(format, ".md")
    pipeline = Pipeline(settings)

    passed = failed = 0
    console.print(f"\n[bold]Batch: {len(files)} file(s)[/bold]\n")

    for i, video_path in enumerate(files, 1):
        console.print(f"[bold][{i}/{len(files)}][/bold] {video_path.name}")
        result = pipeline.run(video_path)

        if not result.ok:
            console.print(f"  [red]Failed: {result.error}[/red]")
            failed += 1
            continue

        summary = result.data
        if output_dir:
            out_path = output_dir / f"{video_path.stem}{ext}"
            summary.save(out_path)
            console.print(f"  [green]Saved → {out_path}[/green]")
        else:
            console.print("\n" + summary.summary_text)

        console.print(
            f"  [dim]{summary.transcript_segments} segments, "
            f"{summary.keyframes_analyzed} keyframes, {result.duration_ms:.0f}ms[/dim]"
        )
        passed += 1

    console.print(f"\n[bold]Done:[/bold] {passed} succeeded, {failed} failed\n")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
