"""CLI entry point for video-analyzer."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from video_analyzer.config.settings import WhisperModel, load_settings
from video_analyzer.pipeline import Pipeline

app = typer.Typer(
    name="video-analyzer",
    help="Analyze videos using hybrid audio transcription + keyframe vision AI.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def analyze(
    video_path: Annotated[Path, typer.Argument(help="Path to video file", exists=True)],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Output format: md, json, txt")] = "md",
    scene_threshold: Annotated[float, typer.Option(help="Scene detection threshold (0.05-0.9)")] = 0.3,
    max_keyframes: Annotated[int, typer.Option(help="Maximum keyframes to extract")] = 20,
    whisper_model: Annotated[str, typer.Option(help="Whisper model: tiny, base, small, medium, large")] = "medium",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show cost estimate without running")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Analyze a video file using audio transcription + keyframe vision AI."""
    # Validate video file
    if not video_path.is_file():
        console.print(f"[red]File not found: {video_path}[/red]")
        raise typer.Exit(1)

    settings = load_settings(
        scene_threshold=scene_threshold,
        max_keyframes=max_keyframes,
        whisper_model=WhisperModel(whisper_model),
        output_format=format,
        verbose=verbose,
    )

    pipeline = Pipeline(settings)

    # Handle Ctrl+C gracefully
    def _signal_handler(sig, frame):
        console.print("\n[yellow]Interrupted — cleaning up...[/yellow]")
        pipeline._cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if dry_run:
        pipeline.dry_run(video_path)
        return

    console.print(f"\n[bold]Analyzing:[/bold] {video_path.name}\n")

    result = pipeline.run(video_path)

    if not result.ok:
        console.print(f"\n[red]Pipeline failed: {result.error}[/red]")
        raise typer.Exit(1)

    summary = result.data

    # Write output
    if output:
        output.write_text(summary.summary_text)
        console.print(f"\n[green]Summary written to {output}[/green]")
    else:
        console.print("\n" + summary.summary_text)

    # Stats
    console.print(f"\n[dim]Transcript segments: {summary.transcript_segments}[/dim]")
    console.print(f"[dim]Keyframes analyzed: {summary.keyframes_analyzed}[/dim]")
    console.print(f"[dim]Pipeline duration: {result.duration_ms:.0f}ms[/dim]")


if __name__ == "__main__":
    app()
