"""CLI entry point — placeholder for Task #6."""

import typer

app = typer.Typer(
    name="video-analyzer",
    help="Analyze videos using hybrid audio transcription + keyframe vision AI.",
)


@app.command()
def analyze(video_path: str) -> None:
    """Analyze a video file."""
    typer.echo(f"TODO: analyze {video_path}")


if __name__ == "__main__":
    app()
