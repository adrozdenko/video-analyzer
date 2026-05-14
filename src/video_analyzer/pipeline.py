"""Pipeline orchestrator — runs all stages in the correct order with progress tracking."""

from __future__ import annotations

import asyncio
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from video_analyzer.config.settings import Settings
from video_analyzer.stages.analysis import analyze_keyframes, transcribe
from video_analyzer.stages.extraction import extract_audio, extract_keyframes, probe_video
from video_analyzer.stages.merge import merge_timeline
from video_analyzer.stages.summarize import summarize
from video_analyzer.types import AnalysisSummary, OutputFormat, StageResult

console = Console()


class Pipeline:
    """Orchestrates the full video analysis pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tmp_dir: Path | None = None
        self._register_signals()

    def _register_signals(self) -> None:
        def _handler(sig, frame):
            console.print("\n[yellow]Interrupted — cleaning up...[/yellow]")
            self.cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def run(self, video_path: Path) -> StageResult[AnalysisSummary]:
        """Run the full pipeline synchronously (uses asyncio internally for vision)."""
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="video_analyzer_"))
        try:
            return self._execute(video_path)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = None

    def dry_run(self, video_path: Path) -> None:
        """Show what would happen without running the pipeline."""
        probe = probe_video(video_path)
        if not probe.ok:
            console.print(f"[red]Cannot probe video: {probe.error}[/red]")
            return

        meta = probe.data
        console.print("\n[bold]Dry Run — Cost Estimate[/bold]\n")
        console.print(f"  Video: {video_path.name}")
        console.print(f"  Duration: {meta.duration_seconds:.0f}s")
        console.print(f"  Resolution: {meta.width}x{meta.height}")
        console.print(f"  Has audio: {meta.has_audio}")
        console.print(f"  Whisper model: {self.settings.whisper_model.value} (local, free)")
        console.print(f"  Scene threshold: {self.settings.scene_threshold}")
        console.print(f"  Max keyframes: {self.settings.max_keyframes}")

        est_keyframes = min(
            int(meta.duration_seconds / 10), self.settings.max_keyframes
        )
        # Cost estimates assume haiku ($0.0008/image, $0.0003 summary at typical input size).
        est_vision_cost = est_keyframes * 0.0008
        est_summary_cost = 0.0003
        total = est_vision_cost + est_summary_cost

        console.print(f"\n  Estimated keyframes: ~{est_keyframes}")
        console.print(f"  Transcription cost: $0.00 (local Whisper)")
        console.print(f"  Vision cost: ~${est_vision_cost:.4f} ({est_keyframes} frames)")
        console.print(f"  Summarization cost: ~${est_summary_cost:.4f}")
        console.print(f"  [bold]Total estimated: ~${total:.4f}[/bold]\n")

    def _execute(self, video_path: Path) -> StageResult[AnalysisSummary]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("Probing video...", total=None)
            probe = probe_video(video_path)
            if not probe.ok:
                progress.update(task, description=f"[red]Probe failed: {probe.error}")
                return StageResult.fail(stage="pipeline", error=f"Probe failed: {probe.error}")
            meta = probe.data
            progress.update(
                task,
                description=f"[green]Probed: {meta.duration_seconds:.0f}s, {meta.width}x{meta.height}",
            )
            progress.remove_task(task)

            task_audio = progress.add_task("Extracting audio...", total=None)
            audio_result = extract_audio(video_path, self._tmp_dir)
            self._log_stage(progress, task_audio, "Audio", audio_result)

            if self.settings.audio_only:
                kf_result = StageResult.skipped(
                    stage="keyframe_extraction", reason="Audio-only mode"
                )
            else:
                task_kf = progress.add_task("Extracting keyframes...", total=None)
                kf_result = extract_keyframes(
                    video_path,
                    self._tmp_dir,
                    threshold=self.settings.scene_threshold,
                    max_keyframes=self.settings.max_keyframes,
                    max_width=self.settings.keyframe_max_width,
                    fallback_interval=self.settings.fallback_interval_seconds,
                )
                self._log_stage(progress, task_kf, "Keyframes", kf_result)

            task_tr = progress.add_task("Transcribing audio (Whisper)...", total=None)
            if audio_result.ok:
                transcript_result = transcribe(
                    audio_result.data, self.settings.whisper_model.value
                )
            else:
                transcript_result = StageResult.skipped(
                    stage="transcribe",
                    reason=f"No audio: {audio_result.error or 'skipped'}",
                )
            self._log_stage(progress, task_tr, "Transcript", transcript_result)

            task_vis = progress.add_task(
                f"Analyzing keyframes (claude-{self.settings.vision_model})...", total=None
            )
            if kf_result.ok and kf_result.data:
                vision_result = asyncio.run(
                    analyze_keyframes(
                        kf_result.data,
                        self.settings.vision_concurrency,
                        model=self.settings.vision_model,
                    )
                )
            else:
                vision_result = StageResult.skipped(
                    stage="vision_analysis",
                    reason=f"No keyframes: {kf_result.error or 'skipped'}",
                )
            self._log_stage(progress, task_vis, "Vision", vision_result)

            task_merge = progress.add_task("Merging timeline...", total=None)
            timeline_result = merge_timeline(transcript_result, vision_result)
            self._log_stage(progress, task_merge, "Timeline", timeline_result)

            if not timeline_result.ok:
                return StageResult.fail(
                    stage="pipeline",
                    error=f"Timeline merge failed: {timeline_result.error}",
                )

            task_sum = progress.add_task("Generating summary...", total=None)
            output_fmt = OutputFormat(self.settings.output_format)
            summary_result = summarize(
                meta, timeline_result.data, output_fmt, self.settings.detail_mode
            )
            self._log_stage(progress, task_sum, "Summary", summary_result)

        return summary_result

    def _log_stage(self, progress: Progress, task_id, name: str, result: StageResult) -> None:
        if result.ok:
            detail = ""
            if result.data and isinstance(result.data, list):
                detail = f" ({len(result.data)} items)"
            elif result.data and isinstance(result.data, Path):
                size_kb = result.data.stat().st_size / 1024
                detail = f" ({size_kb:.0f}KB)"
            progress.update(
                task_id,
                description=f"[green]{name}: done{detail} [{result.duration_ms:.0f}ms]",
            )
        elif result.status.value == "skipped":
            progress.update(task_id, description=f"[yellow]{name}: skipped — {result.error}")
        else:
            progress.update(task_id, description=f"[red]{name}: failed — {result.error}")
        progress.remove_task(task_id)
