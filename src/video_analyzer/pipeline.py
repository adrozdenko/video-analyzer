"""Pipeline orchestrator — runs all stages in the correct order with progress tracking."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from video_analyzer.config.settings import Settings
from video_analyzer.stages.analysis import Transcriber, VisionAnalyzer
from video_analyzer.stages.extraction import AudioExtractor, KeyframeExtractor, probe_video
from video_analyzer.stages.merge import TimelineMerger
from video_analyzer.stages.summarize import Summarizer
from video_analyzer.types import AnalysisSummary, Keyframe, OutputFormat, StageResult, VideoMetadata

console = Console()


class Pipeline:
    """Orchestrates the full video analysis pipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tmp_dir: Path | None = None

    def run(self, video_path: Path) -> StageResult[AnalysisSummary]:
        """Run the full pipeline synchronously (uses asyncio internally for vision)."""
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="video_analyzer_"))

        try:
            return self._execute(video_path)
        finally:
            self._cleanup()

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

        # Estimate keyframes (rough: 1 scene change per 10s for typical video)
        est_keyframes = min(
            int(meta.duration_seconds / 10), self.settings.max_keyframes
        )
        # Claude vision: ~$0.004 per 1024px image (sonnet)
        est_vision_cost = est_keyframes * 0.004
        # Claude summarization: ~$0.003
        est_summary_cost = 0.003
        total = est_vision_cost + est_summary_cost

        console.print(f"\n  Estimated keyframes: ~{est_keyframes}")
        console.print(f"  Transcription cost: $0.00 (local Whisper)")
        console.print(f"  Vision cost: ~${est_vision_cost:.3f} ({est_keyframes} frames)")
        console.print(f"  Summarization cost: ~${est_summary_cost:.3f}")
        console.print(f"  [bold]Total estimated: ~${total:.3f}[/bold]\n")

    def _execute(self, video_path: Path) -> StageResult[AnalysisSummary]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            # Stage 1: Probe
            task = progress.add_task("Probing video...", total=None)
            probe = probe_video(video_path)
            if not probe.ok:
                progress.update(task, description=f"[red]Probe failed: {probe.error}")
                return StageResult.fail(stage="pipeline", error=f"Probe failed: {probe.error}")
            meta = probe.data
            progress.update(task, description=f"[green]Probed: {meta.duration_seconds:.0f}s, {meta.width}x{meta.height}")
            progress.remove_task(task)

            # Stage 2: Extract audio + keyframes in parallel
            task_audio = progress.add_task("Extracting audio...", total=None)
            task_kf = progress.add_task("Extracting keyframes...", total=None)

            audio_extractor = AudioExtractor()
            kf_extractor = KeyframeExtractor()

            audio_result = audio_extractor.extract(video_path, self._tmp_dir)
            self._log_stage(progress, task_audio, "Audio", audio_result)

            kf_result = kf_extractor.extract(
                video_path,
                self._tmp_dir,
                threshold=self.settings.scene_threshold,
                max_keyframes=self.settings.max_keyframes,
                max_width=self.settings.keyframe_max_width,
                fallback_interval=self.settings.fallback_interval_seconds,
            )
            self._log_stage(progress, task_kf, "Keyframes", kf_result)

            # Stage 3: Transcribe + Vision analyze
            task_tr = progress.add_task("Transcribing audio (Whisper)...", total=None)
            transcriber = Transcriber()
            if audio_result.ok:
                transcript_result = transcriber.transcribe(
                    audio_result.data, self.settings.whisper_model.value
                )
            else:
                transcript_result = StageResult.skipped(
                    stage="transcriber", reason=f"No audio: {audio_result.error or 'skipped'}"
                )
            self._log_stage(progress, task_tr, "Transcript", transcript_result)

            task_vis = progress.add_task("Analyzing keyframes (Claude)...", total=None)
            if kf_result.ok and kf_result.data:
                vision_analyzer = VisionAnalyzer()
                vision_result = asyncio.run(
                    vision_analyzer.analyze(
                        kf_result.data,
                        self.settings.vision_concurrency,
                    )
                )
            else:
                vision_result = StageResult.skipped(
                    stage="vision_analyzer",
                    reason=f"No keyframes: {kf_result.error or 'skipped'}",
                )
            self._log_stage(progress, task_vis, "Vision", vision_result)

            # Stage 4: Merge timeline
            task_merge = progress.add_task("Merging timeline...", total=None)
            merger = TimelineMerger()
            timeline_result = merger.merge(transcript_result, vision_result)
            self._log_stage(progress, task_merge, "Timeline", timeline_result)

            if not timeline_result.ok:
                return StageResult.fail(
                    stage="pipeline",
                    error=f"Timeline merge failed: {timeline_result.error}",
                )

            # Stage 5: Summarize
            task_sum = progress.add_task("Generating summary...", total=None)
            output_fmt = OutputFormat(self.settings.output_format)

            summarizer = Summarizer()
            summary_result = summarizer.summarize(meta, timeline_result.data, output_fmt)

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

    def _cleanup(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
