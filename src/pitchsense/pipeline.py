"""Deterministic video pipeline orchestration."""

import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from pitchsense.artifacts.jsonl import JsonlWriter
from pitchsense.artifacts.manifest import RunManifest, write_manifest
from pitchsense.config import PipelineConfig
from pitchsense.contracts import FrameResult, RunArtifacts
from pitchsense.detection.base import Detector
from pitchsense.errors import InputVideoError, OutputWriteError
from pitchsense.provenance import input_sha256, installed_versions
from pitchsense.rendering.opencv import render
from pitchsense.tracking.base import Tracker
from pitchsense.video.opencv import VideoReader, VideoWriter


class PipelineRunner:
    def __init__(self, detector: Detector, tracker: Tracker, config: PipelineConfig):
        self.detector = detector
        self.tracker = tracker
        self.config = PipelineConfig.model_validate(config)

    def run(self, source: Path, output: Path) -> RunArtifacts:
        source, output = Path(source), Path(output)
        started_at = datetime.now(UTC)
        start = perf_counter()
        if not source.is_file():
            raise InputVideoError(f"input video does not exist: {source}")
        digest = input_sha256(source)
        size = source.stat().st_size
        artifacts = RunArtifacts(
            annotated_video_filename="annotated.avi",
            results_filename="results.jsonl",
            manifest_filename="manifest.json",
        )
        video_path = output / artifacts.annotated_video_filename
        results_path = output / artifacts.results_filename
        manifest_path = output / artifacts.manifest_filename
        partial_video = output / "annotated.partial.avi"
        partial_results = output / "results.partial.jsonl"
        partial_manifest = output / "manifest.partial.json"
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OutputWriteError(f"cannot create output directory: {exc}") from exc
        if any(path.exists() for path in (video_path, results_path, manifest_path)):
            raise OutputWriteError(f"run artifacts already exist in {output}")
        try:
            with VideoReader(source) as reader:
                metadata = reader.metadata
                self.tracker.reset()
                processed = 0
                with VideoWriter(
                    partial_video, metadata, self.config.output.codec
                ) as video_writer:
                    with JsonlWriter(partial_results) as results_writer:
                        for frame in reader:
                            detections = self.detector.detect(frame)
                            tracks = self.tracker.update(detections, frame)
                            result = FrameResult(
                                frame_index=frame.index,
                                timestamp_s=frame.timestamp_s,
                                detections=list(detections),
                                tracks=list(tracks),
                            )
                            results_writer.write(result)
                            video_writer.write(
                                render(frame, tracks, self.config.rendering.enabled)
                            )
                            processed += 1
            partial_video.replace(video_path)
            partial_results.replace(results_path)
            manifest = RunManifest(
                application_version=version("pitchsense"),
                input_filename=source.name,
                input_size_bytes=size,
                input_sha256=digest,
                video=metadata,
                processed_frame_count=processed,
                model_identity=self.detector.model_identity,
                tracker_config=self.config.tracker,
                pipeline_config=self.config,
                python_version=sys.version.split()[0],
                dependency_versions=installed_versions(),
                started_at_utc=started_at,
                completed_at_utc=datetime.now(UTC),
                elapsed_seconds=perf_counter() - start,
                artifacts=artifacts,
            )
            write_manifest(partial_manifest, manifest)
            partial_manifest.replace(manifest_path)
            return artifacts
        except OSError as exc:
            raise OutputWriteError(f"cannot write run artifacts: {exc}") from exc
        finally:
            if not manifest_path.exists():
                for path in (
                    partial_video,
                    partial_results,
                    partial_manifest,
                    video_path,
                    results_path,
                ):
                    path.unlink(missing_ok=True)
