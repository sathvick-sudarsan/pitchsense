"""Versioned run manifest."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pitchsense.config import PipelineConfig, TrackerConfig
from pitchsense.contracts import ModelIdentity, RunArtifacts, VideoMetadata
from pitchsense.errors import OutputWriteError


class RunManifest(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    application_version: str
    input_filename: str
    input_size_bytes: int = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video: VideoMetadata
    processed_frame_count: int = Field(ge=0)
    model_identity: ModelIdentity
    tracker_config: TrackerConfig
    pipeline_config: PipelineConfig
    python_version: str
    dependency_versions: dict[str, str]
    started_at_utc: datetime
    completed_at_utc: datetime
    elapsed_seconds: float = Field(ge=0)
    artifacts: RunArtifacts


def write_manifest(path: Path, manifest: RunManifest) -> None:
    try:
        path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise OutputWriteError(f"cannot write manifest: {exc}") from exc
