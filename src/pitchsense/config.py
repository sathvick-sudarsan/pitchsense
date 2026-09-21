"""Validated pipeline configuration; loading is added in PR2."""

from pydantic import BaseModel, ConfigDict, Field

from pitchsense.contracts import Confidence, NonBlank


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_ConfigModel):
    backend: NonBlank
    model: NonBlank
    confidence_threshold: Confidence


class TrackerConfig(_ConfigModel):
    track_buffer: int = Field(gt=0)


class RenderingConfig(_ConfigModel):
    enabled: bool


class OutputConfig(_ConfigModel):
    codec: str = Field(pattern=r"^[!-~]{4}$")


class PipelineConfig(_ConfigModel):
    detector: DetectorConfig
    tracker: TrackerConfig
    rendering: RenderingConfig
    output: OutputConfig
