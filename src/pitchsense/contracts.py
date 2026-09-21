"""JSON-serializable domain contracts."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class BoundingBox(BaseModel):
    x1: FiniteFloat
    y1: FiniteFloat
    x2: FiniteFloat
    y2: FiniteFloat

    @model_validator(mode="after")
    def validate_order(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive width and height")
        return self


class Detection(BaseModel):
    class_id: int
    class_name: NonBlank
    confidence: Confidence
    bbox: BoundingBox


class Track(BaseModel):
    track_id: int
    class_id: int
    class_name: NonBlank
    confidence: Confidence | None = None
    bbox: BoundingBox


class VideoMetadata(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: FiniteFloat = Field(gt=0)
    reported_frame_count: int | None = Field(default=None, ge=0)


class FrameResult(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    frame_index: int = Field(ge=0)
    timestamp_s: FiniteFloat = Field(ge=0)
    detections: list[Detection] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)


class ModelIdentity(BaseModel):
    backend: NonBlank
    identifier: NonBlank
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    class_names: list[NonBlank] = Field(min_length=1)
    resolved_device: NonBlank


class RunArtifacts(BaseModel):
    annotated_video_filename: NonBlank
    results_filename: NonBlank
    manifest_filename: NonBlank

    @field_validator(
        "annotated_video_filename", "results_filename", "manifest_filename"
    )
    @classmethod
    def relative_filename(cls, value: str) -> str:
        if value in {".", ".."} or any(char in value for char in "/\\:\0"):
            raise ValueError("artifact must be a relative filename")
        return value
