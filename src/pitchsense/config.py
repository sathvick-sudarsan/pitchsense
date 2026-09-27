"""Validated pipeline configuration and TOML loading."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from pitchsense.contracts import Confidence, NonBlank
from pitchsense.errors import ConfigurationError


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_ConfigModel):
    backend: NonBlank
    model: NonBlank
    confidence_threshold: Confidence
    device: Literal["auto", "cpu", "cuda"] = "auto"
    classes: list[NonBlank] = Field(default=["person", "sports ball"], min_length=1)


class TrackerConfig(_ConfigModel):
    # Field names follow supervision 0.30.5 sv.ByteTrack arguments.
    backend: NonBlank = "bytetrack"
    classes: list[NonBlank] = Field(default=["person"], min_length=1)
    track_activation_threshold: Confidence = 0.25
    lost_track_buffer: int = Field(default=30, gt=0)
    minimum_matching_threshold: Confidence = 0.8


class RenderingConfig(_ConfigModel):
    enabled: bool


class OutputConfig(_ConfigModel):
    codec: str = Field(pattern=r"^[!-~]{4}$")


class PipelineConfig(_ConfigModel):
    detector: DetectorConfig
    tracker: TrackerConfig
    rendering: RenderingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def tracked_classes_are_detected(self):
        missing = [
            name for name in self.tracker.classes if name not in self.detector.classes
        ]
        if missing:
            raise ValueError(
                f"tracker classes {missing} are not in detector classes "
                f"{self.detector.classes}"
            )
        return self


def load_pipeline_config(
    path: Path | None = None,
    *,
    model: str | None = None,
    device: str | None = None,
    confidence: float | None = None,
) -> PipelineConfig:
    values = {
        "detector": {
            "backend": "rfdetr",
            "model": "rfdetr-nano",
            "confidence_threshold": 0.3,
        },
        "tracker": {},
        "rendering": {"enabled": True},
        "output": {"codec": "MJPG"},
    }
    if path is not None:
        try:
            with Path(path).open("rb") as stream:
                loaded = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot load config {path}: {exc}") from exc
        for section, fields in loaded.items():
            if section not in values or not isinstance(fields, dict):
                raise ConfigurationError(f"invalid config section: {section}")
            values[section].update(fields)
    overrides = {"model": model, "device": device, "confidence_threshold": confidence}
    values["detector"].update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    try:
        return PipelineConfig.model_validate(values)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
