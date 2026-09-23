"""Validated pipeline configuration and TOML loading."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pitchsense.contracts import Confidence, NonBlank
from pitchsense.errors import ConfigurationError


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_ConfigModel):
    backend: NonBlank
    model: NonBlank
    confidence_threshold: Confidence
    device: NonBlank = "cpu"


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
            "model": "rf-detr-base",
            "confidence_threshold": 0.3,
            "device": "cpu",
        },
        "tracker": {"track_buffer": 30},
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
