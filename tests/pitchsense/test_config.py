import pytest
from pydantic import ValidationError

from pitchsense.config import (
    DetectorConfig,
    OutputConfig,
    PipelineConfig,
    RenderingConfig,
    TrackerConfig,
)


def test_pipeline_config_nests_valid_models():
    config = PipelineConfig(
        detector=DetectorConfig(
            backend="sample", model="weights", confidence_threshold=0.5
        ),
        tracker=TrackerConfig(track_buffer=30),
        rendering=RenderingConfig(enabled=True),
        output=OutputConfig(codec="mp4v"),
    )
    assert PipelineConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_detector_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        DetectorConfig(
            backend="sample", model="weights", confidence_threshold=confidence
        )


@pytest.mark.parametrize("fields", [{"backend": " "}, {"model": ""}])
def test_detector_requires_names(fields):
    with pytest.raises(ValidationError):
        DetectorConfig(**({"backend": "sample", "model": "weights"} | fields))


def test_tracker_requires_positive_buffer():
    with pytest.raises(ValidationError):
        TrackerConfig(track_buffer=0)


@pytest.mark.parametrize("codec", ["abc", "abcde", "💥💥💥💥"])
def test_output_requires_four_ascii_codec_characters(codec):
    with pytest.raises(ValidationError):
        OutputConfig(codec=codec)
