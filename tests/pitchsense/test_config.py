import pytest
from pydantic import ValidationError

from pitchsense.config import (
    DetectorConfig,
    OutputConfig,
    PipelineConfig,
    RenderingConfig,
    TrackerConfig,
    load_pipeline_config,
)
from pitchsense.errors import ConfigurationError


def test_pipeline_config_nests_valid_models():
    config = PipelineConfig(
        detector=DetectorConfig(
            backend="sample", model="weights", confidence_threshold=0.5
        ),
        tracker=TrackerConfig(),
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
        TrackerConfig(lost_track_buffer=0)


def test_defaults_select_rfdetr_nano_bytetrack_and_auto_device():
    config = load_pipeline_config()
    assert config.detector.backend == "rfdetr"
    assert config.detector.model == "rfdetr-nano"
    assert config.detector.device == "auto"
    assert config.detector.classes == ["person", "sports ball"]
    assert config.tracker.backend == "bytetrack"
    assert config.tracker.classes == ["person"]


def test_rejects_unknown_device():
    with pytest.raises(ConfigurationError, match="device"):
        load_pipeline_config(device="gpu")


def test_tracker_classes_must_be_detected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[tracker]\nclasses = ["person", "ball"]\n')
    with pytest.raises(ConfigurationError, match=r"tracker classes \['ball'\]"):
        load_pipeline_config(path)


@pytest.mark.parametrize("codec", ["abc", "abcde", "💥💥💥💥"])
def test_output_requires_four_ascii_codec_characters(codec):
    with pytest.raises(ValidationError):
        OutputConfig(codec=codec)


def test_toml_load_and_explicit_cli_overrides(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[detector]\nmodel = "from-file"\nconfidence_threshold = 0.2\n')
    config = load_pipeline_config(path, model="from-cli", confidence=0.8, device="cuda")
    assert config.detector.model == "from-cli"
    assert config.detector.confidence_threshold == 0.8
    assert config.detector.device == "cuda"
    assert config.tracker.lost_track_buffer > 0


@pytest.mark.parametrize("content", ["[detector\n", "[detector]\nunknown = 1\n"])
def test_bad_toml_and_unknown_fields_raise_configuration_error(tmp_path, content):
    path = tmp_path / "bad.toml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_pipeline_config(path)


def test_missing_toml_raises_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError):
        load_pipeline_config(tmp_path / "missing.toml")
