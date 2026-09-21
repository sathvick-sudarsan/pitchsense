import json
import math

import pytest
from pydantic import ValidationError

from pitchsense.contracts import (
    BoundingBox,
    Detection,
    FrameResult,
    ModelIdentity,
    RunArtifacts,
    Track,
    VideoMetadata,
)
from pitchsense.errors import ConfigurationError, EmptyVideoError, PitchSenseError


def test_bounding_box_rejects_nonfinite_and_reversed_coordinates():
    for values in [(0, 0, 0, 1), (0, 2, 1, 1), (math.inf, 0, 1, 1)]:
        with pytest.raises(ValidationError):
            BoundingBox(x1=values[0], y1=values[1], x2=values[2], y2=values[3])


def test_detection_and_track_confidence_validation():
    bbox = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    for model in (Detection, Track):
        fields = {"class_id": 1, "class_name": "player", "bbox": bbox}
        if model is Track:
            fields["track_id"] = 7
        for invalid in (-0.01, 1.01, math.nan):
            with pytest.raises(ValidationError):
                model(**fields, confidence=invalid)


def test_video_metadata_rejects_invalid_dimensions_fps_and_frame_count():
    valid = {"width": 1920, "height": 1080, "fps": 30}
    for replacement in (
        {"width": 0},
        {"height": -1},
        {"fps": 0},
        {"fps": math.inf},
        {"reported_frame_count": -1},
    ):
        with pytest.raises(ValidationError):
            VideoMetadata(**(valid | replacement))
    assert VideoMetadata(**valid).reported_frame_count is None


def test_frame_result_round_trips_as_json():
    bbox = BoundingBox(x1=1, y1=2, x2=3, y2=4)
    frame = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        detections=[
            Detection(class_id=2, class_name="ball", confidence=0.9, bbox=bbox)
        ],
        tracks=[
            Track(track_id=4, class_id=2, class_name="ball", confidence=None, bbox=bbox)
        ],
    )
    assert FrameResult.model_validate_json(frame.model_dump_json()) == frame
    assert json.loads(frame.model_dump_json())["schema_version"] == "1.0.0"
    with pytest.raises(ValidationError):
        FrameResult(frame_index=0, timestamp_s=0, schema_version="2.0.0")


def test_model_identity_and_artifacts_validate_fields():
    identity = ModelIdentity(
        backend="test",
        identifier="model",
        class_names=["player"],
        resolved_device="cpu",
    )
    assert identity.sha256 is None
    with pytest.raises(ValidationError):
        ModelIdentity(
            backend=" ", identifier="model", class_names=[], resolved_device="cpu"
        )
    with pytest.raises(ValidationError):
        ModelIdentity(
            backend="test",
            identifier="model",
            class_names=["player"],
            resolved_device="cpu",
            sha256="bad",
        )
    artifacts = RunArtifacts(
        annotated_video_filename="annotated.mp4",
        results_filename="results.jsonl",
        manifest_filename="manifest.json",
    )
    assert artifacts.model_dump()["results_filename"] == "results.jsonl"
    with pytest.raises(ValidationError):
        RunArtifacts(
            annotated_video_filename="../video.mp4",
            results_filename="results.jsonl",
            manifest_filename="manifest.json",
        )


def test_domain_errors_share_base_class():
    assert issubclass(ConfigurationError, PitchSenseError)
    assert issubclass(EmptyVideoError, PitchSenseError)
