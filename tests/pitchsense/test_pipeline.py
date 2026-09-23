import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from pitchsense.cli import main
from pitchsense.config import load_pipeline_config
from pitchsense.contracts import (
    BoundingBox,
    Detection,
    ModelIdentity,
    Track,
    VideoMetadata,
)
from pitchsense.errors import EmptyVideoError, InputVideoError, OutputWriteError
from pitchsense.pipeline import PipelineRunner
from pitchsense.video.opencv import VideoReader, VideoWriter


class Detector:
    model_identity = ModelIdentity(
        backend="scripted",
        identifier="test-model",
        class_names=["player"],
        resolved_device="cpu",
    )

    def detect(self, frame):
        return [
            Detection(
                class_id=1,
                class_name="player",
                confidence=0.875,
                bbox=BoundingBox(x1=frame.index + 2, y1=3, x2=frame.index + 12, y2=18),
            )
        ]


class Tracker:
    def reset(self):
        self.next_id = 7

    def update(self, detections, frame):
        return [
            Track(
                track_id=self.next_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
            for detection in detections
        ]


def make_video(path: Path, frames: int = 8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()


def test_deterministic_video_pipeline(tmp_path):
    source = tmp_path / "input.avi"
    make_video(source)
    output = tmp_path / "run"
    artifacts = PipelineRunner(Detector(), Tracker(), load_pipeline_config()).run(
        source, output
    )

    records = [
        json.loads(line)
        for line in (output / artifacts.results_filename).read_text().splitlines()
    ]
    manifest = json.loads((output / artifacts.manifest_filename).read_text())
    assert len(records) == 8
    assert [record["frame_index"] for record in records] == list(range(8))
    assert [record["tracks"][0]["track_id"] for record in records] == [7] * 8
    assert [record["detections"][0]["bbox"]["x1"] for record in records] == list(
        range(2, 10)
    )
    assert records[0]["detections"][0]["bbox"] == {
        "x1": 2.0,
        "y1": 3.0,
        "x2": 12.0,
        "y2": 18.0,
    }
    assert {record["detections"][0]["class_name"] for record in records} == {"player"}
    assert manifest["processed_frame_count"] == 8
    assert manifest["video"]["width"] == 64
    assert manifest["video"]["height"] == 48
    assert manifest["video"]["fps"] == pytest.approx(12, abs=0.01)
    if manifest["video"]["reported_frame_count"] is not None:
        assert manifest["video"]["reported_frame_count"] == 8
    assert manifest["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["input_size_bytes"] == source.stat().st_size
    assert manifest["model_identity"]["identifier"] == "test-model"
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["input_filename"] == "input.avi"
    assert manifest["tracker_config"]["track_buffer"] == 30
    assert manifest["pipeline_config"]["output"]["codec"] == "MJPG"
    assert manifest["application_version"]
    assert manifest["python_version"]
    assert manifest["dependency_versions"]["pydantic"]
    assert (
        datetime.fromisoformat(manifest["started_at_utc"]).utcoffset().total_seconds()
        == 0
    )
    assert (
        datetime.fromisoformat(manifest["completed_at_utc"]).utcoffset().total_seconds()
        == 0
    )
    assert manifest["elapsed_seconds"] >= 0
    assert manifest["artifacts"] == artifacts.model_dump()
    assert all(
        "/" not in name and "\\" not in name for name in artifacts.model_dump().values()
    )
    assert not list(output.glob("*.partial*"))

    video = cv2.VideoCapture(str(output / artifacts.annotated_video_filename))
    assert video.isOpened()
    assert (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    ) == (64, 48)
    decoded, first_frame = video.read()
    assert decoded
    green = first_frame[:, :, 1].astype(int)
    red_blue = np.maximum(first_frame[:, :, 0], first_frame[:, :, 2]).astype(int)
    assert np.max(green - red_blue) > 70
    count = 1
    while video.read()[0]:
        count += 1
    video.release()
    assert count == 8


def test_reader_rejects_missing_and_empty_video(tmp_path):
    with pytest.raises(InputVideoError):
        with VideoReader(tmp_path / "missing.avi"):
            pass
    unreadable = tmp_path / "unreadable.avi"
    unreadable.write_text("not a video")
    with pytest.raises(InputVideoError):
        with VideoReader(unreadable):
            pass
    empty = tmp_path / "empty.avi"
    empty.touch()

    class EmptyCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return 12 if property_id == cv2.CAP_PROP_FPS else 64

        def read(self):
            return False, None

        def release(self):
            pass

    with pytest.raises(EmptyVideoError):
        with VideoReader(empty, capture_factory=lambda _: EmptyCapture()) as reader:
            list(reader)


@pytest.mark.parametrize(
    "bad_property",
    [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT],
)
def test_reader_rejects_invalid_metadata(tmp_path, bad_property):
    source = tmp_path / "input.avi"
    make_video(source)

    class BadCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return 0 if property_id == bad_property else 64

        def release(self):
            pass

    with pytest.raises(InputVideoError):
        with VideoReader(source, capture_factory=lambda _: BadCapture()):
            pass


def test_reader_allows_unreported_frame_count(tmp_path):
    source = tmp_path / "input.avi"
    source.touch()

    class Capture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return None if property_id == cv2.CAP_PROP_FRAME_COUNT else 12

        def release(self):
            pass

    with VideoReader(source, capture_factory=lambda _: Capture()) as reader:
        assert reader.metadata.reported_frame_count is None


def test_writer_rejects_open_failure_and_wrong_size(tmp_path):
    metadata = VideoMetadata(width=64, height=48, fps=12)

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            pass

    with pytest.raises(OutputWriteError):
        with VideoWriter(
            tmp_path / "bad.avi",
            metadata,
            "MJPG",
            writer_factory=lambda *args: ClosedWriter(),
        ):
            pass
    with VideoWriter(tmp_path / "good.avi", metadata, "MJPG") as writer:
        with pytest.raises(OutputWriteError):
            writer.write(np.zeros((47, 64, 3), dtype=np.uint8))

    class FailingWriter:
        def isOpened(self):
            return True

        def write(self, image):
            raise cv2.error("write failed")

        def release(self):
            pass

    with VideoWriter(
        tmp_path / "failure.avi",
        metadata,
        "MJPG",
        writer_factory=lambda *args: FailingWriter(),
    ) as writer:
        with pytest.raises(OutputWriteError, match="write failed"):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))


def test_mid_run_failure_leaves_no_manifest_or_final_artifacts(tmp_path):
    source = tmp_path / "input.avi"
    make_video(source)

    class FailingDetector(Detector):
        def detect(self, frame):
            if frame.index == 3:
                raise RuntimeError("backend failed")
            return super().detect(frame)

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="backend failed"):
        PipelineRunner(FailingDetector(), Tracker(), load_pipeline_config()).run(
            source, output
        )
    assert not list(output.glob("*.partial*"))
    assert not (output / "manifest.json").exists()
    assert not (output / "annotated.avi").exists()
    assert not (output / "results.jsonl").exists()


def test_refused_output_preserves_existing_artifacts(tmp_path):
    source = tmp_path / "input.avi"
    make_video(source)
    output = tmp_path / "run"
    output.mkdir()
    existing = output / "results.jsonl"
    existing.write_text("earlier run")
    with pytest.raises(OutputWriteError):
        PipelineRunner(Detector(), Tracker(), load_pipeline_config()).run(
            source, output
        )
    assert existing.read_text() == "earlier run"


def test_cli_injected_backends_use_overridden_toml_config(tmp_path):
    source = tmp_path / "input.avi"
    make_video(source, frames=1)
    toml = tmp_path / "config.toml"
    toml.write_text('[detector]\nmodel = "from-file"\nconfidence_threshold = 0.2\n')
    output = tmp_path / "run"
    assert (
        main(
            [
                "run",
                str(source),
                "--output",
                str(output),
                "--config",
                str(toml),
                "--model",
                "from-cli",
                "--confidence",
                "0.8",
                "--device",
                "cpu",
            ],
            backend_factory=lambda config: (Detector(), Tracker()),
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["pipeline_config"]["detector"] == {
        "backend": "rfdetr",
        "model": "from-cli",
        "confidence_threshold": 0.8,
        "device": "cpu",
    }
